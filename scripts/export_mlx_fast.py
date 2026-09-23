"""Export a bounded shared-prefix scorer for low-latency requests."""

import argparse
import json
from pathlib import Path
import types
import gc
import torch
from export_mlx import patch_recurrent_modules
from decider.infer import Decider
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime
import executorch.backends.mlx.custom_ops


class SharedScorer(torch.nn.Module):
    def __init__(self, decision_model, batch):
        super().__init__()
        self.core = decision_model.lm.model
        self.register_buffer(
            "head",
            decision_model.lm.lm_head.weight[decision_model.letters].detach().clone(),
        )
        self.batch = batch
        patch_recurrent_modules(self)
        self.recurrent = [m for m in self.core.modules() if hasattr(m, "mlx_state")]
        for module in self.recurrent:
            module.register_buffer(
                "branch_state", torch.zeros(batch, *module.mlx_state.shape[1:])
            )
            module.forward = types.MethodType(recurrent_forward, module)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("decider_shared", shared_attention)
        self.core.config._attn_implementation = "decider_shared"

    def forward(self, prefix, suffix, slots):
        # Scratch tensors remain inside one delegated request, never on the host.
        for layer in self.core.layers:
            module = (
                layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
            )
            module.prefix_phase = True
        self.core(
            input_ids=prefix,
            attention_mask={"full_attention": None, "linear_attention": None},
            use_cache=False,
        )
        for layer in self.core.layers:
            module = (
                layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
            )
            module.prefix_phase = False
        positions = (
            (torch.arange(suffix.shape[1], device=suffix.device) + prefix.shape[1])
            .unsqueeze(0)
            .expand(self.batch, -1)
        )
        h = self.core(
            input_ids=suffix,
            position_ids=positions,
            attention_mask={"full_attention": None, "linear_attention": None},
            use_cache=False,
        ).last_hidden_state
        return torch.nn.functional.linear(
            h[torch.arange(self.batch), slots], self.head
        ).float()


def recurrent_forward(
    self, hidden_states, cache_params=None, attention_mask=None, **kwargs
):
    B, T, _ = hidden_states.shape
    qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    history = self.conv1d.kernel_size[0] - 1
    if self.prefix_phase:
        self.prefix_conv = torch.nn.functional.pad(qkv, (history, 0))[:, :, -history:]
        mixed = self.conv1d(qkv)[..., :T]
        self.mlx_state.zero_()
        state = self.mlx_state
    else:
        joined = torch.cat([self.prefix_conv.expand(B, -1, -1), qkv], dim=-1)
        mixed = torch.nn.functional.conv1d(
            joined, self.conv1d.weight, self.conv1d.bias, groups=self.conv1d.groups
        )
        self.branch_state.copy_(self.mlx_state.expand(B, -1, -1, -1))
        state = self.branch_state
    qkv = torch.nn.functional.silu(mixed).transpose(1, 2)
    q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], -1)
    q = q.reshape(B, T, self.num_k_heads, self.head_k_dim).float()
    k = k.reshape(B, T, self.num_k_heads, self.head_k_dim).float()
    v = v.reshape(B, T, self.num_v_heads, self.head_v_dim).float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * self.head_k_dim**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    beta = self.in_proj_b(hidden_states).sigmoid().float()
    g = (
        -self.A_log.float().exp()
        * torch.nn.functional.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )
    ).exp()
    out = torch.ops.mlx.gated_delta_rule(q, k, v, g, beta, state)
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    out = self.norm(out.to(hidden_states.dtype).reshape(-1, self.head_v_dim), z)
    return self.out_proj(out.reshape(B, T, self.value_dim))


def shared_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    if module.prefix_phase:
        module.prefix_key, module.prefix_value = key, value
        start = 0
    else:
        start = module.prefix_key.shape[2]
        key = torch.cat(
            [module.prefix_key.expand(key.shape[0], -1, -1, -1), key], dim=2
        )
        value = torch.cat(
            [module.prefix_value.expand(value.shape[0], -1, -1, -1), value], dim=2
        )
    out = torch.ops.mlx.custom_sdpa(
        query, key, value, start_pos=start, is_causal=True, scale=scaling
    )
    return out.transpose(1, 2).contiguous(), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Mapika/decider-2b")
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--bits", type=int, choices=[4, 8])
    p.add_argument("--packed", action="store_true")
    p.add_argument("--fuse-projections", action="store_true")
    p.add_argument("--parallel-suffix", action="store_true")
    p.add_argument("--parallel-prefix", action="store_true")
    p.add_argument("--dynamic-batch", action="store_true")
    p.add_argument("--fuse-norms", action="store_true")
    p.add_argument("--compact", action="store_true")
    args = p.parse_args()
    if args.parallel_suffix and not args.packed:
        p.error("--parallel-suffix requires --packed")
    if args.compact and not args.packed:
        p.error("--compact requires --packed")
    if args.parallel_prefix and not args.parallel_suffix:
        p.error("--parallel-prefix requires --parallel-suffix")
    if args.dynamic_batch and not args.parallel_suffix:
        p.error("--dynamic-batch requires --parallel-suffix")
    # Keep at least one padding position in the 64-token parallel solve so
    # export does not specialize the zero-padding boundary separately.
    max_suffix = 63 if args.parallel_suffix else 256
    max_prefix = 63 if args.parallel_prefix else 512
    torch.set_num_threads(8)
    d = Decider(args.model, device="cpu", dtype=torch.float32, use_graphs=False)
    config_path = Path(args.model) / "decider_config.json"
    if not config_path.is_file():
        from huggingface_hub import hf_hub_download
        config_path = Path(hf_hub_download(args.model, "decider_config.json"))
    decider_config = json.loads(config_path.read_text())
    # Qwen3.5's MLX casting recipe preserves the recurrent decay parameters.
    for name, parameter in d.m.named_parameters():
        if not name.endswith("A_log"):
            parameter.data = parameter.data.to(getattr(torch, args.dtype))
    tok = d.m.tok
    text = tok.encode(
        "Context: My card was charged twice. Please refund it.",
        add_special_tokens=False,
    )
    question = tok.encode(
        "\nWhich team?\n(A) billing\n(B) technical\nAnswer: (", add_special_tokens=False
    )
    prefix = torch.tensor([text])
    suffix = torch.tensor([question] * args.batch)
    slots = torch.full((args.batch,), len(question) - 1, dtype=torch.long)
    with torch.no_grad():
        joined = torch.cat([prefix.expand(args.batch, -1), suffix], dim=1)
        ref_h = d.m.lm.model(input_ids=joined, use_cache=False).last_hidden_state
        ref = torch.nn.functional.linear(
            ref_h[:, -1], d.m.lm.lm_head.weight[d.m.letters]
        ).float()
        if args.packed:
            from mlx_packed import PackedScorer

            scorer = PackedScorer(
                d.m,
                args.batch,
                fuse=args.fuse_projections,
                parallel_suffix=args.parallel_suffix,
                parallel_prefix=args.parallel_prefix,
                compact=args.compact,
            ).eval()
        else:
            scorer = SharedScorer(d.m, args.batch).eval()
        if args.fuse_norms:
            from mlx_packed import fuse_norms

            fuse_norms(scorer)
        inputs = (prefix, suffix, slots)
        if args.compact:
            inputs += (torch.arange(prefix.shape[1] + args.batch * suffix.shape[1]),)
        actual = scorer(*inputs)
        torch.testing.assert_close(actual, ref, atol=0.04, rtol=0.01)
        print("Prefix eager parity", (actual - ref).abs().max().item(), flush=True)
        if args.bits:
            from executorch.extension.llm.export.quantize import quantize_model_

            quantize_model_(
                scorer,
                qlinear_config=f"{args.bits}w",
                qlinear_group_size=32 if args.bits == 4 else 128,
                qembedding_config="8w",
                qembedding_group_size=128,
            )
        batch_dim = (
            torch.export.Dim("rows", min=1, max=args.batch)
            if args.dynamic_batch
            else args.batch
        )
        shapes = (
            {1: torch.export.Dim("prefix_tokens", min=2, max=max_prefix)},
            {0: batch_dim, 1: torch.export.Dim("suffix_tokens", min=2, max=max_suffix)},
            {0: batch_dim},
        )
        if args.compact:
            shapes += ({0: torch.export.Dim("compact_tokens", min=2, max=2048)},)
        ep = torch.export.export(scorer, inputs, strict=False, dynamic_shapes=shapes)
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[MLXPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        delegates = [
            n
            for n in edge.exported_program().graph.nodes
            if "executorch_call_delegate" in str(n.target)
        ]
        if len(delegates) != 1:
            raise RuntimeError(f"Expected one MLX partition, found {len(delegates)}")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path = Path(str(output) + ".tokenizer")
        tok.save_pretrained(tokenizer_path)
        with output.open("wb") as f:
            edge.to_executorch().write_to_file(f)
        temperature = d.T
        del edge, ep, scorer, d, ref_h
        gc.collect()
        program = Runtime.get().load_program(str(output))
        method = program.load_method("forward")
        out = method.execute(list(inputs))[0]
        err = (
            (
                torch.softmax(out[:, :2] / temperature, -1)
                - torch.softmax(ref[:, :2] / temperature, -1)
            )
            .abs()
            .max()
            .item()
        )
        print("Runtime probability error", err, flush=True)
        assert err < (0.03 if args.bits else 0.005)
        Path(str(output) + ".json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "length": max_prefix + max_suffix,
                    "dynamic": True,
                    "alignment": 1,
                    "tokenizer": tokenizer_path.name,
                    "decider_config": decider_config,
                    "format": "shared_prefix",
                    "batch": args.batch,
                    "max_prefix": max_prefix,
                    "max_suffix": max_suffix,
                    "dtype": args.dtype,
                    "bits": args.bits,
                    "packed": args.packed,
                    "parallel_prefix": args.parallel_prefix,
                    "dynamic_batch": args.dynamic_batch,
                    "parallel_suffix": args.parallel_suffix,
                    "fused_norms": args.fuse_norms,
                    "compact": args.compact,
                    "fused_projections": args.fuse_projections,
                }
            )
        )


if __name__ == "__main__":
    main()
