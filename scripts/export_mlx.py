"""Export Decider's stateless scoring forward to ExecuTorch/MLX."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from decider.infer import Decider, Example, Q
from decider.prompt import build
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime


def patch_recurrent_modules(model):
    """Expose state as a buffer so the delegate recognizes mutation; reset each request."""
    import types
    import executorch.backends.mlx.custom_kernel_ops.gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        assert cache_params is None and attention_mask is None
        B, T, _ = hidden_states.shape
        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        qkv = torch.nn.functional.silu(self.conv1d(qkv)[..., :T]).transpose(1, 2)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], -1)
        q = q.reshape(B, T, self.num_k_heads, self.head_k_dim).float()
        k = k.reshape(B, T, self.num_k_heads, self.head_k_dim).float()
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * self.head_k_dim**-0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = (-self.A_log.float().exp() * torch.nn.functional.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias)).exp()
        self.mlx_state.zero_()
        out = torch.ops.mlx.gated_delta_rule(q, k, v.float(), g, beta.float(), self.mlx_state)
        z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
        out = self.norm(out.to(hidden_states.dtype).reshape(-1, self.head_v_dim), z)
        return self.out_proj(out.reshape(B, T, self.value_dim))

    for layer in model.modules():
        if isinstance(layer, Qwen3_5GatedDeltaNet):
            layer.register_buffer("mlx_state", torch.zeros(1, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim))
            layer.forward = types.MethodType(forward, layer)


def patch_attention(model):
    """Bound the attention score matrix to 128 queries at a time."""
    from mlx_attention import attention as chunk_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
        assert attention_mask is None
        out = chunk_attention(query, key, value, scaling)
        return out.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register("decider_mlx", attention)
    model.core.config._attn_implementation = "decider_mlx"


class ScoringForward(torch.nn.Module):
    def __init__(self, decision_model, length):
        super().__init__()
        self.core = decision_model.lm.model
        self.register_buffer("head", decision_model.lm.lm_head.weight[decision_model.letters].detach().clone())
        assert self.core.config._attn_implementation == "sdpa"

    def forward(self, ids, slot):
        h = self.core(input_ids=ids, attention_mask={"full_attention": None, "linear_attention": None},
                      use_cache=False).last_hidden_state
        return torch.nn.functional.linear(h[0, slot], self.head).float()


class Keep:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Mapika/decider-0.8b")
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--output", required=True, help="Output .pte path; tokenizer and metadata are saved beside it")
    p.add_argument("--dynamic", action="store_true", help="Allow variable input lengths up to --length")
    args = p.parse_args()
    if args.length < 256 or args.length % 128:
        p.error("--length must be a multiple of 128, at least 256")
    if args.model == "Mapika/decider-2b-vision":
        from mlx_vision import main as export_vision
        export_vision(args.model, args.output, args.length)
        return
    torch.set_num_threads(8)
    print("Loading", args.model, flush=True)
    d = Decider(args.model, device="cpu", dtype=torch.float32, use_graphs=False)
    item = build(Example("My card was charged twice.", [Q("Which team?", ["billing", "technical"])]),
                 d.m.tok, Keep(), max_ctx_tokens=64)
    sample_length = 256 if args.dynamic else args.length
    assert len(item["ids"]) <= sample_length <= args.length
    ids = torch.full((1, sample_length), d.m.tok.pad_token_id, dtype=torch.long)
    ids[0, :len(item["ids"])] = torch.tensor(item["ids"])
    slot = torch.tensor(item["slots"], dtype=torch.long)
    inputs = (ids, slot)
    model = ScoringForward(d.m, args.length).eval()
    with torch.no_grad():
        print("Reference forward", flush=True)
        ref = model(*inputs)
        patch_recurrent_modules(model)
        patch_attention(model)
        transformed = model(*inputs)
        torch.testing.assert_close(transformed, ref, atol=0.02, rtol=0.01)
        print("Transform parity", (transformed-ref).abs().max().item(), flush=True)
        print("Exporting", flush=True)
        dynamic_shapes = ({1: 128 * torch.export.Dim("blocks", min=1, max=args.length // 128)}, None) if args.dynamic else None
        ep = torch.export.export(model, inputs, strict=False, dynamic_shapes=dynamic_shapes)
        print("Lowering", flush=True)
        edge = to_edge_transform_and_lower(ep, partitioner=[MLXPartitioner()],
                                          compile_config=EdgeCompileConfig(_check_ir_validity=False))
        et = edge.to_executorch()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            et.write_to_file(f)
        print("Saved", out, flush=True)
        runtime = Runtime.get()
        assert "MLXBackend" in runtime.backend_registry.registered_backend_names
        program = runtime.load_program(str(out))
        method = program.load_method("forward")
        t = time.perf_counter()
        actual = method.execute(list(inputs))[0]
        elapsed = time.perf_counter() - t
        torch.testing.assert_close(actual, ref, atol=0.02, rtol=0.01)
        from huggingface_hub import hf_hub_download
        cfg = json.loads(Path(hf_hub_download(args.model, "decider_config.json")).read_text())
        d.m.tok.save_pretrained(str(out) + ".tokenizer")
        Path(str(out) + ".json").write_text(json.dumps({"model": args.model, "length": args.length,
            "dynamic": args.dynamic, "alignment": 128, "decider_config": cfg,
            "tokenizer": out.name + ".tokenizer"}))
        print(json.dumps({"max_logit_error": (actual-ref).abs().max().item(),
                          "seconds": elapsed, "temperature": d.T,
                          "reference_probs": torch.softmax(ref[0, :2]/d.T, -1).tolist(),
                          "mlx_probs": torch.softmax(actual[0, :2]/d.T, -1).tolist()}), flush=True)


if __name__ == "__main__":
    main()
