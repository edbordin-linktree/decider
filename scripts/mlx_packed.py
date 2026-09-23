"""Shared-prefix tree scoring with one set of projections per decoder layer."""

import types
import torch
from export_mlx import patch_recurrent_modules
import executorch.backends.mlx.custom_ops


class PackedScorer(torch.nn.Module):
    def __init__(
        self,
        decision_model,
        batch,
        fuse=False,
        parallel_suffix=False,
        parallel_prefix=False,
        compact=False,
    ):
        super().__init__()
        self.core = decision_model.lm.model
        self.batch = batch
        self.register_buffer(
            "head",
            decision_model.lm.lm_head.weight[decision_model.letters].detach().clone(),
        )
        patch_recurrent_modules(self)
        for m in self.core.modules():
            if hasattr(m, "mlx_state"):
                m.register_buffer(
                    "branch_state", torch.zeros(batch, *m.mlx_state.shape[1:])
                )
                m.forward = types.MethodType(
                    compact_recurrent if compact else packed_recurrent, m
                )
                m.branches = batch
                m.parallel_suffix = parallel_suffix
                m.parallel_prefix = parallel_prefix
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("decider_tree", tree_attention)
        self.core.config._attn_implementation = "decider_tree"
        if fuse:
            for layer in self.core.layers:
                mlp = layer.mlp
                mlp.gate_up_proj = combine(mlp.gate_proj, mlp.up_proj)
                del mlp.gate_proj, mlp.up_proj
                mlp.forward = types.MethodType(fused_mlp, mlp)
                if hasattr(layer, "linear_attn"):
                    m = layer.linear_attn
                    m.in_proj_all = combine(
                        m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a
                    )
                    del m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a

    def forward(self, prefix, suffix, slots, keep=None):
        P, S = prefix.shape[1], suffix.shape[1]
        N = suffix.shape[0]
        if keep is not None:
            grid = torch.arange(P + N * S)
            ranks = (grid[:, None] >= keep[None, :]).sum(1) - 1
            restore = torch.where(keep[ranks] == grid, ranks, keep.shape[0])
        for layer in self.core.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.prefix_length = P
                layer.linear_attn.suffix_length = S
                layer.linear_attn.branches = N
                if keep is not None:
                    layer.linear_attn.compact_keep = keep
                    layer.linear_attn.compact_restore = restore
        ids = torch.cat([prefix, suffix.reshape(1, -1)], dim=1)
        positions = torch.cat([torch.arange(P), torch.arange(P, P + S).repeat(N)])
        groups = torch.cat([torch.full((P,), -1), torch.arange(N).repeat_interleave(S)])
        if keep is not None:
            ids = ids.index_select(1, keep)
            positions = positions.index_select(0, keep)
            groups = groups.index_select(0, keep)
        visible = (groups[None, :] == -1) | (groups[:, None] == groups[None, :])
        mask = (
            (visible & (positions[None, :] <= positions[:, None]))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        h = self.core(
            input_ids=ids,
            position_ids=positions.unsqueeze(0),
            attention_mask={"full_attention": mask, "linear_attention": None},
            use_cache=False,
        ).last_hidden_state
        indices = (
            P + torch.arange(N) * S + slots
            if keep is None
            else P + (slots + 1).cumsum(0) - 1
        )
        return torch.nn.functional.linear(h[0, indices], self.head).float()


def tree_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    out = torch.ops.mlx.custom_sdpa(
        query,
        key,
        value,
        start_pos=0,
        attn_mask=attention_mask,
        is_causal=False,
        scale=scaling,
    )
    return out.transpose(1, 2).contiguous(), None


def combine(*modules):
    assert all(m.bias is None for m in modules)
    linear = torch.nn.Linear(
        modules[0].in_features,
        sum(m.out_features for m in modules),
        bias=False,
        dtype=modules[0].weight.dtype,
        device="meta",
    )
    linear.weight = torch.nn.Parameter(
        torch.cat([m.weight.detach() for m in modules], dim=0), requires_grad=False
    )
    return linear


def fused_mlp(self, x):
    gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


def fuse_norms(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5RMSNorm,
        Qwen3_5RMSNormGated,
    )

    for module in model.modules():
        if isinstance(module, Qwen3_5RMSNorm):
            module.register_buffer("mlx_scale", 1 + module.weight.detach().float())
            module.forward = types.MethodType(fused_norm, module)
        elif isinstance(module, Qwen3_5RMSNormGated):
            module.forward = types.MethodType(fused_gated_norm, module)


def fused_norm(self, x):
    return torch.nn.functional.rms_norm(
        x.float(), (x.shape[-1],), self.mlx_scale, self.eps
    ).to(x.dtype)


def fused_gated_norm(self, x, gate):
    normalized = torch.nn.functional.rms_norm(
        x.float(), (x.shape[-1],), eps=self.variance_epsilon
    ).to(x.dtype)
    return (self.weight * normalized * torch.nn.functional.silu(gate.float())).to(
        x.dtype
    )


def compact_recurrent(self, hidden_states, **kwargs):
    return packed_recurrent(self, hidden_states, compact=True, **kwargs)


def packed_recurrent(
    self, hidden_states, cache_params=None, attention_mask=None, compact=False, **kwargs
):
    P, S, N = self.prefix_length, self.suffix_length, self.branches
    L = P + N * S
    if hasattr(self, "in_proj_all"):
        qkv, z, b, a = self.in_proj_all(hidden_states).split(
            [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads], dim=-1
        )
    else:
        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
    if compact:

        def restore(x):
            padded = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
            return padded.index_select(1, self.compact_restore)

        qkv, b, a = restore(qkv), restore(b), restore(a)
    qkv = qkv.transpose(1, 2)
    pre = qkv[:, :, :P]
    suf = qkv[:, :, P:L].reshape(self.conv_dim, N, S).permute(1, 0, 2)
    history = self.conv1d.kernel_size[0] - 1
    prefix_history = torch.nn.functional.pad(pre, (history, 0))[:, :, -history:].expand(
        N, -1, -1
    )
    pre_out = self.conv1d(pre)[..., :P]
    suf_out = torch.nn.functional.conv1d(
        torch.cat([prefix_history, suf], dim=-1),
        self.conv1d.weight,
        self.conv1d.bias,
        groups=self.conv1d.groups,
    )
    mixed = torch.cat(
        [pre_out, suf_out.permute(1, 0, 2).reshape(1, self.conv_dim, N * S)], dim=-1
    )
    mixed = torch.nn.functional.silu(mixed).transpose(1, 2)
    q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], -1)
    q = q.reshape(1, L, self.num_k_heads, self.head_k_dim).float()
    k = k.reshape(1, L, self.num_k_heads, self.head_k_dim).float()
    v = v.reshape(1, L, self.num_v_heads, self.head_v_dim).float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * self.head_k_dim**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    beta = b.sigmoid().float()
    log_g = -self.A_log.float().exp() * torch.nn.functional.softplus(
        a.float() + self.dt_bias
    )
    g = log_g.exp()
    if self.parallel_prefix:
        from mlx_parallel_delta import parallel_delta

        out_pre, state = parallel_delta(
            q[:, :P],
            k[:, :P],
            v[:, :P],
            log_g[:, :P],
            beta[:, :P],
            torch.zeros_like(self.mlx_state),
        )
    else:
        self.mlx_state.zero_()
        out_pre = torch.ops.mlx.gated_delta_rule(
            q[:, :P], k[:, :P], v[:, :P], g[:, :P], beta[:, :P], self.mlx_state
        )
        state = self.mlx_state.transpose(-1, -2)

    def branch(x):
        return x[:, P:L].reshape(N, S, *x.shape[2:]).contiguous()

    if self.parallel_suffix:
        from mlx_parallel_delta import parallel_delta

        state = state.expand(N, -1, -1, -1)
        out_suf, _ = parallel_delta(
            branch(q), branch(k), branch(v), branch(log_g), branch(beta), state
        )
    else:
        self.branch_state.copy_(self.mlx_state.expand(N, -1, -1, -1))
        out_suf = torch.ops.mlx.gated_delta_rule(
            branch(q), branch(k), branch(v), branch(g), branch(beta), self.branch_state
        )
    out = torch.cat(
        [out_pre, out_suf.reshape(1, N * S, self.num_v_heads, self.head_v_dim)], dim=1
    )
    if compact:
        out = out.index_select(1, self.compact_keep)
    z = z.reshape(-1, self.head_v_dim)
    out = self.norm(out.to(hidden_states.dtype).reshape(-1, self.head_v_dim), z)
    return self.out_proj(out.reshape(1, -1, self.value_dim))
