"""Bound causal attention memory using an MLX scan over query blocks."""
import torch
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid, ItemIntNode, ScanNode, SdpaNode, SliceNode,
)

BLOCK = 128


@torch.library.custom_op("decider::chunked_attention", mutates_args=())
def chunked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      ends: torch.Tensor, scale: float) -> torch.Tensor:
    results = []
    for i, end in enumerate(ends.tolist()):
        # PyTorch's causal mask is upper-left; MLX's is lower-right.
        mask = torch.arange(end, device=q.device)[None, :] <= torch.arange(
            end - BLOCK, end, device=q.device)[:, None]
        results.append(torch.nn.functional.scaled_dot_product_attention(
            q[i], k[:, :, :end], v[:, :, :end], attn_mask=mask,
            scale=scale, enable_gqa=True))
    return torch.stack(results)


@chunked_attention.register_fake
def fake(q, k, v, ends, scale):
    return torch.empty_like(q)


@REGISTRY.register(target=[torch.ops.decider.chunked_attention.default])
def emit_attention(P, n):
    q, k, v, ends, scale = P.args(n)
    out = P.make_or_get_slot(n)
    _, query = P.make_tmp_slot()
    _, end_tensor = P.make_tmp_slot()
    _, end = P.make_tmp_value_slot()
    _, keys = P.make_tmp_slot()
    _, values = P.make_tmp_slot()
    tid = P.slot_to_tid
    with P.new_chain() as body:
        P.emit(ItemIntNode(x=tid(end_tensor), out=P.slot_to_vid(end)))
        for source, target in [(k, keys), (v, values)]:
            P.emit(SliceNode(x=tid(source), out=tid(target),
                            axis=IntOrVid.from_literal(2), start=IntOrVid.from_literal(0),
                            stop=P.to_int_or_vid(end)))
        P.emit(SdpaNode(q=tid(query), k=tid(keys), v=tid(values), out=tid(out),
                        scale=scale, causal=True))
    P.emit(ScanNode(originals=[tid(q), tid(ends)], sliced=[tid(query), tid(end_tensor)],
                    outputs=[tid(out)], carry=[], scan_axis=0, body_chain_idx=body))
    return out


def attention(query, key, value, scale):
    batch, heads, tokens, dim = query.shape
    q = query.reshape(batch, heads, tokens // BLOCK, BLOCK, dim).permute(2, 0, 1, 3, 4)
    ends = torch.arange(BLOCK, tokens + 1, BLOCK, device=query.device)
    out = chunked_attention(q, key, value, ends, scale)
    return out.permute(1, 2, 0, 3, 4).reshape(batch, heads, tokens, dim)
