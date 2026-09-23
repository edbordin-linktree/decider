"""Experimental bounded parallel gated-delta prefill, with FP32 state."""

import torch


def inverse_unit_lower(matrix, size):
    """Block triangular inversion avoids unstable powers of correlated keys."""
    if size == 1:
        return torch.ones_like(matrix)
    half = size // 2
    diagonal = torch.stack(
        [matrix[..., :half, :half], matrix[..., half:size, half:size]], dim=-3
    )
    inverse = inverse_unit_lower(diagonal, half)
    left, right = inverse[..., 0, :, :], inverse[..., 1, :, :]
    lower = -(right @ matrix[..., half:size, :half]) @ left
    return torch.cat(
        [torch.cat([left, torch.zeros_like(left)], -1), torch.cat([lower, right], -1)],
        -2,
    )


def parallel_delta(q, k, v, log_g, beta, state):
    """Solve a <=64-token triangular update system by block inversion.

    Inputs use B,T,H,D; state uses B,H,Dk,Dv. Queries are already scaled.
    """
    length = q.shape[1]
    q, k, v = [
        torch.nn.functional.pad(x.transpose(1, 2), (0, 0, 0, 64 - length))
        for x in (q, k, v)
    ]
    log_g = torch.nn.functional.pad(log_g.transpose(1, 2), (0, 64 - length))
    beta = torch.nn.functional.pad(beta.transpose(1, 2), (0, 64 - length)).unsqueeze(-1)
    cumulative = log_g.cumsum(-1)
    positions = torch.arange(64)
    lower = positions[:, None] >= positions[None, :]
    strict = positions[:, None] > positions[None, :]
    decay = torch.where(
        lower, cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2), -float("inf")
    ).exp()
    a = torch.where(strict, (k * beta) @ k.transpose(-1, -2) * decay, 0.0)
    inverse = inverse_unit_lower(a, 64)
    exp_g = cumulative.exp().unsqueeze(-1)
    updates = inverse @ (beta * v - ((k * beta) * exp_g) @ state)
    output = (q * exp_g) @ state + ((q @ k.transpose(-1, -2)) * decay) @ updates
    tail_decay = (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    final = (
        cumulative[..., -1:].exp().unsqueeze(-1) * state
        + (k * tail_decay).transpose(-1, -2) @ updates
    )
    return output[:, :, :length].transpose(1, 2), final
