"""CPU algebra checks, including correlated keys that broke matrix powers."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "mlx_parallel_delta", Path(__file__).parents[1] / "scripts/mlx_parallel_delta.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("length", [2, 35, 63, 64])
@pytest.mark.parametrize("correlated", [False, True])
def test_parallel_delta_matches_sequential_updates(length, correlated):
    torch.manual_seed(32)
    q = torch.nn.functional.normalize(torch.randn(1, length, 2, 8), dim=-1)
    k = q if not correlated else q[:, :1].expand_as(q)
    v = torch.randn_like(q)
    log_g = -torch.rand(1, length, 2) * 0.01
    beta = torch.ones_like(log_g)
    state = torch.randn(1, 2, 8, 8) * 0.1
    out, final = module.parallel_delta(q, k, v, log_g, beta, state)
    expected = []
    for t in range(length):
        state = state * log_g[:, t].exp()[..., None, None]
        delta = v[:, t] - (k[:, t].unsqueeze(-2) @ state).squeeze(-2)
        state = state + k[:, t].unsqueeze(-1) * (beta[:, t, :, None] * delta).unsqueeze(
            -2
        )
        expected.append((q[:, t].unsqueeze(-2) @ state).squeeze(-2))
    torch.testing.assert_close(out, torch.stack(expected, dim=1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final, state, atol=1e-5, rtol=1e-5)
