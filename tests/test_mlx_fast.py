import torch
import json
import sys
from types import SimpleNamespace
import pytest
from decider.mlx_fast import FastScorer


def scorer(batch=6, max_prefix=512, max_suffix=256):
    calls = []

    def execute(inputs):
        calls.append(inputs)
        return [torch.zeros(batch, 255)]

    s = FastScorer.__new__(FastScorer)
    s.exports = [
        (
            {"batch": batch, "max_prefix": max_prefix, "max_suffix": max_suffix},
            None,
            SimpleNamespace(execute=execute),
        )
    ]
    return s, calls


def inputs(n=2, length=10):
    ids = torch.arange(length).repeat(n, 1)
    ids[:, 5] += torch.arange(n)
    return (
        ids,
        torch.ones_like(ids),
        torch.full((n,), length - 1),
        torch.arange(n),
        torch.full((n,), 2),
    )


def test_prefix_and_slot_offsets():
    s, calls = scorer()
    out = s.score(*inputs(), pad_id=0, limit=100)
    prefix, suffix, slots = calls[0]
    assert prefix.tolist() == [[0, 1, 2, 3, 4]]
    assert suffix.shape == (6, 5)
    assert slots[:2].tolist() == [4, 4]
    assert torch.isneginf(out[:, 2:]).all()
    assert out.shape == (2, 255)


def test_overflow_rejected_before_execution():
    s, calls = scorer()
    with pytest.raises(ValueError, match="10 tokens"):
        s.score(*inputs(), pad_id=0, limit=9)
    assert not calls


def test_large_suffix_falls_back_without_partial_execution():
    s, calls = scorer(batch=2, max_prefix=2, max_suffix=5)
    assert s.score(*inputs(n=4), pad_id=0, limit=100) is None
    assert not calls


def test_request_larger_than_batch_splits():
    s, calls = scorer(batch=2)
    assert s.score(*inputs(n=5), pad_id=0, limit=100).shape == (5, 255)
    assert len(calls) == 3


def test_no_common_prefix_falls_back():
    s, calls = scorer()
    args = list(inputs())
    args[0][1, 0] = 999
    assert s.score(*args, pad_id=0, limit=100) is None
    assert not calls


def test_smallest_sufficient_batch_selected():
    small, small_calls = scorer(batch=2)
    large, large_calls = scorer(batch=6)
    small.exports += large.exports
    small.score(*inputs(n=2), pad_id=0, limit=100)
    assert len(small_calls) == 1 and not large_calls
    small.score(*inputs(n=3), pad_id=0, limit=100)
    assert len(large_calls) == 1


def test_padding_is_not_part_of_prompt_limit():
    s, calls = scorer()
    args = list(inputs())
    args[1][:, 8:] = 0
    args[2][:] = 7
    s.score(*args, pad_id=0, limit=8)
    assert calls[0][1].shape == (6, 3)


def test_dynamic_batch_does_not_pad_unused_rows():
    s, calls = scorer()
    s.exports[0][0]["dynamic_batch"] = True
    s.score(*inputs(n=2), pad_id=0, limit=100)
    assert calls[0][1].shape == (2, 5)


def test_specialized_bounds_fall_back_to_general_fast_export():
    specialized, specialized_calls = scorer(max_suffix=4)
    general, general_calls = scorer()
    specialized.exports += general.exports
    assert specialized.score(*inputs(), pad_id=0, limit=100).shape == (2, 255)
    assert not specialized_calls and len(general_calls) == 1


def test_non_finite_logits_are_rejected():
    s, _ = scorer()
    s.exports[0][2].execute = lambda _: [torch.full((6, 255), float("nan"))]
    with pytest.raises(RuntimeError, match="non-finite"):
        s.score(*inputs(), pad_id=0, limit=100)


def test_compaction_retains_only_causal_tokens():
    s, calls = scorer()
    s.exports[0][0].update(compact=True, dynamic_batch=True)
    args = list(inputs())
    args[2][1] = 7
    s.score(*args, pad_id=0, limit=100)
    prefix, suffix, slots, keep = calls[0]
    assert prefix.shape == (1, 5)
    assert suffix.shape == (2, 5)
    assert slots.tolist() == [4, 2]
    assert keep.tolist() == list(range(13))


def test_loads_only_selected_artifact_once(tmp_path, monkeypatch):
    paths = [tmp_path / "specialized.pte", tmp_path / "general.pte"]
    for path, maximum in zip(paths, [4, 256]):
        path.write_bytes(b"test")
        path.with_suffix(".pte.json").write_text(
            json.dumps(
                {
                    "model": "test",
                    "format": "shared_prefix",
                    "batch": 6,
                    "max_prefix": 512,
                    "max_suffix": maximum,
                }
            )
        )
    loads = []
    method = SimpleNamespace(execute=lambda _: [torch.zeros(6, 255)])

    def load(path):
        loads.append(path)
        return SimpleNamespace(load_method=lambda _: method)

    runtime = SimpleNamespace(get=lambda: SimpleNamespace(load_program=load))
    monkeypatch.setitem(sys.modules, "executorch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(Runtime=runtime)
    )
    s = FastScorer(paths, "test")
    assert not loads
    for _ in range(2):
        s.score(*inputs(), pad_id=0, limit=100)
    assert loads == [str(paths[1])]
