"""Server backend selection without model downloads or Apple hardware."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_cuda_startup_without_executorch():
    script = '''
import asyncio, importlib.abc, importlib.util, sys
original_find_spec = importlib.util.find_spec
def find_spec(name, *args, **kw):
    if name.split(".")[0] in ("executorch", "mlx"):
        return None
    return original_find_spec(name, *args, **kw)
importlib.util.find_spec = find_spec
class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "executorch" or fullname.startswith("executorch.") or fullname == "mlx":
            raise ModuleNotFoundError("MLX intentionally unavailable")
sys.meta_path.insert(0, NoMLX())
from decider import serve
assert "decider.mlx_backend" not in sys.modules
assert serve.BACKEND == "cuda"
class Engine:
    def __init__(self, *a, **kw): self.cfg = {}; self.max_ctx = 128
    def warmup(self, shapes): return 0
serve.Engine = Engine
asyncio.run(serve._start())
assert isinstance(serve.eng, Engine)
assert "decider.mlx_backend" not in sys.modules
from decider.mlx_backend import MLXDecider
try:
    MLXDecider("unused.pte")
except RuntimeError as e:
    assert "requires an ExecuTorch installation" in str(e)
else:
    raise AssertionError("Missing optional backend was not reported")
'''
    env = dict(os.environ, DECIDER_BACKEND="cuda", DECIDER_MODEL="nonexistent-test-model")
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], env=env, check=True)


def test_cli_selects_backend(monkeypatch):
    from decider import serve
    import uvicorn
    monkeypatch.setattr(serve, "BACKEND", "cuda")
    monkeypatch.setattr(serve, "MODEL", "old")
    monkeypatch.setattr(sys, "argv", ["decider.serve", "--backend", "mlx", "--model", "ready.pte", "--port", "8123"])
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((serve.BACKEND, serve.MODEL, kw)))
    serve.main()
    assert calls == [("mlx", "ready.pte", {"host": "127.0.0.1", "port": 8123})]


def test_mlx_request_forwards_options_and_validation(monkeypatch):
    from decider import serve
    from fastapi import HTTPException
    calls = []
    def evaluate(*args):
        calls.append(args)
        if not args[1]:
            raise ValueError("At least one question is required")
        return {"answers": {"x": {"noul": 0.5}}}
    monkeypatch.setattr(serve, "BACKEND", "mlx")
    monkeypatch.setattr(serve, "eng", SimpleNamespace(decider=SimpleNamespace(evaluate=evaluate)))
    request = serve.S1Req(state="s", questions={"x": {}}, max_prompt_tokens=8192,
                          image_base64="image", independent=False, layout="state_first")
    assert asyncio.run(serve.systemone(request))["answers"]["x"]["noul"] == 0.5
    assert calls == [("s", {"x": {}}, 8192, "image", False, "state_first")]
    with pytest.raises(HTTPException) as exc:
        asyncio.run(serve.systemone(serve.S1Req(state="s", questions={})))
    assert exc.value.status_code == 422


def test_engine_scores_rows_and_rejects_overflow():
    import torch
    from decider.mlx_backend import MLXDecider
    d = MLXDecider.__new__(MLXDecider)
    d.length = d.request_limit = 256
    d.dynamic = True
    d.alignment = 128
    d.features = None
    d.m = SimpleNamespace(tok=SimpleNamespace(pad_token_id=0))
    seen = []
    def execute(ids, slots):
        seen.append((ids.shape, slots.tolist()))
        return torch.zeros(1, 255)
    d.execute = execute
    ids = torch.ones(1, 129, dtype=torch.long)
    out = d.slot_logits(ids, ids, torch.tensor([128]), torch.tensor([0]), torch.tensor([2]))
    assert seen == [(torch.Size([1, 256]), [128])]
    assert out.shape == (1, 255) and torch.isneginf(out[0, 2:]).all()
    d.request_limit = 128
    with pytest.raises(ValueError, match="129 tokens"):
        d.slot_logits(ids, ids, torch.tensor([128]), torch.tensor([0]), torch.tensor([2]))


def test_standalone_compact_export_is_lazy_and_never_falls_back(tmp_path, monkeypatch):
    import json
    import torch
    import decider.mlx_backend as backend
    artifact = tmp_path / "compact.pte"
    artifact.write_bytes(b"test fixture")
    metadata = dict(model="test", format="shared_prefix", length=575, dynamic=True,
                    batch=6, max_prefix=512, max_suffix=63, compact=True,
                    tokenizer="tokenizer", decider_config={"temperature": 1.3})
    Path(str(artifact) + ".json").write_text(json.dumps(metadata))
    loads = []
    runtime = SimpleNamespace(backend_registry=SimpleNamespace(registered_backend_names=["MLXBackend"]),
                              load_program=lambda path: loads.append(path))
    monkeypatch.setitem(sys.modules, "executorch.runtime", SimpleNamespace(
        Runtime=SimpleNamespace(get=lambda: runtime)))
    tokenizers = []
    def tokenizer(path):
        tokenizers.append(path)
        return SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(backend.AutoTokenizer, "from_pretrained", tokenizer)
    d = backend.MLXDecider(artifact)
    assert d.compact_only and d.T == 1.3 and not loads
    assert tokenizers == [str(tmp_path / "tokenizer")]
    assert len(d.fast_scorer.exports) == 1
    with pytest.raises(ValueError, match="no reference fallback"):
        d.execute(torch.ones(1, 2), torch.tensor([1]))
    assert not loads


def test_public_compact_package_preserves_weights_and_calibration(tmp_path):
    import hashlib
    import json
    from scripts.package_mlx_fast import package
    export = tmp_path / 'compact.pte'
    export.write_bytes(b'fixture weights')
    Path(str(export) + '.json').write_text(json.dumps(dict(model='Mapika/test',
        format='shared_prefix', compact=True, dtype='float16', bits=None,
        max_prefix=512, max_suffix=63, batch=6)))
    tokenizer = tmp_path / 'tokenizer'
    tokenizer.mkdir()
    (tokenizer / 'tokenizer.json').write_text('{}')
    reference = tmp_path / 'reference.pte.json'
    reference.write_text(json.dumps(dict(model='Mapika/test', tokenizer='tokenizer',
                                        decider_config={'temperature': 1.3})))
    output = tmp_path / 'publish'
    package(export, reference, output, 'a' * 40, 'https://github.com/example/repo/tree/commit')
    metadata = json.loads((output / 'model.pte.json').read_text())
    assert metadata['length'] == 575
    assert metadata['decider_config']['temperature'] == 1.3
    assert metadata['tokenizer'] == 'model.pte.tokenizer'
    assert (output / 'model.pte').read_bytes() == export.read_bytes()
    assert str(tmp_path) not in (output / 'provenance.json').read_text()
    for line in (output / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match='overwrite'):
        package(export, reference, output, 'a' * 40, 'https://example.com')
