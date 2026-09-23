"""Hub/local loading contracts without network access or model weights."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from decider import mlx_backend as backend


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    artifact = tmp_path / "model.pte"
    artifact.write_bytes(b"fixture")
    Path(str(artifact) + ".json").write_text("{}")
    monkeypatch.setattr(backend.MLXDecider, "__init__",
                        lambda self, path: setattr(self, "artifact", path))
    return artifact


@pytest.mark.parametrize("offline", [False, True])
def test_hub_load_forwards_download_options(bundle, monkeypatch, offline):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(bundle.parent)
    monkeypatch.setattr(backend, "snapshot_download", download)
    model = backend.MLXDecider.from_pretrained("owner/model", revision="pinned-commit",
        cache_dir="cache", local_files_only=offline, token=False)
    assert model.artifact == bundle
    assert calls == [dict(repo_id="owner/model", revision="pinned-commit",
                          cache_dir="cache", local_files_only=offline, token=False)]


@pytest.mark.parametrize("directory", [False, True])
def test_local_load_never_downloads(bundle, monkeypatch, directory):
    def forbidden(**kwargs):
        raise AssertionError("Local loading must not use the Hub")
    monkeypatch.setattr(backend, "snapshot_download", forbidden)
    model = backend.MLXDecider.from_pretrained(bundle.parent if directory else bundle)
    assert model.artifact == bundle


def test_missing_sidecar_is_clear(bundle):
    Path(str(bundle) + ".json").unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        backend.MLXDecider.from_pretrained(bundle.parent)


@pytest.mark.parametrize("filename", ["../outside.pte", "/outside.pte"])
def test_filename_stays_inside_bundle(bundle, filename):
    with pytest.raises(ValueError, match="relative path"):
        backend.MLXDecider.from_pretrained(bundle.parent, filename=filename)


def test_call_uses_validated_evaluate_path(bundle, monkeypatch):
    calls = []
    def evaluate(self, state, questions, **kwargs):
        calls.append((state, questions, kwargs))
        return {"answers": {}}
    monkeypatch.setattr(backend.MLXDecider, "evaluate", evaluate)
    model = backend.MLXDecider.from_pretrained(bundle)
    assert model({"ticket": "refund"}, {"team": {}}, limit=512) == {"answers": {}}
    assert calls == [({"ticket": "refund"}, {"team": {}}, {"limit": 512})]


def test_server_engine_uses_hub_loader(monkeypatch):
    calls = []
    decider = SimpleNamespace(m=SimpleNamespace(tok="tokenizer"), length=575,
                              cfg={}, neutralize_none=False)
    def load(source, **kwargs):
        calls.append((source, kwargs))
        return decider
    monkeypatch.setattr(backend.MLXDecider, "from_pretrained", load)
    engine = backend.MLXEngine("owner/model", revision="pinned")
    assert engine.decider is decider
    assert calls == [("owner/model", {"revision": "pinned"})]


def test_server_cli_accepts_hub_revision(monkeypatch):
    from decider import serve
    import sys
    import uvicorn
    monkeypatch.setattr(serve, "MODEL", "old")
    monkeypatch.setattr(serve, "MODEL_REVISION", None)
    monkeypatch.setattr(serve, "BACKEND", "cuda")
    monkeypatch.setattr(sys, "argv", ["serve", "--backend", "mlx", "--model",
                                     "owner/model", "--revision", "pinned"])
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw:
                        calls.append((serve.MODEL, serve.MODEL_REVISION)))
    serve.main()
    assert calls == [("owner/model", "pinned")]
