"""Optional inference adapter for precompiled ExecuTorch MLX programs.

ExecuTorch is imported only when constructing MLXDecider.
"""
import base64
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer
from decider.infer import Decider
from decider.systemone import render_state


class MLXDecider(Decider):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, revision=None,
                        cache_dir=None, local_files_only=False, token=None,
                        filename="model.pte"):
        """Load a local artifact/directory or a cached Hugging Face bundle.

        Pin ``revision`` to a Hub commit for reproducible downloads. No export,
        training or repository Python code is executed. Only load trusted PTEs.
        """
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("filename must be a relative path inside the model bundle")
        source = Path(pretrained_model_name_or_path)
        if source.is_file():
            artifact = source
        elif source.is_dir():
            artifact = source / relative
        else:
            snapshot = snapshot_download(
                repo_id=str(pretrained_model_name_or_path), revision=revision,
                cache_dir=cache_dir, local_files_only=local_files_only, token=token,
            )
            artifact = Path(snapshot) / relative
        if not artifact.is_file() or not Path(str(artifact) + ".json").is_file():
            raise FileNotFoundError(f"Expected {artifact} and its .json sidecar in the model bundle")
        return cls(artifact)

    def __call__(self, state, questions, **kwargs):
        """Evaluate a typed-decision request with the usual token/shape checks."""
        return self.evaluate(state, questions, **kwargs)

    def __init__(self, artifact):
        artifact = str(artifact)
        try:
            from executorch.runtime import Runtime
        except ImportError as exc:
            raise RuntimeError("The MLX backend requires an ExecuTorch installation with MLXBackend support") from exc
        runtime = Runtime.get()
        if "MLXBackend" not in runtime.backend_registry.registered_backend_names:
            raise RuntimeError("This ExecuTorch installation does not include MLXBackend")
        metadata = json.loads(Path(artifact + ".json").read_text())
        model = metadata["model"]
        cfg = metadata.get("decider_config")
        if cfg is None:
            config_path = Path(model) / "decider_config.json"
            if not config_path.is_file():
                config_path = Path(hf_hub_download(model, "decider_config.json"))
            cfg = json.loads(config_path.read_text())
        self.cfg = cfg
        self.metadata = metadata
        self.length = metadata["length"]
        self.dynamic = metadata.get("dynamic", False)
        self.alignment = metadata.get("alignment", 1)
        self.artifact = artifact
        self.program = self.method = None
        self.fast_scorer = None
        self.compact_only = metadata.get("format") == "shared_prefix"
        if self.compact_only:
            from decider.mlx_fast import FastScorer
            self.fast_scorer = FastScorer([Path(artifact)], model)
        elif metadata.get("fast_artifacts") and not metadata.get("vision"):
            from decider.mlx_fast import FastScorer
            self.fast_scorer = FastScorer([Path(artifact).parent / p for p in metadata["fast_artifacts"]], model)
        else:
            self.program = runtime.load_program(artifact)
            self.method = self.program.load_method("forward")
        def asset(key):
            return str(Path(artifact).parent / metadata[key]) if key in metadata else model
        self.m = SimpleNamespace(tok=AutoTokenizer.from_pretrained(asset("tokenizer")), slot_logits=self.slot_logits)
        self.eng = None
        self.dev = "cpu"  # Host inputs; the exported program delegates computation to MLX.
        self.T = float(cfg.get("temperature", 1.0))
        self.neutralize_none = bool(cfg.get("neutralize_none", True))
        self.isolated_levels = bool(cfg.get("isolated_levels", False))
        self.schema_first = False
        self.name = model + "/executorch-mlx"
        self.lock = threading.Lock()
        self.request_limit = self.length
        self.abstain_below = 0.0
        self.vision = metadata.get("vision", False)
        self.features = None
        if self.vision:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(asset("processor"))
            image_path = Path(metadata["image_encoder"])
            if not image_path.is_absolute():
                image_path = Path(artifact).parent / image_path.name
            self.image_program = runtime.load_program(str(image_path))
            self.image_method = self.image_program.load_method("forward")
            self.hidden_size = metadata["hidden_size"]
            self.image_prefix = self.m.tok.encode(
                "<|vision_start|>" + "<|image_pad|>" * 64 + "<|vision_end|>", add_special_tokens=False)

    def encode_image(self, encoded):
        if not self.vision:
            raise ValueError("Images require the decider-2b-vision model")
        from PIL import Image, ImageOps
        if len(encoded) > 1400000:
            raise ValueError("Image must be at most 1 MB")
        try:
            data = base64.b64decode(encoded, validate=True)
            if len(data) > 1024 * 1024:
                raise ValueError("Image must be at most 1 MB")
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > 16000000:
                    raise ValueError("Image exceeds 16 million pixels")
                source.load()
                img = ImageOps.pad(ImageOps.exif_transpose(source).convert("RGB"), (256, 256))
        except Exception as exc:
            raise ValueError("image_base64 must contain a valid image") from exc
        inputs = self.processor.image_processor(images=[img], return_tensors="pt")
        if inputs["image_grid_thw"].tolist() != [[1, 16, 16]]:
            raise ValueError("Image processor produced an unsupported patch grid")
        return self.image_method.execute([inputs["pixel_values"].contiguous()])[0].clone()

    def execute(self, ids, slot):
        if getattr(self, "compact_only", False):
            raise ValueError("Request exceeds the standalone fast export's shape bounds; no reference fallback is loaded")
        if self.method is None:
            from executorch.runtime import Runtime
            self.program = Runtime.get().load_program(self.artifact)
            self.method = self.program.load_method("forward")
        if not self.vision:
            return self.method.execute([ids, slot])[0].clone()
        n = ids.shape[1]
        positions = torch.arange(n).reshape(1, 1, n).expand(3, 1, n).clone()
        if self.features is not None:
            positions[0, 0, 1:65] = 1
            positions[1, 0, 1:65] = torch.arange(8).repeat_interleave(8) + 1
            positions[2, 0, 1:65] = torch.arange(8).repeat(8) + 1
            positions[:, :, 65:] -= 56
        features = self.features if self.features is not None else torch.zeros(64, self.hidden_size)
        return self.method.execute([ids, slot, features, positions])[0].clone()

    def slot_logits(self, ids, mask, slots, rows, nopts):
        if self.features is None and getattr(self, "fast_scorer", None) is not None:
            result = self.fast_scorer.score(ids, mask, slots, rows, nopts, self.m.tok.pad_token_id, self.request_limit)
            if result is not None:
                return result
        outputs = []
        for row, slot, n in zip(rows.tolist(), slots.tolist(), nopts.tolist()):
            length = int(mask[row].sum())
            values = ids[row, :length]
            if self.features is not None:
                values = torch.cat([torch.tensor(self.image_prefix), values])
                length += len(self.image_prefix)
                slot += len(self.image_prefix)
            if length > self.request_limit:
                raise ValueError(f"Prompt has {length} tokens; selected limit is {self.request_limit} (state + question/options).")
            run_length = max(2, length) if self.dynamic else self.length
            run_length = ((run_length + self.alignment - 1) // self.alignment) * self.alignment
            padded = torch.full((1, run_length), self.m.tok.pad_token_id, dtype=torch.long)
            padded[0, :length] = values
            logits = self.execute(padded, torch.tensor([slot]))
            logits[:, n:] = float("-inf")
            outputs.append(logits)
        return torch.cat(outputs)

    def evaluate(self, state, questions, limit=None, image_base64=None, independent=True, layout=None):
        if not questions:
            raise ValueError("At least one question is required")
        if layout not in (None, "state_first"):
            raise ValueError("The MLX export supports only state_first layout")
        limit = self.length if limit is None else limit
        if not 2 <= limit <= self.length:
            raise ValueError(f"max_prompt_tokens must be between 2 and {self.length}")
        # Decider truncates state internally; reject oversized state before that happens.
        state_tokens = len(self.m.tok.encode("Context:\n" + render_state(state), add_special_tokens=False))
        if state_tokens > limit:
            raise ValueError(f"State alone has {state_tokens} tokens; selected prompt limit is {limit}")
        with self.lock, torch.no_grad():
            self.request_limit = limit
            try:
                self.features = self.encode_image(image_base64) if image_base64 else None
                result = self.system_one(state, questions, max_state_tokens=limit, independent=independent)
                if self.features is not None:
                    result["usage"]["input_tokens"] += len(self.image_prefix)
                return result
            finally:
                self.request_limit = self.length
                self.features = None


class MLXEngine:
    """Serve the existing batching interface using serial, stateless MLX rows."""

    def __init__(self, artifact, revision=None):
        self.decider = MLXDecider.from_pretrained(artifact, revision=revision)
        self.tok = self.decider.m.tok
        self.max_ctx = self.decider.length
        self.cfg = self.decider.cfg
        self.graphs = {}
        self.stats = {"forwards": 0}
        self.neutralize_none = self.decider.neutralize_none

    @torch.no_grad()
    def score_items(self, items, temperature=1.0):
        from decider.model import collate
        results = []
        with self.decider.lock:
            for item in items:
                batch = collate([item], self.tok.pad_token_id)
                logits = self.decider.slot_logits(*[batch[k] for k in (
                    "input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
                results.append(torch.softmax(logits / temperature, -1))
                self.stats["forwards"] += len(item["slots"])
        return results

    score_shared = score_items  # No prefix cache in the stateless export.

    def warmup(self, shapes=()):
        return 0.0
