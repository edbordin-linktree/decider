# ExecuTorch MLX backend

The optional MLX backend runs precompiled Decider models on Apple Silicon. It
does not use PyTorch MPS. CUDA remains the default; importing the server or
running the CUDA backend does not import ExecuTorch or require MLX.

## Download and call a model

After installing the [MLX environment](#install-the-runtime), use a
Hugging Face model ID. The loader downloads the complete bundle once and caches
it; you do not need to manage `.pte` files or tokenizer directories.

```python
from decider.mlx_backend import MLXDecider

model = MLXDecider.from_pretrained(
    "edbordin-linktree/decider-2b-executorch-mlx",
    revision="0ecf677be2014eb548c199597c17bc4edb5a3954",
)
result = model(
    {"ticket": "I was charged twice. Please refund the duplicate."},
    {"team": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Charges, invoices and refunds",
                           "technical": "Bugs and technical problems"}}},
)
print(result["answers"])
```

Use `local_files_only=True` to use an already cached revision offline;
`cache_dir=...` optionally selects a different cache. `revision` is optional,
but pinning a commit makes downloads reproducible. Only load trusted artifacts.
No repository Python code, model export or training runs during download.

## Run the HTTP server

The server accepts the same Hub IDs and manages the download/cache automatically:

```sh
python -m decider.serve --backend mlx \
  --model edbordin-linktree/decider-2b-executorch-mlx \
  --revision 0ecf677be2014eb548c199597c17bc4edb5a3954
```

The server binds to `127.0.0.1:8000`. Use `--host` and `--port` to change this.
The existing environment-based entry point also works:

```sh
DECIDER_BACKEND=mlx DECIDER_MODEL=edbordin-linktree/decider-2b-executorch-mlx \
  DECIDER_REVISION=0ecf677be2014eb548c199597c17bc4edb5a3954 \
  uvicorn decider.serve:app --host 127.0.0.1 --port 8000
```

The bundle includes calibration and tokenizer assets. After download, cached
inference can run offline without the original checkpoint. The server does
not export models at startup.

### Standalone compact exports

Public precompiled bundles are available; no Hugging Face login is required:

| Model | Binary size | Pinned Hub revision |
| --- | --- | --- |
| [0.8B compact](https://huggingface.co/edbordin-linktree/decider-0.8b-executorch-mlx) | 1.41 GiB | `f6f0468901f8ae6a14e6c05b19f7d116b3b90377` |
| [2B compact](https://huggingface.co/edbordin-linktree/decider-2b-executorch-mlx) | 3.51 GiB | `0ecf677be2014eb548c199597c17bc4edb5a3954` |

For 0.8B, use its repository and revision from the table in either interface.
The public repositories
also retain Mapika's license and identify the exact original checkpoint revision.
These compact bundles do not include 32k reference or vision exports.

### Development: local artifacts

Local loading remains available for export development. Either pass a `.pte`
path to the server, or call `MLXDecider.from_pretrained(local_directory)` or
`MLXDecider(local_pte_path)`. It is not needed for normal use.

A newly generated fast export can be served directly, without a reference model:

```sh
python -m decider.serve --backend mlx --model artifacts/decider-2b-compact-fp16.pte
```

Distribute the `.pte`, its `.pte.json` sidecar and `.pte.tokenizer/` directory
together. The sidecar embeds Decider's calibration settings. Standalone fast
exports load one program on demand; requests outside their shape bounds fail
instead of loading a larger fallback model. The compact configuration below
supports up to 575 total tokens per row (512 prefix plus 63 suffix), subject to
the individual shape bounds. This is not a 32k export.

To package an older compact artifact with its existing calibration/tokenizer:

```sh
python scripts/package_mlx_fast.py \
  --export artifacts/decider-2b-compact-fp16.pte \
  --reference artifacts/decider-2b.pte.json \
  --checkpoint-revision FULL_SOURCE_CHECKPOINT_COMMIT \
  --source-url https://github.com/OWNER/decider/tree/SOURCE_COMMIT \
  --output dist-models/decider-2b-executorch-mlx
```

Replace the uppercase placeholders with the exact revisions and source URL.
This copies existing weights without loading them, embeds metadata, and writes
a model card, original license, provenance and `SHA256SUMS`. It refuses to
overwrite a directory. Upload the complete output directory, not only the binary.

`/decide` and `/v1/systemone` retain their existing formats. On the MLX backend,
`/v1/systemone` also accepts `max_prompt_tokens` and optional `image_base64`.
The latter is a base64-encoded PNG or JPEG, without a data-URL prefix. Images
require the vision checkpoint. Each request can include one image, at most
1 MiB and 16 million pixels; it is fitted and padded to 256 × 256 pixels.

## Install the runtime

The tested stack is Python 3.12, Torch 2.14.0, ExecuTorch 1.5.0 and Transformers
5.17.0 on Apple Silicon. The ExecuTorch wheel must register `MLXBackend`.
Precompiled bundles need no model export. Operation on a clean Mac without
full Xcode has not yet been validated.

Use a separate environment. Upstream Decider's CUDA-focused dependencies
include `flash-linear-attention` and NumPy below 2, which conflict with this
ExecuTorch environment. Run from this checkout without installing those default
dependencies:

```sh
uv venv --python 3.12 .venv-mlx
uv pip install --python .venv-mlx/bin/python -r scripts/requirements-mlx.txt
uv pip install --python .venv-mlx/bin/python --no-deps .
source .venv-mlx/bin/activate
```

## Development: generate model files

Export requires Apple's Metal compiler. Install it if needed:

```sh
xcodebuild -downloadComponent MetalToolchain
```

```sh
.venv-mlx/bin/python scripts/export_mlx.py --model Mapika/decider-0.8b \
  --dynamic --length 32768 --output artifacts/decider-0.8b.pte
.venv-mlx/bin/python scripts/export_mlx.py --model Mapika/decider-2b \
  --dynamic --length 32768 --output artifacts/decider-2b.pte
.venv-mlx/bin/python scripts/export_mlx.py --model Mapika/decider-2b-vision \
  --dynamic --length 32768 --output artifacts/decider-2b-vision.pte
```

Export one model at a time and do not overwrite a bundle a server is using.
The script compares exported outputs with CPU FP32 before publishing metadata.
Text artifacts are about 2.8 GiB and 7 GiB; vision adds a roughly 1.2 GiB image
encoder to its 7 GiB text scorer. Original checkpoint caches take extra space.

## Optional short-request fast path

Generate a separate packed FP16 scorer for text models:

```sh
PYTHONPATH=. .venv-mlx/bin/python scripts/export_mlx_fast.py \
  --model Mapika/decider-2b --dtype float16 --packed --fuse-projections \
  --fuse-norms \
  --output artifacts/decider-2b-packed-fp16.pte
```

Add `"fast_artifacts": ["decider-2b-packed-fp16.pte"]` to the reference
artifact's `.pte.json`, then restart the server. Paths are relative to that
metadata file. Keep the reference artifact: inputs outside the fast export's
bounds fall back to it, loading it on demand. Remove `fast_artifacts` to disable
the optimization. Vision does not use this path.

For lower latency on short questions, generate a compact scorer as well:

```sh
PYTHONPATH=. .venv-mlx/bin/python scripts/export_mlx_fast.py \
  --model Mapika/decider-2b --dtype float16 --packed --fuse-projections \
  --fuse-norms --parallel-suffix --dynamic-batch --compact \
  --output artifacts/decider-2b-compact-fp16.pte
```

Put the compact export first in `fast_artifacts`, followed by the broader packed
export. The compact path supports 1–6 rows, prefixes of 2–512 tokens and suffixes
of 2–63 tokens. It removes padded positions from the MLP, full attention and
recurrent projections, and uses a block-triangular parallel recurrent update.
The 63-token bound leaves one padding position in its 64-token solve. Unsupported
shapes try the next fast artifact, then the FP32 reference. The token budget is
still checked against the original prompt, not its compact representation.

The same command supports `Mapika/decider-0.8b`. Fast artifacts must match the
reference checkpoint. Artifacts load only when a request needs them, then remain
resident until the model is unloaded. Each loaded artifact keeps its own weights;
the 2B FP16 exports each occupy about 3.5 GiB.

The fast export shares a common prefix within each request, packs six independent
suffixes into one forward, and fuses projections. It supports prefixes of 2–512
tokens and suffixes of 2–256 tokens; larger batches split into groups. Its attention
mask and recurrent-state branches isolate questions. Recurrent state stays FP32
and resets for every group. No answers or state are cached between requests.

On an M4 Pro, the compact 2B Support Routing playground example took about
143 ms through Python HTTP, versus 476 ms for the initial serial version.
Ten compact-path validation cases covered changed inputs, ordering, multi-batch
requests, long prefixes, both fallback levels and request isolation; the largest
probability difference was 0.0021, with unchanged winning choices.
This is a small parity suite, not a general accuracy evaluation.

`--bits 4` or `--bits 8` enables experimental ExecuTorch Qwen-style weight-only
quantization with 8-bit embeddings. Neither is enabled by default: the tested
4-bit export changed a winning answer, and 8-bit offered no latency benefit.

## Scope and limits

- Tested checkpoints: 0.8B, 2B and 2B Vision. The 35B model is not included.
- FP32 reference scoring, optional FP16 fast exports, up to 255 options, and up
  to 32,768 prompt tokens per row.
  The budget includes state, rendered questions/options and 66 image-prefix
  tokens when an image is supplied. Oversized System One prompts are rejected.
- Inputs are right-padded to multiples of 128. Chunked causal attention avoids
  allocating a single full-context attention matrix. There is no inherent 8k
  MLX limit here; long inputs still require substantial memory and time.
- Reference exports execute rows serially; optional fast exports share prefixes
  within a request. There is no persistent prefix cache or CUDA graph capture.
  System One retains question independence, checkpoint calibration and score
  isolation settings. Only the state-first layout is supported.
- Vision uses fixed-size image preprocessing, not arbitrary-resolution or video
  inputs. Its v5 checkpoint is distinct from the newer 2B text checkpoint.
- The existing CUDA path remains unchanged. Tests simulate missing ExecuTorch
  and check default startup, CLI selection, option forwarding and token limits.

These exports passed short-prompt CPU parity and full-length execution checks
on a 48 GB Mac. This is not an accuracy benchmark across long documents or a
guarantee that 32k inputs fit smaller-memory machines.

## Attribution and release notes

Decider's models, training, typed-decision interface and calibration are the
work of Mark Marosi / Mapika, based on Qwen3.5. This fork adds export/runtime
engineering; it does not retrain the checkpoints. It relies on PyTorch,
ExecuTorch's MLX delegate, Apple MLX and Hugging Face Transformers. The delegate
and GPU primitives are upstream work, not inventions of this fork.

Local additions include the MLX adapter/export scripts, compact padding removal,
fused projections and normalization, and a bounded parallel suffix recurrence.
Recurrent state remains FP32. No answers or state are cached between requests.
The original Apache-2.0 license remains in `LICENSE`.

The initial implementation is based on upstream commit
`c4daaac28af9fea95d627015cffa2dd5a5926ee6`. The separate Doom demonstration and
its prompt adaptations are not part of this server patch. Published model
exports should identify the exact source checkpoint revision, export flags,
runtime versions and checksum, and must not imply endorsement by upstream authors.

Precompiled runtime wheels contain MLX binaries and Metal resources. Running
without full Xcode has not yet been validated on a clean Mac; do not treat the
export being precompiled as proof that all developer-tool prerequisites vanish.
