"""Package an existing compact FP16 export for offline use and Hub publication."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def package(export, reference, output, checkpoint_revision, source_url):
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    fast = json.loads(Path(str(export) + '.json').read_text())
    ref = json.loads(reference.read_text())
    if fast.get('model') != ref.get('model'):
        raise ValueError('Fast export and reference metadata must identify the same checkpoint')
    if fast.get('format') != 'shared_prefix' or not fast.get('compact') or fast.get('dtype') != 'float16' or fast.get('bits'):
        raise ValueError('Expected an unquantized compact FP16 export')
    cfg = ref['decider_config']
    if 'temperature' not in cfg:
        raise ValueError('Missing checkpoint calibration temperature')
    tokenizer = reference.parent / ref['tokenizer']
    if not (tokenizer / 'tokenizer.json').is_file():
        raise ValueError('Missing local tokenizer assets')
    if len(checkpoint_revision) != 40 or any(c not in '0123456789abcdef' for c in checkpoint_revision):
        raise ValueError('Use the full source checkpoint commit hash')
    output.mkdir(parents=True)
    shutil.copy2(export, output / 'model.pte')
    shutil.copytree(tokenizer, output / 'model.pte.tokenizer')
    fast.update(length=fast['max_prefix'] + fast['max_suffix'], dynamic=True, alignment=1,
                tokenizer='model.pte.tokenizer', decider_config=cfg)
    (output / 'model.pte.json').write_text(json.dumps(fast, indent=2) + '\n')
    shutil.copy2(Path(__file__).resolve().parents[1] / 'LICENSE', output / 'LICENSE')
    with (output / 'model.pte').open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    model = ref['model']
    provenance = dict(base_model=model, base_revision=checkpoint_revision, source=source_url,
                      sha256=digest, bytes=export.stat().st_size,
                      runtime=dict(torch='2.14.0', executorch='1.5.0', transformers='5.17.0'))
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    card = f'''---
license: apache-2.0
base_model: {model}
language:
- en
pipeline_tag: text-classification
tags:
- executorch
- mlx
- apple-silicon
- fp16
---

# {model.split('/')[-1]}: compact ExecuTorch MLX export

Converted from [{model}](https://huggingface.co/{model}), checkpoint revision
`{checkpoint_revision}`. Original model, training, typed-decision interface and
calibration: **Mark Marosi / Mapika**, based on Qwen3.5. No additional training
or Doom-specific fine-tuning was performed for this export.

## Execution changes and credits

FP16 weights/activations, FP32 recurrent state, fused projections/normalization,
padding removal, and bounded parallel suffix recurrence. Prefixes may be shared
within a request; no answers or state are cached across requests.

Source and setup: [Decider MLX fork]({source_url}). PyTorch/ExecuTorch, Apple MLX,
Qwen and Hugging Face Transformers supply the underlying implementation and
runtime components. No upstream endorsement is implied. The Apache-2.0 license
is included. These are export/runtime changes, not a newly trained model.

## Download and serve

Download the entire repository at a pinned revision using
`huggingface_hub.snapshot_download`, then run the fork's server:

```sh
python -m decider.serve --backend mlx --model /path/to/snapshot/model.pte
```

Follow `docs/mlx.md` in the fork to install the Apple Silicon dependencies.
Keep the `.pte.json` sidecar and `.pte.tokenizer/` directory beside the binary.
Calibration and tokenizer assets are bundled for offline inference.
This is an ExecuTorch artifact, not an `mlx-lm` or Transformers checkpoint, and
does not run in the standard Hugging Face inference widget.

Only load trusted artifacts. Verify files against `SHA256SUMS`. Provenance and
the binary checksum are in `provenance.json`. Tested: Python 3.12, Torch 2.14.0,
ExecuTorch 1.5.0, Transformers 5.17.0, M4 Pro Mac with 48 GB memory. Other runtime
versions/hardware and operation without full Xcode are not yet validated.

## Limits

- Maximum {fast['length']} rendered tokens per row, subject to separate prefix
  and suffix bounds of {fast['max_prefix']} and {fast['max_suffix']} tokens.
- Up to {fast['batch']} rows per forward; the adapter splits larger groups.
- No larger reference fallback: unsupported shapes fail. No vision or 32k
  context support in this compact artifact.
- Inherits the base model's limitations, including imperfect conditional-rule
  following. Small numerical parity checks are not a general accuracy benchmark.
- Run one model at a time. Export/parity tests use much more memory than serving.
  A 48 GB test machine does not establish compatibility with every smaller Mac.

See the upstream model card for training details and evaluations; those results
belong to the model authors, not this conversion.
'''
    (output / 'README.md').write_text(card)
    sums = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            if path.name == 'model.pte':
                checksum = digest
            else:
                with path.open('rb') as stream:
                    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            sums.append(f'{checksum}  {path.relative_to(output).as_posix()}')
    (output / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    print(f'Packaged {model}: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Reference .pte.json metadata, not its weights')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-revision', required=True)
    parser.add_argument('--source-url', required=True, help='URL of the exact source commit')
    args = parser.parse_args()
    package(args.export, args.reference, args.output, args.checkpoint_revision, args.source_url)
