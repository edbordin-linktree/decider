"""Optional bounded shared-prefix programs with reference-export fallback."""

import json
from pathlib import Path

import torch


class FastScorer:
    def __init__(self, artifacts, model):
        self.exports = []
        self.artifact_paths = {}
        for artifact in artifacts:
            if not Path(artifact).is_file():
                raise ValueError(f"Missing fast artifact: {artifact}")
            metadata = json.loads(Path(str(artifact) + ".json").read_text())
            if metadata["model"] != model or metadata["format"] != "shared_prefix":
                raise ValueError(f"Fast artifact does not match {model}: {artifact}")
            if (
                metadata["batch"] < 1
                or min(metadata["max_prefix"], metadata["max_suffix"]) < 2
            ):
                raise ValueError(f"Invalid fast artifact dimensions: {artifact}")
            self.artifact_paths[id(metadata)] = str(artifact)
            self.exports.append((metadata, None, None))
        self.exports.sort(key=lambda entry: entry[0]["batch"])
        if not self.exports:
            raise ValueError("At least one fast artifact is required")

    @staticmethod
    def _plan(metadata, ids, lengths, slots, rows, nopts, pad_id):
        plans = []
        capacity = metadata["batch"]
        for offset in range(0, len(rows), capacity):
            selected = rows[offset : offset + capacity].tolist()
            selected_slots = slots[offset : offset + capacity].tolist()
            counts = nopts[offset : offset + capacity].tolist()
            values = [
                ids[row, :length].tolist()
                for row, length in zip(selected, lengths[offset : offset + capacity])
            ]
            if metadata.get("compact"):
                # Causal scoring cannot depend on tokens after this answer slot.
                values = [
                    value[: slot + 1] for value, slot in zip(values, selected_slots)
                ]
            prefix = 0
            while prefix < min(min(selected_slots), metadata["max_prefix"]) and all(
                value[prefix] == values[0][prefix] for value in values
            ):
                prefix += 1
            suffix = max(2, max(len(value) - prefix for value in values))
            if prefix < 2 or suffix > metadata["max_suffix"]:
                return None
            batch = len(selected) if metadata.get("dynamic_batch") else capacity
            tokens = torch.full((batch, suffix), pad_id, dtype=torch.long)
            indices = torch.zeros(batch, dtype=torch.long)
            for i, (value, slot) in enumerate(zip(values, selected_slots)):
                tokens[i, : len(value) - prefix] = torch.tensor(value[prefix:])
                indices[i] = slot - prefix
            inputs = [torch.tensor([values[0][:prefix]]), tokens, indices]
            if metadata.get("compact"):
                keep = list(range(prefix))
                for i, slot in enumerate(indices.tolist()):
                    keep.extend(
                        range(prefix + i * suffix, prefix + i * suffix + slot + 1)
                    )
                inputs.append(torch.tensor(keep))
            plans.append((inputs, counts))
        return plans

    def score(self, ids, mask, slots, rows, nopts, pad_id, limit):
        lengths = [int(mask[row].sum()) for row in rows.tolist()]
        for length in lengths:
            if length > limit:
                raise ValueError(
                    f"Prompt has {length} tokens; selected limit is {limit}"
                )
        # Prefer the smallest sufficient batch. Preserve artifact order on ties,
        # so specialized exports can precede a more general fast export.
        candidates = sorted(
            self.exports,
            key=lambda entry: (
                entry[0]["batch"] < len(rows),
                entry[0]["batch"]
                if entry[0]["batch"] >= len(rows)
                else -entry[0]["batch"],
            ),
        )
        for metadata, _, method in candidates:
            plans = self._plan(metadata, ids, lengths, slots, rows, nopts, pad_id)
            if plans is None:
                continue
            if method is None:
                from executorch.runtime import Runtime

                program = Runtime.get().load_program(self.artifact_paths[id(metadata)])
                method = program.load_method("forward")
                index = next(
                    i for i, entry in enumerate(self.exports) if entry[0] is metadata
                )
                self.exports[index] = (metadata, program, method)
            results = []
            for inputs, counts in plans:
                logits = method.execute(inputs)[0][: len(counts)].clone()
                for i, count in enumerate(counts):
                    if not torch.isfinite(logits[i, :count]).all():
                        raise RuntimeError(
                            "MLX fast program produced non-finite logits"
                        )
                    logits[i, count:] = float("-inf")
                results.append(logits)
            return torch.cat(results)
        return None
