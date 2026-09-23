"""Export Decider 2B Vision: fixed 256px image encoder and dynamic text scorer."""
import json
from pathlib import Path
import types
import torch
from export_mlx import patch_attention, patch_recurrent_modules
from decider.vision.model import VisionDecisionModel
from decider.infer import Example, Q
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb_vision, get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime
from PIL import Image

NAME = "Mapika/decider-2b-vision"
BASE = "artifacts/decider-2b-vision-32768.pte"


def vision_attention(self, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs):
    length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scaling)
    return self.proj(out.transpose(1, 2).reshape(length, -1))


class ImageEncoder(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        grid = torch.tensor([[1, 16, 16]])
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=visual.num_grid_per_side, mode="bilinear",
            align_corners=True, spatial_merge_size=2, kwargs={})
        self.register_buffer("positions", (visual.pos_embed(indices) * weights[:, :, None]).sum(1).detach())
        pos_ids = get_vision_position_ids(grid, 2, kwargs={})
        cos, sin = visual.rotary_pos_emb(self.positions, pos_ids)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        for block in visual.blocks:
            block.attn.forward = types.MethodType(vision_attention, block.attn)

    def forward(self, pixels):
        h = self.visual.patch_embed(pixels) + self.positions
        for block in self.visual.blocks:
            h = block(h, cu_seqlens=None, position_embeddings=(self.cos, self.sin))
        return self.visual.merger(h)


class VisionScorer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.core = model.lm.model.language_model
        self.image_id = model.lm.config.image_token_id
        self.register_buffer("head", model.lm.lm_head.weight[model.letters].detach().clone())

    def forward(self, ids, slot, features, positions):
        embeddings = self.core.embed_tokens(ids)
        indices = (torch.arange(ids.shape[1], device=ids.device) - 1).clamp(0, 63)
        embeddings = torch.where((ids == self.image_id).unsqueeze(-1), features[indices].unsqueeze(0), embeddings)
        h = self.core(inputs_embeds=embeddings, position_ids=positions,
                      attention_mask={"full_attention": None, "linear_attention": None},
                      use_cache=False).last_hidden_state
        return torch.nn.functional.linear(h[0, slot], self.head).float()


def export(model, inputs, path, dynamic=None):
    reference = model(*inputs)
    ep = torch.export.export(model, inputs, strict=False, dynamic_shapes=dynamic)
    edge = to_edge_transform_and_lower(ep, partitioner=[MLXPartitioner()],
                                      compile_config=EdgeCompileConfig(_check_ir_validity=False))
    with open(path, "wb") as f:
        edge.to_executorch().write_to_file(f)
    program = Runtime.get().load_program(path)
    actual = program.load_method("forward").execute(list(inputs))[0]
    torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.01)
    print("PASS export", path, "max error", (actual-reference).abs().max().item(), flush=True)
    return actual


def main(name=NAME, output=BASE, length=32768):
    torch.set_num_threads(8)
    m = VisionDecisionModel(name, dtype=torch.float32, grad_ckpt=False).eval()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        example = Example("Look at the image.", [Q("What color is it?", ["red", "blue"])])
        inp = m.prepare([(Image.new("RGB", (256, 256), "red"), example)])
        print("Image grid", inp["image_grid_thw"], flush=True)
        assert inp["image_grid_thw"].tolist() == [[1, 16, 16]]
        expected = m.slot_logits(inp)
        original_features = m.lm.model.visual(inp["pixel_values"], grid_thw=inp["image_grid_thw"]).pooler_output
        encoder = ImageEncoder(m.lm.model.visual).eval()
        torch.testing.assert_close(encoder(inp["pixel_values"]), original_features)
        features = export(encoder, (inp["pixel_values"],), output + ".image.pte")
        n = inp["input_ids"].shape[1]
        ids = torch.full((1, 256), m.tok.pad_token_id, dtype=torch.long)
        ids[:, :n] = inp["input_ids"]
        mm = torch.zeros_like(ids)
        mm[:, 1:65] = 1
        positions, _ = m.lm.model.get_rope_index(ids, mm, image_grid_thw=inp["image_grid_thw"])
        scorer = VisionScorer(m).eval()
        inputs = (ids, inp["slot_idx"], features, positions)
        torch.testing.assert_close(scorer(*inputs)[:, :2], expected[:, :2], atol=0.02, rtol=0.01)
        patch_recurrent_modules(scorer)
        patch_attention(scorer)
        torch.testing.assert_close(scorer(*inputs)[:, :2], expected[:, :2], atol=0.02, rtol=0.01)
        tokens = 128 * torch.export.Dim("blocks", min=1, max=length // 128)
        actual = export(scorer, inputs, output, ({1: tokens}, None, None, {2: tokens}))
        torch.testing.assert_close(actual[:, :2], expected[:, :2], atol=0.02, rtol=0.01)
        from huggingface_hub import hf_hub_download
        cfg = json.loads(Path(hf_hub_download(name, "decider_config.json")).read_text())
        m.proc.save_pretrained(output + ".processor")
        Path(output + ".json").write_text(json.dumps({"model": name, "length": length, "dynamic": True,
            "alignment": 128, "vision": True, "image_encoder": Path(output + ".image.pte").name, "image_size": 256,
            "decider_config": cfg,
            "tokenizer": Path(output + ".processor").name, "processor": Path(output + ".processor").name,
            "hidden_size": m.lm.config.text_config.hidden_size}))
        print("PASS vision reference parity", torch.softmax(actual[0, :2], -1).tolist(), flush=True)


if __name__ == "__main__":
    main()
