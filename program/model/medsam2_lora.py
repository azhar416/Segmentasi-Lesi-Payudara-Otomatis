import sys
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

SAM2_REPO_ROOT = Path(__file__).resolve().parents[1] / "sam2"
if str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class _HieraPeftWrapper(nn.Module):
    def __init__(self, original_trunk):
        super().__init__()
        self.original_trunk = original_trunk

    def forward(self, *args, **kwargs):
        return self.original_trunk(*args, **kwargs)


class MedSAM2LoRA:
    def __init__(
        self,
        weights_path,
        img_size=512,
        config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
        device=None,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.1,
    ):
        if img_size <= 0 or img_size % 16 != 0:
            raise ValueError("img_size must be a positive multiple of 16.")

        self.img_size = img_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._build_model(
            config_file=config_file,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self._load_weights(weights_path)

        self.predictor = SAM2ImagePredictor(self.model)
        # SAM 2 uses feature maps with strides 4, 8, and 16.
        feat_s0 = self.img_size // 4
        feat_s1 = self.img_size // 8
        feat_s2 = self.img_size // 16
        self.predictor._bb_feat_sizes = [
            (feat_s0, feat_s0),
            (feat_s1, feat_s1),
            (feat_s2, feat_s2),
        ]

    def _build_model(self, config_file, lora_rank, lora_alpha, lora_dropout):
        model = build_sam2(
            config_file=config_file,
            ckpt_path=None,
            device=str(self.device),
            mode="eval",
        )

        model.image_size = self.img_size
        embedding_size = self.img_size // 16
        model.sam_image_embedding_size = embedding_size
        model.sam_prompt_encoder.image_embedding_size = (
            embedding_size,
            embedding_size,
        )
        model.sam_prompt_encoder.input_image_size = (self.img_size, self.img_size)

        model.image_encoder.trunk = _HieraPeftWrapper(model.image_encoder.trunk)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["qkv", "proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        model.image_encoder.trunk = get_peft_model(
            model.image_encoder.trunk, lora_config
        )
        return model

    def _load_weights(self, weights_path):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {weights_path}")

        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=True,
        )
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image,
        bbox=None,
        point_prompt=None,
        point_labels=None,
        multimask_output=False,
    ):
        if bbox is None and point_prompt is None:
            raise ValueError("Provide bbox, point_prompt, or both.")

        if point_prompt is not None and point_labels is None:
            points = torch.as_tensor(point_prompt)
            if points.ndim == 1 and points.numel() == 2:
                point_count = 1
            elif points.ndim == 2 and points.shape[1] == 2:
                point_count = points.shape[0]
            else:
                raise ValueError("point_prompt must have shape (2,) or (N, 2).")
            point_labels = [1] * point_count

        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_prompt,
            point_labels=point_labels,
            box=bbox,
            multimask_output=multimask_output,
            normalize_coords=True,
        )
        return masks[0], scores, logits
