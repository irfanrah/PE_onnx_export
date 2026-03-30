import os
import torch
from typing import Any, Callable

from ..config import PEInitConfig
from .base import LoadStrategy


class WeightLoad(LoadStrategy):
    def validate(self, cfg: PEInitConfig) -> None:
        if not cfg.weight_path:
            raise ValueError("weight_load: weight_path is required")
        if cfg.lora_adapter_path is not None:
            raise ValueError("weight_load: lora_adapter_path must be None")

    def build_model(self, cfg: PEInitConfig, load_clip: Callable[..., Any]) -> Any:
        model = load_clip(cfg, pretrained=False)

        if not os.path.exists(cfg.weight_path):
            raise FileNotFoundError(f"Fine-tuned weights not found: {cfg.weight_path}")

        sd = torch.load(cfg.weight_path, map_location=cfg.device)
        model.load_state_dict(sd)
        print("Loaded fine-tuned model")
        return model
