import json
import torch
from typing import Any, Callable

from peft import get_peft_model, LoraConfig

from ..config import PEInitConfig
from ..utils_paths import find_adapter_config_path
from .base import LoadStrategy


class LoRAWeightLoad(LoadStrategy):
    def validate(self, cfg: PEInitConfig) -> None:
        if cfg.lora_adapter_path is not None:
            raise ValueError("lora_weight_load: lora_adapter_path must be None")

    def build_model(self, cfg: PEInitConfig, load_clip: Callable[..., Any]) -> Any:
        model = load_clip(cfg, pretrained=True)

        # training-mode: allow weight_path None
        if not cfg.weight_path:
            print("LoRAWeightLoad: weight_path is None -> returning base model (training mode).")
            return model, None

        adapter_cfg = find_adapter_config_path(cfg.weight_path)
        with open(adapter_cfg, "r") as f:
            j = json.load(f)

        lora_config = LoraConfig(
            r=j["r"],
            lora_alpha=j["lora_alpha"],
            target_modules=j["target_modules"],
            lora_dropout=j.get("lora_dropout", 0.0),
            bias=j.get("bias", "none"),
            modules_to_save=j.get("modules_to_save", None),
            task_type=j.get("task_type", "FEATURE_EXTRACTION"),
        )

        model = get_peft_model(model, lora_config)
        model.load_state_dict(torch.load(cfg.weight_path, map_location=cfg.device))
        print("Loaded LoRA with weights (lora_weight_load)")
        return model, None
