import os
from typing import Any, Callable

from peft import PeftModel

from ..config import PEInitConfig
from .base import LoadStrategy


class LoRAAdapterLoad(LoadStrategy):
    def validate(self, cfg: PEInitConfig) -> None:
        if not cfg.lora_adapter_path:
            raise ValueError("lora_adapter_load: lora_adapter_path is required")
        if cfg.weight_path is not None:
            raise ValueError("lora_adapter_load: weight_path must be None")

    def build_model(self, cfg: PEInitConfig, load_clip: Callable[..., Any]) -> Any:
        model = load_clip(cfg, pretrained=True)

        adapter_cfg = os.path.join(cfg.lora_adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_cfg):
            raise FileNotFoundError(f"Adapter config not found: {adapter_cfg}")

        model = PeftModel.from_pretrained(model, cfg.lora_adapter_path)
        print("Loaded LoRA with Adapter (lora_adapter_load)")
        return model, None
