from typing import Any, Callable

from ..config import PEInitConfig
from .base import LoadStrategy


class DefaultLoad(LoadStrategy):
    def validate(self, cfg: PEInitConfig) -> None:
        if cfg.weight_path is not None:
            raise ValueError("default: weight_path must be None")
        if cfg.lora_adapter_path is not None:
            raise ValueError("default: lora_adapter_path must be None")

    def build_model(self, cfg: PEInitConfig, load_clip: Callable[..., Any]) -> Any:
        print("Loaded base model")
        return load_clip(cfg, pretrained=True), None
