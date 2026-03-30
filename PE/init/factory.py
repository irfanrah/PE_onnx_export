import os
import sys
from typing import Any, Tuple
import torch
# Safe fallback (only needed if running from deep subfolder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from .config import PEInitConfig
from .strategies.base import LoadStrategy
from .strategies.default import DefaultLoad
from .strategies.weight import WeightLoad
from .strategies.lora_weight import LoRAWeightLoad
from .strategies.lora_adapter import LoRAAdapterLoad
from .strategies.lora_weight_head import LoRAWeightHeadLoad


def _setup_transforms(model) -> Tuple[Any, Any, int, int]:
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    return preprocess, tokenizer, model.context_length, model.image_size


def load_clip(cfg: PEInitConfig, *, pretrained: bool) -> Any:
    """
    Centralized CLIP creation to avoid repeating the split-qkv logic in each strategy.
    """
    model_name = cfg.model_name

    if cfg.split_qkv:
        model_name = model_name + "-splitqkv"
        print(f"### Model name: {model_name}")
        model = pe.CLIP.from_config(model_name, pretrained=False).to(cfg.device)

        if pretrained:
            if not cfg.pretrained_split_qkv_path:
                raise ValueError("pretrained_split_qkv_path is required when split_qkv=True and pretrained=True")


            state_dict = torch.load(cfg.pretrained_split_qkv_path, map_location=cfg.device)
            model.load_state_dict(state_dict)
            print(f"### Loaded weights with split QKV: {cfg.pretrained_split_qkv_path}")
        return model

    return pe.CLIP.from_config(model_name, pretrained=pretrained).to(cfg.device)


class PEModelInitializer:
    """
    Facade: pick a strategy, build model, then attach transforms.
    """
    _STRATEGIES = {
        "default": DefaultLoad(),
        "weight_load": WeightLoad(),
        "lora_weight_load": LoRAWeightLoad(),
        "lora_adapter_load": LoRAAdapterLoad(),
        "lora_weight_head_load": LoRAWeightHeadLoad(),
    }

    def __init__(self, cfg: PEInitConfig):
        self.cfg = cfg
        if cfg.load_type not in self._STRATEGIES:
            raise ValueError(f"Unknown load_type: {cfg.load_type}")

    def initialize(self):
        strategy: LoadStrategy = self._STRATEGIES[self.cfg.load_type]
        strategy.validate(self.cfg)

        model, PE_base = strategy.build_model(self.cfg, load_clip=load_clip)

        if not PE_base:
            PE_base = model

        preprocess, tokenizer, max_words, image_resolution = _setup_transforms(PE_base)
        print("### Successfully initialized model")
        return model, preprocess, tokenizer, max_words, image_resolution
