import os
from typing import Any, Callable

from peft import get_peft_model, LoraConfig

from bluevlm_trainer.modules.vision_text_head import VisionTextHead

from ..config import PEInitConfig
from ..utils_ckpt import load_state_dict_flexible
from .base import LoadStrategy


class LoRAWeightHeadLoad(LoadStrategy):
    def validate(self, cfg: PEInitConfig) -> None:
        if cfg.head_yml is None:
            raise ValueError("lora_weight_head_load: head_yml is required")

        # if split_qkv enabled, we need the pretrained split-qkv weights
        if cfg.split_qkv and not cfg.pretrained_split_qkv_path:
            raise ValueError("lora_weight_head_load: pretrained_split_qkv_path is required when split_qkv=True")

    def build_model(self, cfg: PEInitConfig, load_clip: Callable[..., Any]) -> Any:
        PE_base = load_clip(cfg, pretrained=True)

        lora_args = cfg.head_yml.training.lora
        if not getattr(lora_args, "use", False):
            raise ValueError("lora_weight_head_load: head_yml.training.lora.use must be True")

        lora_config = LoraConfig(
            r=lora_args.rank,
            lora_alpha=lora_args.alpha,
            target_modules=lora_args.target_modules,
            lora_dropout=lora_args.dropout,
            bias=lora_args.bias,
            modules_to_save=lora_args.modules_to_save,
            task_type="FEATURE_EXTRACTION",
        )

        PE_base = get_peft_model(PE_base, lora_config)
        print("Loaded LoRA config from head_yml (lora_weight_head_load)")

        model = VisionTextHead(cfg.head_yml, PE_base, cfg.device)
        print("Loaded VisionTextHead layer")

        if cfg.weight_path:
            if not os.path.isfile(cfg.weight_path):
                raise FileNotFoundError(f"Weight file not found: {cfg.weight_path}")
            sd = load_state_dict_flexible(cfg.weight_path, cfg.device)
            model.load_state_dict(sd, strict=True)
            print("Loaded VisionTextHead weight")

        return model, PE_base
