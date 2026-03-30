from dataclasses import dataclass
from typing import Optional, Any


@dataclass(frozen=True)
class PEInitConfig:
    model_name: str = "PE-Core-L14-336"
    device: str = "cuda:0"
    load_type: str = "default"  # default|weight_load|lora_weight_load|lora_adapter_load|lora_weight_head_load

    weight_path: Optional[str] = None
    lora_adapter_path: Optional[str] = None

    split_qkv: bool = False
    head_yml: Optional[Any] = None
    pretrained_split_qkv_path: Optional[str] = None
