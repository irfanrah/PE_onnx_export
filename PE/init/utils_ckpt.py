from collections import OrderedDict
from typing import Dict, Any

import torch


def load_state_dict_flexible(path: str, device: str) -> Dict[str, Any]:
    """
    Supports:
      - plain state_dict
      - {'state_dict': ...}
      - {'model': ...}
    Also removes 'module.' prefix (DDP).
    """
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = OrderedDict()
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned
