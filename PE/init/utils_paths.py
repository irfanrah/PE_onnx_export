import os
import glob


def find_adapter_config_path(weight_path: str) -> str:
    """
    Find adapter_config.json based on:
    1) If weight filename ends with numeric epoch suffix: <dir>/lora_adapter.bin.{epoch}/adapter_config.json
    2) Else: find *_adapter folder in same directory (pick latest if multiple)
    """
    folder_path, filename = os.path.split(weight_path)
    parts = filename.split(".")
    last = parts[-1] if parts else ""

    if last.isdigit():
        epoch = int(last)
        adapter_dir = os.path.join(folder_path, f"lora_adapter.bin.{epoch}")
        adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")
    else:
        candidates = [p for p in glob.glob(os.path.join(folder_path, "*_adapter")) if os.path.isdir(p)]
        if not candidates:
            raise FileNotFoundError("No '*_adapter' directory found and no epoch suffix in weight filename.")
        adapter_dir = max(candidates, key=os.path.getmtime) if len(candidates) > 1 else candidates[0]
        adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")

    if not os.path.exists(adapter_cfg):
        raise FileNotFoundError(f"Adapter config not found: {adapter_cfg}")

    return adapter_cfg
