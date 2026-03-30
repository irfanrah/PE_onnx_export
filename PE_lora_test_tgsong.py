import os
import torch

from PE.init import PEInitConfig, PEModelInitializer
from peft import PeftModel


WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


class VideoEncoderWrapper(torch.nn.Module):
    """
    Thin wrapper so the exported ONNX graph matches the runtime call in main.py:
    frames -> video embedding vector.
    """
    def __init__(self, base_model: torch.nn.Module, normalize: bool = False):
        super().__init__()
        self.base_model = base_model
        self.normalize = normalize

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: float32 tensor shaped (B, T, 3, H, W)
        """
        return self.base_model.encode_video(video, normalize=self.normalize)


def maybe_merge_lora(model):
    """
    If the model is wrapped with PEFT LoRA, merge adapter weights into base model.
    """
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model


def main():
    PRETRAINED_SPLIT_QKV_PATH = "/mnt/nas192/Research_materials/Kur/Blue-VLMTF-PVLM/code/Research-AI-mono/PE_FineTuning/other_model/PE/PE-Core-L14-336-split-qkv.pt"
    BASE_MODEL = "PE-Core-L14-336"
    DEVICE = "cuda:1"

    weight_path = os.path.join(WEIGHTS_DIR, "FT_PE-Core-L14-336_260318.pt")
    output_path = "video_encoder.onnx"

    normalize_output = False
    temporal_size = 8
    image_size = 336
    opset = 17

    cfg = PEInitConfig(
        model_name=BASE_MODEL,
        device=DEVICE,
        weight_path=weight_path,
        load_type="lora_weight_load",
        lora_adapter_path=None,
        head_yml=None,
        pretrained_split_qkv_path=PRETRAINED_SPLIT_QKV_PATH,
        split_qkv=True,
    )

    print(f"Config: {cfg}")

    initializer = PEModelInitializer(cfg)
    model, preprocess, tokenizer, max_words, image_resolution = initializer.initialize()
    model = maybe_merge_lora(model).to(DEVICE)
    model.eval()

    wrapper = VideoEncoderWrapper(model, normalize=normalize_output).to(DEVICE)
    wrapper.eval()

    dummy = torch.randn(1, temporal_size, 3, image_size, image_size, device=DEVICE)

    dynamic_axes = {
        "video": {0: "batch", 1: "frames"},
        "video_embedding": {0: "batch"},
    }

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["video"],
        output_names=["video_embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    print(f"[INFO] ONNX export complete -> {output_path}")


if __name__ == "__main__":
    main()