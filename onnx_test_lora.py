import os
import numpy as np
import onnxruntime as ort
from PIL import Image

import torch

from PE.init import PEInitConfig, PEModelInitializer


def _providers():
    provs = ort.get_available_providers()
    if "CUDAExecutionProvider" in provs:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def main():
    PRETRAINED_SPLIT_QKV_PATH = "/mnt/nas192/Research_materials/Kur/Blue-VLMTF-PVLM/code/Research-AI-mono/PE_FineTuning/other_model/PE/PE-Core-L14-336-split-qkv.pt"
    BASE_MODEL = "PE-Core-L14-336"
    DEVICE = "cuda:0"
    WEIGHT_PATH = "weights/FT_PE-Core-L14-336_260318.pt"
    VISION_ONNX_PATH = "onnx_export_lora/FT_PE-Core-L14-336_260318/FT_PE-Core-L14-336_260318_vision.onnx"
    IMAGE_PATH = "assets/cat.jpg"

    # ========== 1. PyTorch Reference ==========
    print("Loading PyTorch model with LoRA...")
    cfg = PEInitConfig(
        model_name=BASE_MODEL,
        device=DEVICE,
        weight_path=WEIGHT_PATH,
        load_type="lora_weight_load",
        lora_adapter_path=None,
        head_yml=None,
        pretrained_split_qkv_path=PRETRAINED_SPLIT_QKV_PATH,
        split_qkv=True,
    )

    initializer = PEModelInitializer(cfg)
    model, preprocess, tokenizer, max_words, image_resolution = initializer.initialize()
    model.eval()

    image = preprocess(Image.open(IMAGE_PATH)).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.autocast("cuda"):
        image_features_pt = model.encode_image(image)

    print(f"[PyTorch] image_features shape: {image_features_pt.shape}")
    print(f"[PyTorch] image_features:\n{image_features_pt}")

    # ========== 2. ONNX Vision Inference ==========
    print("\nLoading ONNX vision model...")
    vision_sess = ort.InferenceSession(VISION_ONNX_PATH, providers=_providers())
    image_np = image.detach().cpu().numpy().astype(np.float32)
    image_features_onnx = vision_sess.run(None, {vision_sess.get_inputs()[0].name: image_np})[0]
    image_features_onnx = torch.from_numpy(image_features_onnx).to(DEVICE).float()

    print(f"[ONNX]    image_features shape: {image_features_onnx.shape}")
    print(f"[ONNX]    image_features:\n{image_features_onnx}")

    # ========== 3. Comparison ==========
    image_features_pt = image_features_pt.float()
    cos_sim = torch.nn.functional.cosine_similarity(image_features_pt, image_features_onnx).item()
    mse = torch.nn.functional.mse_loss(image_features_pt, image_features_onnx).item()
    print(f"\nCosSim: {cos_sim:.6f}  MSE: {mse:.8f}")


if __name__ == "__main__":
    main()
