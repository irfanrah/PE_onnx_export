import os
import torch
from PIL import Image

from PE.init import PEInitConfig, PEModelInitializer


WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def main():
    PRETRAINED_SPLIT_QKV_PATH = "/mnt/nas192/Research_materials/Kur/Blue-VLMTF-PVLM/code/Research-AI-mono/PE_FineTuning/other_model/PE/PE-Core-L14-336-split-qkv.pt"
    BASE_MODEL = "PE-Core-L14-336"
    DEVICE = "cuda:1"
    weight_path = "/home/kurnianto/code/KhonkaenViolence/PE_onnx_export/weights/FT_PE-Core-L14-336_260318.pt"
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
    model.eval()

    # Run inference on a test image
    image_path = "assets/cat.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")


    image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.autocast("cuda"):
        image_features = model.encode_image(image)
    print(image_features)


if __name__ == "__main__":
    main()
