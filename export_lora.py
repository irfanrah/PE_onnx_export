import os
import torch
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import torch.nn as nn

from PE.init import PEInitConfig, PEModelInitializer


# ---------- Wrappers ----------
class VisionEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor):
        return self.model.encode_video(video, normalize=False)


class TextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text: torch.Tensor):
        return self.model.encode_text(text, normalize=False)


class LogitScaleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self):
        return self.model.logit_scale.exp().unsqueeze(0)


class LoRAONNXExporter:
    def __init__(
        self,
        output_folder="onnx_export_lora",
        opset=17,
        export_vision=True,
        export_text=True,
        export_logit_scale=True,
        cleanup=True,
    ):
        self.output_folder = output_folder
        self.opset = opset
        self.export_vision = export_vision
        self.export_text = export_text
        self.export_logit_scale = export_logit_scale
        self.cleanup = cleanup
        os.makedirs(self.output_folder, exist_ok=True)

    def _consolidate_to_external(self, onnx_model_path):
        base_dir = os.path.dirname(onnx_model_path)
        base_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
        model = onnx.load(onnx_model_path, load_external_data=True)
        data_fname = f"{base_name}.data"
        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=data_fname,
            size_threshold=0,
            convert_attribute=False,
        )
        onnx.save_model(
            model,
            onnx_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_fname,
            size_threshold=0,
        )

    def _clean_up(self, out_dir):
        if not os.path.isdir(out_dir):
            return
        keep_exts = (".onnx", ".data")
        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)
            if os.path.isdir(file_path):
                continue
            if not filename.endswith(keep_exts):
                try:
                    os.remove(file_path)
                    print(f"[INFO] Removed: {file_path}")
                except OSError as e:
                    print(f"[ERROR] Failed to remove {file_path}: {e}")

    def export(self, model, config_name, export_name):
        """Export a LoRA-merged model to ONNX."""
        image_size = int(config_name.split("-")[-1])
        image_shape = (3, image_size, image_size)
        text_seq_len = 32

        out_dir = os.path.join(self.output_folder, export_name)
        os.makedirs(out_dir, exist_ok=True)

        if self.export_vision:
            print(f"Exporting vision: {export_name}")
            out_file = os.path.join(out_dir, f"{export_name}_vision.onnx")
            vmodel = VisionEncoderWrapper(model).eval()
            num_frames = 8
            video = torch.randn((1, num_frames) + image_shape, dtype=torch.float32, device=next(model.parameters()).device)
            torch.onnx.export(
                vmodel, (video,), out_file,
                export_params=True,
                input_names=["video"],
                output_names=["video_features"],
                dynamic_axes={"video": {0: "batch", 1: "num_frames"}, "video_features": {0: "batch"}},
                opset_version=self.opset,
            )
            print(f"  -> {out_file}")

        if self.export_text:
            print(f"Exporting text: {export_name}")
            out_file = os.path.join(out_dir, f"{export_name}_text.onnx")
            tmodel = TextEncoderWrapper(model).eval()
            text = torch.randint(0, 10000, (1, text_seq_len), dtype=torch.int32, device=next(model.parameters()).device)
            torch.onnx.export(
                tmodel, (text,), out_file,
                export_params=True,
                input_names=["text"],
                output_names=["text_features"],
                dynamic_axes={"text": {0: "batch", 1: "seq"}, "text_features": {0: "batch"}},
                opset_version=self.opset,
            )
            self._consolidate_to_external(out_file)
            print(f"  -> {out_file}")

        if self.export_logit_scale:
            print(f"Exporting logit_scale: {export_name}")
            out_file = os.path.join(out_dir, f"{export_name}_logit_scale.onnx")
            lmodel = LogitScaleWrapper(model).eval()
            torch.onnx.export(
                lmodel, tuple(), out_file,
                export_params=True,
                input_names=[],
                output_names=["logit_scale"],
                dynamic_axes={"logit_scale": {0: "batch"}},
                opset_version=self.opset,
            )
            print(f"  -> {out_file}")

        if self.cleanup:
            self._clean_up(out_dir)

        print(f"Done: {out_dir}")


def main():
    PRETRAINED_SPLIT_QKV_PATH = "/mnt/nas192/Research_materials/Kur/Blue-VLMTF-PVLM/code/Research-AI-mono/PE_FineTuning/other_model/PE/PE-Core-L14-336-split-qkv.pt"
    BASE_MODEL = "PE-Core-L14-336"
    DEVICE = "cuda:0"
    WEIGHT_PATH = "weights/FT_PE-Core-L14-336_260318.pt"
    EXPORT_NAME = "FT_PE-Core-L14-336_260318"

    # 1. Load model with LoRA weights
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

    print(f"Config: {cfg}")
    initializer = PEModelInitializer(cfg)
    model, preprocess, tokenizer, max_words, image_resolution = initializer.initialize()

    # 2. Merge LoRA weights into base model and unload adapter
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()
    print(f"Merged model type: {type(model)}")

    # 3. Export to ONNX
    exporter = LoRAONNXExporter(
        output_folder="onnx_export_lora",
        opset=17,
        export_vision=True,
        export_text=True,
        export_logit_scale=True,
    )
    exporter.export(model, BASE_MODEL, EXPORT_NAME)


if __name__ == "__main__":
    main()
