import os
import torch
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import core.vision_encoder.pe as pe
import torch.nn as nn

# ---------- Wrappers ----------
class VisionEncoderWrapper(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, image: torch.Tensor): return self.model.encode_image(image, normalize=True)

class TextEncoderWrapper(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, text: torch.Tensor): return self.model.encode_text(text, normalize=True)

class LogitScaleWrapper(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self): return self.model.logit_scale.exp().unsqueeze(0)


class ONNXExporter:
    def __init__(
        self,
        input_folder="weights",
        output_folder="onnx_export",
        opset=17,
        export_monolith=True,
        export_vision=True,
        export_text=True,
        export_logit_scale=True,
        cleanup=True
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.opset = opset
        self.export_monolith = export_monolith
        self.export_vision = export_vision
        self.export_text = export_text
        self.export_logit_scale = export_logit_scale
        self.cleanup = cleanup
        os.makedirs(self.output_folder, exist_ok=True)

    def infer_input_shapes(self, config_name):
        image_size = int(config_name.split("-")[-1])
        image_shape = (3, image_size, image_size)
        token_length = {"224": 32, "336": 32, "448": 32}.get(str(image_size), 32)
        return image_shape, token_length

    def _clean_up(self, out_dir):
        """
        Remove all files in `out_dir` except files ending with .onnx or .data.
        """
        if not os.path.isdir(out_dir):
            print(f"[WARN] Directory does not exist: {out_dir}")
            return

        # Keep these extensions
        keep_exts = (".onnx", ".data")

        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)

            # Skip directories
            if os.path.isdir(file_path):
                continue

            # Remove files not matching keep_exts
            if not filename.endswith(keep_exts):
                try:
                    os.remove(file_path)
                    print(f"[INFO] Removed: {file_path}")
                except OSError as e:
                    print(f"[ERROR] Failed to remove {file_path}: {e}")

    def export_single_model(self, pt_path, config_name):
        print(f"⏳ Loading model: {config_name}")
        model = pe.CLIP.from_config(config_name, pretrained=False)
        state_dict = torch.load(pt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"ℹ️ Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        image_shape, text_seq_len = self.infer_input_shapes(config_name)
        print(f"✅ Input shapes: {image_shape}, {text_seq_len}")

        out_dir = os.path.join(self.output_folder, config_name)
        os.makedirs(out_dir, exist_ok=True)


        if self.export_monolith:
            print(f"⏳ Exporting monolith: {config_name}")
            out_file = os.path.join(out_dir, f"{config_name}_monolith.onnx")
            self._export_monolith(model, out_file, image_shape, text_seq_len, config_name)

        if self.export_vision:
            print(f"⏳ Exporting vision: {config_name}")
            out_file = os.path.join(out_dir, f"{config_name}_vision.onnx")
            self._export_vision(VisionEncoderWrapper(model).eval(), out_file, image_shape)

        if self.export_text:
            print(f"⏳ Exporting text: {config_name}")
            out_file = os.path.join(out_dir, f"{config_name}_text.onnx")
            self._export_text(TextEncoderWrapper(model).eval(), out_file, text_seq_len)

        if self.export_logit_scale:
            print(f"⏳ Exporting logit_scale: {config_name}")
            out_file = os.path.join(out_dir, f"{config_name}_logit_scale.onnx")
            self._export_logit_scale(LogitScaleWrapper(model).eval(), out_file)

        if self.cleanup:
            self._clean_up(out_dir)

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
            convert_attribute=False
        )
        onnx.save_model(
            model,
            onnx_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_fname,
            size_threshold=0,
        )

    # ---------- Raw export helpers ----------
    def _export_monolith(self, model, output_file, input_size, text_seq_len, config_name, batch_size=1):
        image = torch.randn((batch_size,) + input_size, dtype=torch.float32)
        text = torch.randint(0, 10000, (batch_size, text_seq_len), dtype=torch.int32)
        torch.onnx.export(
                model, (image, text), output_file,
                export_params=True,
                input_names=["image", "text"],
                output_names=["image_features", "text_features", "logit_scale"],
                dynamic_axes={
                    "image": {0: "batch"},
                    "text": {0: "batch", 1: "seq"},
                    "image_features": {0: "batch"},
                    "text_features": {0: "batch"},
                    "logit_scale": {0: "batch"},
                },
                opset_version=self.opset,
            )
        if config_name == "PE-Core-L14-336":
            self._consolidate_to_external(output_file)
            

    def _export_vision(self, vmodel, output_file, input_size, batch_size=1):
        image = torch.randn((batch_size,) + input_size, dtype=torch.float32)
        torch.onnx.export(
            vmodel, (image,), output_file,
            export_params=True,
            input_names=["image"],
            output_names=["image_features"],
            dynamic_axes={"image": {0: "batch"}, "image_features": {0: "batch"}},
            opset_version=self.opset,
        )

    def _export_text(self, tmodel, output_file, text_seq_len, batch_size=1):
        text = torch.randint(0, 10000, (batch_size, text_seq_len), dtype=torch.int32)
        torch.onnx.export(
            tmodel, (text,), output_file,
            export_params=True,
            input_names=["text"],
            output_names=["text_features"],
            dynamic_axes={"text": {0: "batch", 1: "seq"}, "text_features": {0: "batch"}},
            opset_version=self.opset,
        )

    def _export_logit_scale(self, lmodel, output_file):
        torch.onnx.export(
            lmodel, tuple(), output_file,
            export_params=True,
            input_names=[],
            output_names=["logit_scale"],
            dynamic_axes={"logit_scale": {0: "batch"}},
            opset_version=self.opset,
        )

    # ---------- Orchestration ----------
    def export_all(self):
        pt_files = [f for f in os.listdir(self.input_folder) if f.endswith(".pt")]
        assert pt_files, "No .pt files found in input folder"
        print(f"📁 Found {len(pt_files)} .pt files")
        for pt_file in pt_files:
            config_name = pt_file.replace(".pt", "")
            pt_path = os.path.join(self.input_folder, pt_file)
            try:
                self.export_single_model(pt_path, config_name)
            except Exception as e:
                print(f"❌ Failed to export {pt_file}: {e}")


if __name__ == "__main__":
    exporter = ONNXExporter(
        input_folder="weights",
        output_folder="onnx_export",
        opset=17,
        export_monolith=True,
        export_vision=True,
        export_text=True,
        export_logit_scale=True
    )
    exporter.export_all()
