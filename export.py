import os
import torch
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import core.vision_encoder.pe as pe
import shutil


class ONNXExporter:
    def __init__(self, input_folder="weights", output_folder="onnx", opset=17):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.opset = opset
        os.makedirs(self.output_folder, exist_ok=True)

    def infer_input_shapes(self, config_name):
        image_size = int(config_name.split("-")[-1])
        image_shape = (3, image_size, image_size)

        token_dict = {
            "224": 32,
            "336": 32,
            "448": 32
        }
        token_length = token_dict.get(str(image_size), 32)
        return image_shape, token_length

    def export_single_model(self, pt_path, config_name):
        print(f"⏳ Loading model: {config_name}")
        model = pe.CLIP.from_config(config_name, pretrained=False)
        state_dict = torch.load(pt_path, map_location='cuda')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"ℹ️ Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        image_shape, text_seq_len = self.infer_input_shapes(config_name)
        print(f"✅ Input shapes: {image_shape}, {text_seq_len}")

        onnx_folder = os.path.join(self.output_folder, config_name)
        onnx_path = os.path.join(onnx_folder, f"{config_name}.onnx")
        os.makedirs(onnx_folder, exist_ok=True)

        self._export_to_onnx(
            model=model,
            output_file=onnx_path,
            input_size=image_shape,
            text_seq_len=text_seq_len,
        )

        if self._has_external_files(onnx_folder):
            print(f"🔄 Consolidating ONNX model from external files: {onnx_path}")
            self.combine_separate_onnx(onnx_path)

        print(f"✅ Successfully exported: {onnx_path}")

    def _export_to_onnx(self, model, output_file, input_size, text_seq_len, batch_size=1):
        model.eval()

        image_input = torch.randn((batch_size,) + input_size, dtype=torch.float32)
        text_input = torch.randint(0, 10000, (batch_size, text_seq_len), dtype=torch.int32)

        input_names = ["image", "text"]
        output_names = ["image_features", "text_features"]

        dynamic_axes = {
            "image": {0: "batch"},
            "text": {0: "batch", 1: "seq"},
            "image_features": {0: "batch"},
            "text_features": {0: "batch"}
        }

        torch.onnx.export(
            model,
            (image_input, text_input),
            output_file,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=self.opset
        )

    def _has_external_files(self, onnx_folder):
        files = os.listdir(onnx_folder)
        if len(files) < 3:
            return False
        else:
            return True
        

    def combine_separate_onnx(self, onnx_model_path):
        print(f"🔄 Consolidating ONNX model from external files: {onnx_model_path}")

        # Define paths
        onnx_model_folder_path = os.path.dirname(onnx_model_path)
        base_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
        files2move = os.listdir(onnx_model_folder_path)

        # 1. Create temporary folder to isolate external data files
        tmp_dir = os.path.join(onnx_model_folder_path, f"{base_name}_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        for filename in files2move:
            src = os.path.join(onnx_model_folder_path, filename)
            if os.path.isfile(src):
                dst = os.path.join(tmp_dir, filename)
                shutil.move(src, dst)

        # 3. Copy the original model to the temp location
        tmp_model_path = os.path.join(tmp_dir, base_name + ".onnx")

        # 4. Load from the temp location
        onnx_model = onnx.load(tmp_model_path, load_external_data=True)

        # 5. Consolidated model and .data output paths
        data_location = os.path.join(onnx_model_folder_path, f"{base_name}.data")
        consolidated_model_path = os.path.join(onnx_model_folder_path, f"{base_name}.onnx")

        # 6. Convert to a single .data file
        convert_model_to_external_data(
            onnx_model,
            all_tensors_to_one_file=True,
            location=os.path.basename(data_location),
            size_threshold=0,
            convert_attribute=False
        )

        # 7. Save the final consolidated ONNX model
        onnx.save_model(
            onnx_model,
            consolidated_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(data_location),
            size_threshold=0,
        )

        print(f"✅ Consolidated model saved as: {consolidated_model_path}")
        print(f"📦 External weights saved as: {data_location}")

        # 8. Clean up the temp folder
        try:
            shutil.rmtree(tmp_dir)
            print(f"🧹 Temporary folder removed: {tmp_dir}")
        except Exception as e:
            print(f"⚠️ Failed to remove temporary folder {tmp_dir}: {e}")


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
    exporter = ONNXExporter(input_folder="weights", output_folder="onnx_export", opset=17)
    exporter.export_all()
