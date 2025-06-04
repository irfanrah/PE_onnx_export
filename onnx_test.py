import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import torch.nn.functional as F
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms


class CLIPONNXTester:
    def __init__(self, config_name, onnx_path, image_path, texts):
        self.config_name = config_name
        self.onnx_path = onnx_path
        self.image_path = image_path
        self.texts = texts


        print(f"📁 Testing model: {self.config_name}")

        self._load_model()
        self._prepare_inputs()

    def _load_model(self):
        self.model = pe.CLIP.from_config(self.config_name, pretrained=True).cuda().eval()
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

    def _prepare_inputs(self):
        image_pil = Image.open(self.image_path)
        self.image = self.preprocess(image_pil).unsqueeze(0).cuda()
        self.text = self.tokenizer(self.texts).cuda()
        print("Input shapes:", self.image.shape, self.text.shape)

    def run_torch_inference(self):
        with torch.no_grad(), torch.autocast("cuda"):
            image_feat, text_feat, logit_scale = self.model(self.image, self.text)
            sim_score = (logit_scale * image_feat @ text_feat.T).softmax(dim=-1)

        return image_feat, text_feat, logit_scale, sim_score

    def run_onnx_inference(self):
        image_cpu = self.image.cpu().numpy().astype(np.float32)
        text_cpu = self.text.cpu().numpy().astype(np.int32)

        ort_session = ort.InferenceSession(self.onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        input_names = [i.name for i in ort_session.get_inputs()]
        output_names = [o.name for o in ort_session.get_outputs()]
        print("ONNX input names:", input_names)
        print("ONNX output names:", output_names)

        input_feed = {
            input_names[0]: image_cpu,
            input_names[1]: text_cpu
        }

        outputs = ort_session.run(None, input_feed)
        image_feat = torch.from_numpy(outputs[0]).cuda().float()
        text_feat = torch.from_numpy(outputs[1]).cuda().float()
        logit_scale = torch.from_numpy(outputs[2]).cuda().float()

        sim_score = (
            logit_scale * image_feat @ text_feat.T
        ).softmax(dim=-1)

        return image_feat, text_feat, logit_scale, sim_score

    def compare_outputs(self):
        image_feat_pt, text_feat_pt, logit_scale_pt, sim_score_pt = self.run_torch_inference()
        image_feat_onnx, text_feat_onnx, logit_scale_onnx, sim_score_onnx = self.run_onnx_inference()

        image_sim = F.cosine_similarity(image_feat_pt, image_feat_onnx).item()
        text_sim = F.cosine_similarity(text_feat_pt, text_feat_onnx, dim=-1).mean().item()
        logit_scale_mse = F.mse_loss(logit_scale_pt, logit_scale_onnx).item()
        cos_sim = F.cosine_similarity(sim_score_pt, sim_score_onnx).item()
        mse = F.mse_loss(sim_score_pt, sim_score_onnx).item()

        print("\nResults")
        print(f"Image features similarity: {image_sim:.4f}")
        print(f"Text features similarity:  {text_sim:.4f}")
        print(f"logit_scale_mse:    {logit_scale_mse:.4f}")
        print(f"\nPyTorch V-T sim_score: {sim_score_pt.cpu().numpy()}")
        print(f"ONNX V-T sim_score:    {sim_score_onnx.cpu().numpy()}")
        print(f"Cosine similarity: {cos_sim:.4f}")
        print(f"MSE:               {mse:.6f}")

        if cos_sim > 0.95:
            print("✅ ONNX output is sufficiently similar (≥95%) to PyTorch")
        else:
            print("❌ ONNX output differs significantly. Check export fidelity.")

    def run_all(self):
        self.compare_outputs()

if __name__ == "__main__":
    tester = CLIPONNXTester(
        config_name="PE-Core-B16-224",
        onnx_path="onnx_export/PE-Core-B16-224/PE-Core-B16-224.onnx",
        image_path="assets/dog.jpg",
        texts=["a diagram", "a dog", "a cat"]
    )
    tester.run_all()
