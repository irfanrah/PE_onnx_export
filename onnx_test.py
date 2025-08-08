import os
import glob
import math
import numpy as np
import onnxruntime as ort
from PIL import Image

import torch
import torch.nn.functional as F

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms


def _providers():
    # CUDA if available, fallback to CPU
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        _ = ort.get_device()
    except Exception:
        prov = ["CPUExecutionProvider"]
    return prov


def _to_numpy(t):
    return t.detach().cpu().numpy()


def _print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
class CLIPONNXSuiteTester:
    def __init__(self, image_path, texts, onnx_root="onnx_export",
                 test_monolith=True, test_vision=True, test_text=True, test_logit=True, test_modular=True):
        self.image_path = image_path
        self.texts = texts
        self.onnx_root = onnx_root
        self.providers = _providers()

        # test flags
        self.test_monolith_flag = test_monolith
        self.test_vision_flag = test_vision
        self.test_text_flag = test_text
        self.test_logit_flag = test_logit
        self.test_modular_flag = test_modular

    # ---------- PyTorch reference ----------
    def _load_torch(self, config_name):
        model = pe.CLIP.from_config(config_name, pretrained=True).cuda().eval()
        preprocess = transforms.get_image_transform(model.image_size)
        tokenizer = transforms.get_text_tokenizer(model.context_length)
        image = preprocess(Image.open(self.image_path)).unsqueeze(0).cuda()
        text = tokenizer(self.texts).cuda()
        return model, image, text

    @torch.inference_mode()
    def _torch_forward(self, model, image, text):
        with torch.autocast("cuda"):
            image_feat, text_feat, logit_scale = model(image, text)
            sim = (logit_scale * image_feat @ text_feat.T).softmax(dim=-1)
        return image_feat, text_feat, logit_scale, sim

    # ---------- ONNX parts ----------
    def test_monolith(self, path, image, text):
        sess = ort.InferenceSession(path, providers=self.providers)
        ins = {sess.get_inputs()[0].name: _to_numpy(image).astype(np.float32),
               sess.get_inputs()[1].name: _to_numpy(text).astype(np.int32)}
        outs = sess.run(None, ins)
        image_feat = torch.from_numpy(outs[0]).cuda().float()
        text_feat = torch.from_numpy(outs[1]).cuda().float()
        logit_scale = torch.from_numpy(outs[2]).cuda().float()
        sim = (logit_scale * image_feat @ text_feat.T).softmax(dim=-1)
        return image_feat, text_feat, logit_scale, sim

    def test_vision(self, path, image):
        sess = ort.InferenceSession(path, providers=self.providers)
        ins = {sess.get_inputs()[0].name: _to_numpy(image).astype(np.float32)}
        image_feat = torch.from_numpy(sess.run(None, ins)[0]).cuda().float()
        return image_feat

    def test_text(self, path, text):
        sess = ort.InferenceSession(path, providers=self.providers)
        ins = {sess.get_inputs()[0].name: _to_numpy(text).astype(np.int32)}
        text_feat = torch.from_numpy(sess.run(None, ins)[0]).cuda().float()
        return text_feat

    def test_logit_scale(self, path):
        sess = ort.InferenceSession(path, providers=self.providers)
        if len(sess.get_inputs()) == 0:
            out = sess.run(None, {})
        else:
            dummy_shape = tuple(i.shape for i in sess.get_inputs())[0]
            dummy = np.zeros(dummy_shape, dtype=np.float32)
            out = sess.run(None, {sess.get_inputs()[0].name: dummy})
        logit_scale = torch.from_numpy(out[0]).cuda().float()
        return logit_scale

    def test_modular(self, vision_path, text_path, logit_scale_path, image, text):
        image_feat = self.test_vision(vision_path, image)
        text_feat = self.test_text(text_path, text)
        logit_scale = self.test_logit_scale(logit_scale_path)
        sim = (logit_scale * image_feat @ text_feat.T).softmax(dim=-1)
        return image_feat, text_feat, logit_scale, sim

    # ---------- metrics ----------
    def _report(self, tag, pt, onx):
        img_cos = F.cosine_similarity(pt[0], onx[0]).item()
        txt_cos = F.cosine_similarity(pt[1], onx[1], dim=-1).mean().item()
        log_mse = F.mse_loss(pt[2], onx[2]).item()
        sim_cos = F.cosine_similarity(pt[3], onx[3]).item()
        sim_mse = F.mse_loss(pt[3], onx[3]).item()

        print(f"\n[{tag}]")
        print(f"  Image feat CosSim : {img_cos:.6f}")
        print(f"  Text  feat CosSim : {txt_cos:.6f}")
        print(f"  logit_scale   MSE : {log_mse:.8f}")
        print(f"  Softmax(sim) CosS : {sim_cos:.6f}")
        print(f"  Softmax(sim)  MSE : {sim_mse:.8f}")

    def _find_parts(self, cfg_dir, cfg_name):
        parts = {"monolith": None, "vision": None, "text": None, "logit": None}
        for f in glob.glob(os.path.join(cfg_dir, f"{cfg_name}_*.onnx")):
            if "_monolith.onnx" in f:
                parts["monolith"] = f
            elif "_vision.onnx" in f:
                parts["vision"] = f
            elif "_text.onnx" in f:
                parts["text"] = f
            elif "_logit_scale.onnx" in f:
                parts["logit"] = f
        return parts

    def run(self):
        for cfg_dir in sorted(glob.glob(os.path.join(self.onnx_root, "*"))):
            if not os.path.isdir(cfg_dir):
                continue
            cfg_name = os.path.basename(cfg_dir)
            _print_header(f"Testing {cfg_name}")

            model, image, text = self._load_torch(cfg_name)
            pt_out = self._torch_forward(model, image, text)
            parts = self._find_parts(cfg_dir, cfg_name)

            if self.test_monolith_flag and parts["monolith"]:
                onx_out = self.test_monolith(parts["monolith"], image, text)
                self._report("ONNX monolith vs PyTorch", pt_out, onx_out)

            if self.test_modular_flag and all(parts[k] for k in ("vision", "text", "logit")):
                onx_out = self.test_modular(parts["vision"], parts["text"], parts["logit"], image, text)
                self._report("ONNX modular vs PyTorch", pt_out, onx_out)

            if self.test_vision_flag and parts["vision"]:
                vision_feat = self.test_vision(parts["vision"], image)
                print(f"[Vision only] shape={vision_feat.shape}")

            if self.test_text_flag and parts["text"]:
                text_feat = self.test_text(parts["text"], text)
                print(f"[Text only] shape={text_feat.shape}")

            if self.test_logit_flag and parts["logit"]:
                logit_scale = self.test_logit_scale(parts["logit"])
                print(f"[Logit only] value={logit_scale.item():.6f}")


if __name__ == "__main__":
    tester = CLIPONNXSuiteTester(
        image_path="assets/dog.jpg",
        texts=["a diagram", "a dog", "a cat"],
        onnx_root="onnx_export",
        test_monolith=True,
        test_vision=True,
        test_text=True,
        test_logit=True,
        test_modular=True
    )
    tester.run()
