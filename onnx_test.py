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

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 - initializes CUDA context
    TRT_AVAILABLE = True
except Exception:
    trt = None  # type: ignore
    cuda = None  # type: ignore
    TRT_AVAILABLE = False


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
                 test_monolith=True, test_vision=True, test_text=True,
                 test_logit=True, test_modular=True, test_trt=False):
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
        self.test_trt_flag = test_trt and TRT_AVAILABLE

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

    def test_trt_vision(self, path, image):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        for idx in range(engine.num_bindings):
            if engine.binding_is_input(idx):
                inp_idx = idx
            else:
                out_idx = idx
        with engine.create_execution_context() as context:
            context.set_binding_shape(inp_idx, tuple(image.shape))
            inp = _to_numpy(image).astype(np.float32)
            out_shape = tuple(context.get_binding_shape(out_idx))
            d_in = cuda.mem_alloc(inp.nbytes)
            out = np.empty(out_shape, dtype=np.float32)
            d_out = cuda.mem_alloc(out.nbytes)
            stream = cuda.Stream()
            cuda.memcpy_htod_async(d_in, inp, stream)
            bindings = [0] * engine.num_bindings
            bindings[inp_idx] = int(d_in)
            bindings[out_idx] = int(d_out)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(out, d_out, stream)
            stream.synchronize()
        image_feat = torch.from_numpy(out).cuda().float()
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
        parts = {"monolith": None, "vision": None, "text": None,
                 "logit": None, "vision_trt": None}
        for f in glob.glob(os.path.join(cfg_dir, f"{cfg_name}_*.onnx")):
            if "_monolith.onnx" in f:
                parts["monolith"] = f
            elif "_vision.onnx" in f:
                parts["vision"] = f
            elif "_text.onnx" in f:
                parts["text"] = f
            elif "_logit_scale.onnx" in f:
                parts["logit"] = f
        engine_path = os.path.join(cfg_dir, "vision.engine")
        if os.path.exists(engine_path):
            parts["vision_trt"] = engine_path
        alt_engine = os.path.join(cfg_dir, f"{cfg_name}_vision.engine")
        if os.path.exists(alt_engine):
            parts["vision_trt"] = alt_engine
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
                cos_pt_onx = F.cosine_similarity(pt_out[0], vision_feat).item()
                mse_pt_onx = F.mse_loss(pt_out[0], vision_feat).item()
                print(f"[Vision ONNX] shape={vision_feat.shape}")
                print(f"  CosSim(PT vs ONNX): {cos_pt_onx:.6f}")
                print(f"  MSE   (PT vs ONNX): {mse_pt_onx:.8f}")

                if self.test_trt_flag and parts.get("vision_trt"):
                    trt_feat = self.test_trt_vision(parts["vision_trt"], image)
                    cos_pt_trt = F.cosine_similarity(pt_out[0], trt_feat).item()
                    mse_pt_trt = F.mse_loss(pt_out[0], trt_feat).item()
                    cos_onx_trt = F.cosine_similarity(vision_feat, trt_feat).item()
                    mse_onx_trt = F.mse_loss(vision_feat, trt_feat).item()
                    print(f"[Vision TRT] shape={trt_feat.shape}")
                    print(f"  CosSim(PT vs TRT): {cos_pt_trt:.6f}")
                    print(f"  MSE   (PT vs TRT): {mse_pt_trt:.8f}")
                    print(f"  CosSim(ONNX vs TRT): {cos_onx_trt:.6f}")
                    print(f"  MSE   (ONNX vs TRT): {mse_onx_trt:.8f}")

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
        test_modular=True,
        test_trt=True
    )
    tester.run()
