import os
import glob
import numpy as np
import onnxruntime as ort
from PIL import Image

import torch
import torch.nn.functional as F

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except Exception:
    trt = None  # type: ignore
    TRT_AVAILABLE = False


# ---- helpers ----
def _trt_to_torch_dtype(dt):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.INT8:  torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL:  torch.bool,
    }[dt]

def _run_v3(context):
    if hasattr(context, "execute_v3"):
        return context.execute_v3()  # 동기
    elif hasattr(context, "enqueue_v3"):
        return context.enqueue_v3(0)  # 기본 CUDA stream(0)
    elif hasattr(context, "execute_async_v3"):
        return context.execute_async_v3(stream_handle=0)  # 기본 스트림
    else:
        raise RuntimeError("TensorRT v3 execution API not found in this build.")

def _providers():
    # CUDA EP가 실제로 가능한지 체크
    provs = ort.get_available_providers()
    if "CUDAExecutionProvider" in provs:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def _to_numpy(t):
    return t.detach().cpu().numpy()

def _print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


class CLIPONNXSuiteTester:
    def __init__(self, image_path, texts, onnx_root="onnx_export", trt_root="trt_export",
                 test_monolith=True, test_vision=True, test_text=True,
                 test_logit=True, test_modular=True, test_trt=False):
        self.image_path = image_path
        self.texts = texts
        self.onnx_root = onnx_root
        self.trt_root = trt_root
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
        ins = {
            sess.get_inputs()[0].name: _to_numpy(image).astype(np.float32, copy=False),
            sess.get_inputs()[1].name: _to_numpy(text).astype(np.int32, copy=False),
        }
        outs = sess.run(None, ins)
        image_feat = torch.from_numpy(outs[0]).cuda().float()
        text_feat  = torch.from_numpy(outs[1]).cuda().float()
        logit_scale = torch.from_numpy(outs[2]).cuda().float()
        sim = (logit_scale * image_feat @ text_feat.T).softmax(dim=-1)
        return image_feat, text_feat, logit_scale, sim

    def test_vision(self, path, image):
        sess = ort.InferenceSession(path, providers=self.providers)
        ins = {sess.get_inputs()[0].name: _to_numpy(image).astype(np.float32, copy=False)}
        image_feat = torch.from_numpy(sess.run(None, ins)[0]).cuda().float()
        return image_feat

    # ---------- TensorRT (vision only; CUDA torch.Tensor 입력 가정) ----------
    def test_trt_vision(self, path, image_cuda: torch.Tensor):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
        assert image_cuda.is_cuda, "input must be a CUDA torch.Tensor"

        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        num = engine.num_io_tensors
        tensor_names = [engine.get_tensor_name(i) for i in range(num)]
        inp_names  = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        out_names  = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        assert len(inp_names) == 1, "this sample assumes a single input"

        input_name = inp_names[0]

        # 엔진이 기대하는 입력 dtype에 맞춰 캐스팅/contiguous
        trt_in_dtype   = engine.get_tensor_dtype(input_name)
        torch_in_dtype = _trt_to_torch_dtype(trt_in_dtype)
        if image_cuda.dtype != torch_in_dtype:
            image_cuda = image_cuda.to(dtype=torch_in_dtype)
        if not image_cuda.is_contiguous():
            image_cuda = image_cuda.contiguous()

        with engine.create_execution_context() as context:
            # 동적 입력 shape 설정
            context.set_input_shape(input_name, tuple(image_cuda.shape))

            # 출력 텐서들 준비 (CUDA 텐서로 미리 할당 → data_ptr 바인딩)
            out_tensors = {}
            for name in out_names:
                dims = context.get_tensor_shape(name)  # tensorrt.Dims
                out_shape = tuple(int(d) for d in dims)  # 확정 shape로 변환
                trt_out_dtype   = engine.get_tensor_dtype(name)
                torch_out_dtype = _trt_to_torch_dtype(trt_out_dtype)
                out_tensors[name] = torch.empty(out_shape, dtype=torch_out_dtype, device="cuda")

            # 바인딩 주소 채우기 (zero-copy)
            bindings = [0] * num
            bindings[tensor_names.index(input_name)] = int(image_cuda.data_ptr())
            for name in out_names:
                bindings[tensor_names.index(name)] = int(out_tensors[name].data_ptr())

            # 컨텍스트에 주소 등록
            for i in range(num):
                name = engine.get_tensor_name(i)
                context.set_tensor_address(name, bindings[i])

            # 실행
            ok = _run_v3(context)
            if ok is False:
                raise RuntimeError("TensorRT v3 execution failed")

        # 출력 하나면 텐서만, 여러 개면 dict 반환
        if len(out_names) == 1:
            return out_tensors[out_names[0]]
        return out_tensors

    def test_text(self, path, text):
        sess = ort.InferenceSession(path, providers=self.providers)
        ins = {sess.get_inputs()[0].name: _to_numpy(text).astype(np.int32, copy=False)}
        text_feat = torch.from_numpy(sess.run(None, ins)[0]).cuda().float()
        return text_feat

    def test_logit_scale(self, path):
        sess = ort.InferenceSession(path, providers=self.providers)
        if len(sess.get_inputs()) == 0:
            out = sess.run(None, {})
        else:
            # 동적 입력이면 1로 채워 실행
            shp = tuple(dim if isinstance(dim, int) and dim > 0 else 1 for dim in sess.get_inputs()[0].shape)
            dummy = np.zeros(shp, dtype=np.float32)
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
                 "logit": None}
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
                cos_pt_onx = F.cosine_similarity(pt_out[0], vision_feat).item()
                mse_pt_onx = F.mse_loss(pt_out[0], vision_feat).item()
                print(f"[Vision ONNX] shape={vision_feat.shape}")
                print(f"  CosSim(PT vs ONNX): {cos_pt_onx:.6f}")
                print(f"  MSE   (PT vs ONNX): {mse_pt_onx:.8f}")

                if self.test_trt_flag:
                    trt_paths = glob.glob(os.path.join(self.trt_root, "*.engine"))
                    for trt_path in trt_paths:
                        trt_feat = self.test_trt_vision(trt_path, image)
                        cos_pt_trt = F.cosine_similarity(pt_out[0], trt_feat).item()
                        mse_pt_trt = F.mse_loss(pt_out[0], trt_feat).item()
                        cos_onx_trt = F.cosine_similarity(vision_feat, trt_feat).item()
                        mse_onx_trt = F.mse_loss(vision_feat, trt_feat).item()
                        print(f"[Vision TRT] shape={trt_feat.shape}  file={os.path.basename(trt_path)}")
                        print(f"  CosSim(PT vs TRT):   {cos_pt_trt:.6f}")
                        print(f"  MSE   (PT vs TRT):   {mse_pt_trt:.8f}")
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
        trt_root="trt_export",
        test_monolith=True,
        test_vision=True,
        test_text=True,
        test_logit=True,
        test_modular=True,
        test_trt=True
    )
    tester.run()
