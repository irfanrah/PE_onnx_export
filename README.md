# PE ONNX Export

Unofficial implementation for exporting [Perception Encoder (PE)] models to ONNX format.

## 🔧 Features

- ✅ Converts unsupported layers in PE to ONNX-compatible layers
- ✅ Supports consolidation of ONNX external weights (ONNX automatically splits weights if >2GB)
- ✅ Includes validation script to compare `.pt` and `.onnx` outputs for:
  - Image features
  - Text features
  - Similarity score

---

## 🚀 How to Run

1. **Install PE package**
   ```bash
   cd perception_models
   pip install -e .
    ````

2. **Download pretrained weights**

   * Option 1: Download official weights from Hugging Face
   * Option 2: Use your custom PE weights

   Place them into the `weights/` folder.

3. **Export ONNX**

   ```bash
   python export.py
   ```

   This generates ONNX models in the `onnx/` directory based on the weights found in `weights/`.

4. **Test ONNX vs PyTorch equivalence**

   ```bash
   python onnx_test.py
   ```

---

## ✅ Sample Output

```bash
(onnx) kurnianto@pia:~/code/PE_onnx_export$ python onnx_test.py 

================================================================================
Testing PE-Core-B16-224
================================================================================
Missing keys for loading model: []
Unexpected keys for loading model: []

/home/kurnianto/anaconda3/envs/onnx/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:121: UserWarning: 
Specified provider 'CUDAExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
  warnings.warn()

[ONNX monolith vs PyTorch]
  Image feat CosSim : 0.999982
  Text  feat CosSim : 1.000000
  logit_scale   MSE : 0.00000000
  Softmax(sim) CosS : 1.000000
  Softmax(sim)  MSE : 0.00000000

/home/kurnianto/code/PE_onnx_export/onnx_test.py:110: UserWarning: 
Using a target size (torch.Size([1])) that is different from the input size (torch.Size([])).
This may lead to incorrect results due to broadcasting. Please ensure they match in size.
  log_mse = F.mse_loss(pt[2], onx[2]).item()

[ONNX modular vs PyTorch]
  Image feat CosSim : 0.999982
  Text  feat CosSim : 1.000000
  logit_scale   MSE : 0.00000000
  Softmax(sim) CosS : 1.000000
  Softmax(sim)  MSE : 0.00000000

[Vision only] shape = torch.Size([1, 1024])
[Text only]   shape = torch.Size([3, 1024])
[Logit only]  value = 99.926315

================================================================================
Testing PE-Core-L14-336
================================================================================
Missing keys for loading model: []
Unexpected keys for loading model: []

[ONNX monolith vs PyTorch]
  Image feat CosSim : 0.999992
  Text  feat CosSim : 1.000000
  logit_scale   MSE : 0.00000000
  Softmax(sim) CosS : 1.000000
  Softmax(sim)  MSE : 0.00000000

[ONNX modular vs PyTorch]
  Image feat CosSim : 0.999992
  Text  feat CosSim : 1.000000
  logit_scale   MSE : 0.00000000
  Softmax(sim) CosS : 1.000000
  Softmax(sim)  MSE : 0.00000000

[Vision only] shape = torch.Size([1, 1024])
[Text only]   shape = torch.Size([3, 1024])
[Logit only]  value = 99.885727
```


---

## 📁 Folder Structure

```
PE_onnx_export/
├── weights/              # Place .pt weight files here
├── onnx/                 # Output folder for exported ONNX files
├── export.py             # ONNX export script
├── onnx_test.py          # ONNX vs PyTorch validation script
└── README.md             # This file
```

