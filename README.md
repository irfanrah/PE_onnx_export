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

### Model: `PE-Core-B16-224`

```
📁 Testing model: PE-Core-B16-224
Missing keys for loading model: []
Unexpected keys for loading model: []
Input shapes: torch.Size([1, 3, 224, 224]) torch.Size([3, 32])
ONNX input names: ['image', 'text']
ONNX output names: ['image_features', 'text_features', '9485']

Results
Image features similarity: 1.0000
Text features similarity:  1.0000

PyTorch probs: [[3.6975723e-06 9.9997842e-01 1.7917922e-05]]
ONNX probs:    [[3.7763546e-06 9.9997818e-01 1.8028752e-05]]

Cosine similarity: 1.0000
MSE:               0.000000

✅ ONNX output is sufficiently similar (≥95%) to PyTorch
```

---

### Model: `PE-Core-L14-336`

```
📁 Testing model: PE-Core-L14-336
Missing keys for loading model: []
Unexpected keys for loading model: []
Input shapes: torch.Size([1, 3, 336, 336]) torch.Size([3, 32])
ONNX input names: ['image', 'text']
ONNX output names: ['image_features', 'text_features', '13612']

Results
Image features similarity: 1.0000
Text features similarity:  1.0000

PyTorch probs: [[1.2530895e-06 9.9994528e-01 5.3491265e-05]]
ONNX probs:    [[1.2691744e-06 9.9994433e-01 5.4417895e-05]]

Cosine similarity: 1.0000
MSE:               0.000000

✅ ONNX output is sufficiently similar (≥95%) to PyTorch
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

