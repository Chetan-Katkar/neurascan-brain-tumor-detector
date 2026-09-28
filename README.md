# NeuraScan — Brain Tumour Classifier

A deep-learning web app that classifies brain MRI scans into four categories using a
ResNet18 model fine-tuned with PyTorch.

**97.00% test accuracy** · 1,000 held-out scans · Classes: glioma, meningioma, no tumour, pituitary

---

## Project structure

```
neurascan-brain-tumor-detector/
├── app.py                                          Flask server + /predict API
├── brain_tumor_resnet18.pth                        Trained weights (44 MB)
├── brain_tumor_resnet18_optimized.ipynb            Training notebook
├── brain_tumor_resnet18_optimized_executed.ipynb   Executed notebook with results
├── preprocess (4).ipynb                            Dataset preprocessing
├── requirements.txt                                Python dependencies
├── templates/
│   └── index.html                                  Web UI (dark theme)
└── docs/
    ├── REPORT.md                                   Project report (Markdown source)
    ├── report.html / report.pdf                    Project report, formatted (8pp A4)
    ├── poster.html / poster.pdf / poster.png       Conference poster (A2 portrait)
    └── build_docs.py                               Regenerates the poster and report
```

### Regenerating the poster and report

Figures are inline SVG generated from the recorded training results, so no plotting
library is needed:

```bash
python3 docs/build_docs.py     # rewrites docs/poster.html and docs/report.html
```

---

## Running the web app

### 1. Install dependencies (first time only)

```bash
pip3 install -r requirements.txt
```

### 2. Start the server

```bash
python3 app.py
```

Expected output:

```
[device] mps
[model]  loaded from .../brain_tumor_resnet18.pth
[server] http://localhost:5001
 * Running on http://127.0.0.1:5001
```

The device line reads `cuda` on an NVIDIA GPU, `mps` on Apple Silicon, or `cpu` otherwise.

### 3. Open the app

Visit <http://127.0.0.1:5001> in a browser.

### 4. Use it

1. Drag an MRI image onto the upload area, or click **Browse files**.
2. Click **Analyse scan**.
3. Read the predicted class, the confidence, and the full probability distribution.

Stop the server with `Ctrl + C`.

---

## API

| Method | Route      | Body                       | Returns                                                                   |
|--------|------------|----------------------------|---------------------------------------------------------------------------|
| `GET`  | `/`        | —                          | The web UI                                                                 |
| `GET`  | `/health`  | —                          | `{"status": "ok", "device": "..."}`                                        |
| `POST` | `/predict` | `multipart/form-data` with an `image` field | `predicted_class`, `confidence`, `all_probabilities`, `image_preview` |

```bash
curl -F "image=@scan.jpg" http://127.0.0.1:5001/predict
```

---

## Supported image formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.tif` · `.tiff` — up to 10 MB.

---

## Tumour classes

| Class | Description |
|-------|-------------|
| **Glioma** | Arises from glial cells; ranges from slow-growing to highly aggressive. |
| **Meningioma** | Arises from the meninges; usually slow-growing and benign. |
| **No tumour** | No tumour signature detected in the scan. |
| **Pituitary** | Forms in the pituitary gland; mostly benign adenomas, usually treatable. |

---

## Model

| Property | Value |
|----------|-------|
| Architecture | ResNet18 (ImageNet pretrained) |
| Parameters | 11,178,564 total |
| Input | 224 × 224, greyscale expanded to 3 channels |
| Augmentation | Random horizontal flip only |
| Normalisation | ImageNet mean/std |
| Phase 1 | 5 epochs, frozen backbone, head only, Adam LR 1e-4 |
| Phase 2 | 15 epochs, `layer4` + head unfrozen, Adam LR 5e-5, StepLR |
| Split | 4,000 train / 1,000 validation (stratified 80:20) / 1,000 test |
| Test accuracy | **97.00%** |
| Macro F1 | 0.9715 |

Full results, per-class metrics and the confusion matrix are in [`docs/REPORT.md`](docs/REPORT.md).
A one-page summary for presentation is in [`docs/poster.html`](docs/poster.html).

---

## Disclaimer

This tool is for research and educational purposes only. It is not a medical device and
must not be used for clinical diagnosis. Always consult a qualified medical professional.

---

Built with PyTorch, torchvision and Flask.
