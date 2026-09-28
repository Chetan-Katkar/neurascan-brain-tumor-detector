# 🧠 NeuraScan — Brain Tumor AI Classifier

A deep learning web app that classifies brain MRI scans into 4 categories using a ResNet18 model trained with PyTorch.

**Test Accuracy: 97%** | **Device: Apple MPS (Mac GPU)** | **Classes: Glioma · Meningioma · No Tumor · Pituitary**

---

## 📁 Project Structure

```
Brain Tumor Detector/
├── app.py                               ← Flask web server + /predict API
├── brain_tumor_resnet18.pth             ← Trained model weights (44 MB)
├── brain_tumor_resnet18_optimized.ipynb ← Training notebook (clean)
├── brain_tumor_resnet18_optimized_executed.ipynb ← Executed notebook with results
├── templates/
│   └── index.html                       ← Web UI (dark glassmorphism frontend)
├── server.log                           ← Server output log
└── README.md                            ← This file
```

---

## 🚀 How to Run the Web App (Step by Step)

### Step 1 — Open Terminal

Open the **Terminal** app on your Mac.

---

### Step 2 — Navigate to the Project Folder

```bash
cd "/Users/swaroopnaik1905/Brain Tumor Detector"
```

---

### Step 3 — Install Dependencies (First Time Only)

Run this once to install the required Python packages:

```bash
pip3 install flask flask-cors torch torchvision pillow
```

> ✅ If you already ran this before, skip to Step 4.

---

### Step 4 — Start the Server

```bash
python3 app.py
```

You should see output like:

```
🖥️  Using device: mps
✅ Model loaded from: .../brain_tumor_resnet18.pth
🚀 Starting server at http://localhost:5001
 * Running on http://127.0.0.1:5001
```

---

### Step 5 — Open the App in Browser

Open your browser (Chrome/Safari/Firefox) and go to:

```
http://127.0.0.1:5001
```

---

### Step 6 — Use the App

1. **Drag & drop** an MRI image onto the upload zone  
   — OR — click **"Choose File"** to browse your files
2. Click **"✦ Analyze Scan"**
3. View the predicted tumor class, confidence %, and probability bars

---

### Step 7 — Stop the Server

Press `Ctrl + C` in the Terminal window to stop the server.

---

## 🔁 Quick Restart (After First Setup)

Each time you want to use the app again:

```bash
cd "/Users/swaroopnaik1905/Brain Tumor Detector"
python3 app.py
```

Then open **http://127.0.0.1:5001** in your browser.

---

## 🧪 Supported Image Formats

| Format | Supported |
|--------|-----------|
| `.jpg` / `.jpeg` | ✅ Yes |
| `.png` | ✅ Yes |
| `.bmp` | ✅ Yes |
| `.tif` / `.tiff` | ✅ Yes |

---

## 🏷️ Tumor Classes

| Class | Color | Description |
|-------|-------|-------------|
| **Glioma** | 🔴 Red | Tumor from glial cells; most aggressive type |
| **Meningioma** | 🟠 Orange | Tumor from brain's outer membrane; usually benign |
| **No Tumor** | 🟢 Green | No tumor detected in the MRI scan |
| **Pituitary** | 🟣 Purple | Tumor in the pituitary gland; usually treatable |

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Architecture | ResNet18 |
| Pre-training | ImageNet weights |
| Fine-tuning | 2-phase transfer learning |
| Phase 1 | 5 epochs — head only (LR = 1e-4) |
| Phase 2 | 15 epochs — layer4 + head (LR = 5e-5) |
| Input size | 224 × 224 px, grayscale → 3-channel |
| Optimizer | Adam + StepLR scheduler |
| Test accuracy | **97.00%** |

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**.  
Always consult a qualified medical professional for clinical diagnosis.

---

*Built with PyTorch · Flask · ResNet18 · Apple MPS*
