"""
Brain Tumor Classification — Flask Backend
Serves the ResNet18 model for MRI image classification.
"""

import os
import io
import base64
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'brain_tumor_resnet18.pth')
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMG_SIZE    = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ─── Device ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"[device] {device}")

# ─── Load Model ──────────────────────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f"[model]  loaded from {MODEL_PATH}")

# ─── Transform ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
CORS(app)


@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'r') as f:
        return f.read()


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Load and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Save a thumbnail for display (base64-encoded)
        thumb = img.copy()
        thumb.thumbnail((400, 400))
        buf = io.BytesIO()
        thumb.save(buf, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Run inference
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]

        # Build result
        pred_idx   = probs.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item() * 100

        class_probs = {
            cls: round(float(probs[i].item()) * 100, 2)
            for i, cls in enumerate(CLASS_NAMES)
        }

        return jsonify({
            'predicted_class': pred_class,
            'confidence':       round(confidence, 2),
            'all_probabilities': class_probs,
            'image_preview':    img_b64,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    print("[server] http://localhost:5001")
    app.run(debug=False, host='0.0.0.0', port=5001)
