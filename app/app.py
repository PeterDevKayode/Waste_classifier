"""
Flask web app for real-time waste classification — Chapter 3.5.

Loads the trained model once at start-up and exposes:
  GET  /            -> upload page
  POST /predict      -> accepts an image file, returns JSON prediction

Usage:
    cd app
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import sys
import json

from flask import Flask, request, jsonify, render_template
from PIL import Image
from tensorflow.keras.models import load_model

# Allow importing data_utils.py from ../src without installing the project as a package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import preprocess_image_array, IMG_SIZE  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Prefer the MobileNetV2 model (lighter, per Chapter 3.6.3 comparison); fall
# back to the baseline CNN if that's the only one available.
MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "mobilenet_waste.keras"),
    os.path.join(MODELS_DIR, "baseline_cnn_waste.keras"),
]
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_model = None
_class_names = None


def get_model():
    """Lazily load the model once and cache it (avoids reloading per request)."""
    global _model, _class_names
    if _model is None:
        model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError(
                "No trained model found in models/. Train one first "
                "(see src/train_mobilenet.py or src/train_baseline_cnn.py)."
            )
        _model = load_model(model_path)

        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH) as f:
                _class_names = json.load(f)
        else:
            _class_names = [f"class_{i}" for i in range(_model.output_shape[-1])]
    return _model, _class_names


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    try:
        img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    try:
        model, class_names = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    x = preprocess_image_array(img, img_size=IMG_SIZE)
    probs = model.predict(x, verbose=0)[0]

    top_idx = int(probs.argmax())
    result = {
        "predicted_class": class_names[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "all_probabilities": {
            class_names[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }
    return jsonify(result)


@app.route("/health")
def health():
    """Simple health-check endpoint, also useful for integration tests."""
    try:
        get_model()
        return jsonify({"status": "ok"})
    except FileNotFoundError as e:
        return jsonify({"status": "no_model", "detail": str(e)}), 503


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
