"""
Streamlit web app for the Intelligent Waste Classification System.

Run locally:
    streamlit run streamlit_app/app.py

Deploy on Streamlit Community Cloud:
    1. Push this whole project to a public (or private) GitHub repo.
    2. Go to https://share.streamlit.io -> "New app".
    3. Point it at your repo, branch = main, main file path =
       streamlit_app/app.py.
    4. Deploy. Streamlit Cloud installs streamlit_app/requirements.txt
       automatically and builds the app for you.

The app expects a trained model at models/mobilenet_waste.keras (or
models/baseline_cnn_waste.keras as a fallback) and models/class_names.json,
both produced by src/train_mobilenet.py / src/train_baseline_cnn.py — see
notebooks/Waste_Classification_Colab.ipynb to produce these on a free GPU.
"""

import os
import sys
import json

import numpy as np
import streamlit as st
from PIL import Image

# Make src/ importable regardless of the working directory Streamlit is
# launched from (both locally and on Streamlit Cloud).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from data_utils import preprocess_image_array, IMG_SIZE  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "mobilenet_waste.keras"),
    os.path.join(MODELS_DIR, "baseline_cnn_waste.keras"),
]
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

st.set_page_config(
    page_title="Intelligent Waste Classification",
    page_icon="\u267b\ufe0f",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Load the trained Keras model once and cache it across reruns/users.

    Returns (model, class_names, model_path) or raises FileNotFoundError
    with a message explaining exactly what's missing.
    """
    from tensorflow.keras.models import load_model  # imported lazily so the
    # rest of the app can still render a helpful message if TensorFlow or
    # the model file isn't available yet.

    model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(
            "No trained model found in models/. Train one first with "
            "src/train_mobilenet.py or src/train_baseline_cnn.py (see "
            "notebooks/Waste_Classification_Colab.ipynb), then place the "
            "resulting .keras file in the models/ folder."
        )

    model = load_model(model_path)

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    else:
        class_names = [f"class_{i}" for i in range(model.output_shape[-1])]

    return model, class_names, model_path


def classify(model, class_names, image: Image.Image):
    x = preprocess_image_array(image, img_size=IMG_SIZE)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(probs.argmax())
    return {
        "predicted_class": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probabilities": {class_names[i]: float(p) for i, p in enumerate(probs)},
    }


def render_result(result):
    st.success(f"**Prediction: {result['predicted_class'].capitalize()}**")
    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

    st.write("Full probability breakdown:")
    sorted_probs = sorted(result["all_probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    for cls, p in sorted_probs:
        st.progress(p, text=f"{cls.capitalize()}: {p * 100:.1f}%")


def main():
    st.title("\u267b\ufe0f Intelligent Waste Classification System")
    st.caption(
        "Upload a photo of a waste item and the system will classify it as "
        "plastic, paper, metal, glass, or organic waste."
    )

    try:
        model, class_names, model_path = load_trained_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "This app is fully built and working — it just needs a trained "
            "model file. Train one in Google Colab using "
            "notebooks/Waste_Classification_Colab.ipynb, download the "
            ".keras file it produces, and drop it into the models/ folder "
            "of this project (locally, or committed to your GitHub repo "
            "before deploying)."
        )
        st.stop()

    st.sidebar.header("About")
    st.sidebar.write(
        f"Model in use: `{os.path.basename(model_path)}`\n\n"
        f"Classes: {', '.join(class_names)}"
    )
    st.sidebar.write(
        "Built for a Nigerian-context final-year project on intelligent "
        "waste classification using computer vision and deep learning."
    )

    uploaded_file = st.file_uploader(
        "Choose a waste image", type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify", type="primary"):
            with st.spinner("Classifying..."):
                result = classify(model, class_names, image)
            render_result(result)


if __name__ == "__main__":
    main()
