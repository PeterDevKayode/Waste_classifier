"""
Unit tests for streamlit_app/app.py's core logic (classify, render_result).

These test the pure functions that don't depend on Streamlit's runtime
(session state, widgets), using a dummy model — no trained model file or
TensorFlow installation is required to run these.

Run with: pytest tests/
"""

import os
import sys
import json
import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


class DummyModel:
    output_shape = (None, 3)

    def predict(self, x, verbose=0):
        return np.array([[0.7, 0.2, 0.1]], dtype=np.float32)


@pytest.fixture
def app_module(monkeypatch):
    """Import streamlit_app/app.py with tensorflow and streamlit stubbed out,
    so these tests run without either dependency installed."""
    import types

    tf = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    models_mod = types.ModuleType("tensorflow.keras.models")
    preprocessing = types.ModuleType("tensorflow.keras.preprocessing")
    image_mod = types.ModuleType("tensorflow.keras.preprocessing.image")

    class FakeIDG:
        def __init__(self, *a, **k):
            pass

    image_mod.ImageDataGenerator = FakeIDG
    preprocessing.image = image_mod
    models_mod.load_model = lambda path: DummyModel()
    keras.models = models_mod
    keras.preprocessing = preprocessing
    tf.keras = keras

    st_mod = types.ModuleType("streamlit")
    st_mod.set_page_config = lambda **k: None
    st_mod.cache_resource = lambda show_spinner=True: (lambda f: f)
    st_mod.title = lambda *a, **k: None
    st_mod.caption = lambda *a, **k: None
    st_mod.error = lambda *a, **k: None
    st_mod.info = lambda *a, **k: None
    st_mod.success = lambda *a, **k: None
    st_mod.metric = lambda *a, **k: None
    st_mod.write = lambda *a, **k: None
    st_mod.progress = lambda *a, **k: None
    st_mod.image = lambda *a, **k: None

    for name, mod in [
        ("tensorflow", tf), ("tensorflow.keras", keras),
        ("tensorflow.keras.models", models_mod),
        ("tensorflow.keras.preprocessing", preprocessing),
        ("tensorflow.keras.preprocessing.image", image_mod),
        ("streamlit", st_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    import importlib
    import app as app_module
    importlib.reload(app_module)
    return app_module


def test_classify_returns_valid_structure(app_module):
    model = DummyModel()
    class_names = ["plastic", "paper", "metal"]
    img = Image.new("RGB", (80, 80), color=(50, 120, 200))

    result = app_module.classify(model, class_names, img)

    assert result["predicted_class"] == "plastic"  # DummyModel favours index 0
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["all_probabilities"].keys()) == set(class_names)
    assert abs(sum(result["all_probabilities"].values()) - 1.0) < 0.01


def test_classify_handles_grayscale_image(app_module):
    model = DummyModel()
    class_names = ["plastic", "paper", "metal"]
    img = Image.new("L", (60, 60), color=100)

    result = app_module.classify(model, class_names, img)
    assert result["predicted_class"] in class_names


def test_load_trained_model_raises_clear_error_when_missing(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "MODEL_CANDIDATES", [
        str(tmp_path / "mobilenet_waste.keras"),
        str(tmp_path / "baseline_cnn_waste.keras"),
    ])
    monkeypatch.setattr(app_module, "CLASS_NAMES_PATH", str(tmp_path / "class_names.json"))

    with pytest.raises(FileNotFoundError, match="No trained model found"):
        app_module.load_trained_model()


def test_load_trained_model_succeeds_when_model_file_present(app_module, tmp_path, monkeypatch):
    model_path = tmp_path / "mobilenet_waste.keras"
    model_path.write_bytes(b"")  # placeholder file; load_model is stubbed
    class_names_path = tmp_path / "class_names.json"
    class_names_path.write_text(json.dumps(["plastic", "paper", "metal"]))

    monkeypatch.setattr(app_module, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "MODEL_CANDIDATES", [str(model_path)])
    monkeypatch.setattr(app_module, "CLASS_NAMES_PATH", str(class_names_path))

    model, class_names, path = app_module.load_trained_model()
    assert class_names == ["plastic", "paper", "metal"]
    assert path == str(model_path)
