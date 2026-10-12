"""
Integration tests for app/app.py — Chapter 3.10, "Integration Testing" row.

A tiny dummy Keras model is substituted for the real trained model (via
monkeypatch) so these tests run fast and don't require a trained .keras
file to exist.

Run with: pytest tests/
"""

import os
import sys
import io
import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
import app as app_module  # noqa: E402


class DummyModel:
    """Stands in for a trained Keras model: always predicts class index 0
    with high confidence, regardless of input."""

    output_shape = (None, 3)

    def predict(self, x, verbose=0):
        batch_size = x.shape[0]
        probs = np.zeros((batch_size, 3), dtype=np.float32)
        probs[:, 0] = 0.9
        probs[:, 1] = 0.07
        probs[:, 2] = 0.03
        return probs


@pytest.fixture
def client(monkeypatch):
    dummy_classes = ["plastic", "paper", "metal"]

    def fake_get_model():
        return DummyModel(), dummy_classes

    monkeypatch.setattr(app_module, "get_model", fake_get_model)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def make_image_bytes(fmt="JPEG"):
    img = Image.new("RGB", (50, 50), color=(10, 200, 10))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Intelligent Waste Classification" in resp.data


def test_health_endpoint_ok_with_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_missing_file_returns_400(client):
    resp = client.post("/predict", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_rejects_bad_extension(client):
    data = {"file": (io.BytesIO(b"not an image"), "malware.exe")}
    resp = client.post("/predict", content_type="multipart/form-data", data=data)
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]


def test_predict_rejects_corrupt_image(client):
    data = {"file": (io.BytesIO(b"this is not really a jpeg"), "fake.jpg")}
    resp = client.post("/predict", content_type="multipart/form-data", data=data)
    assert resp.status_code == 400
    assert "Could not read image" in resp.get_json()["error"]


def test_predict_returns_valid_prediction_for_real_image(client):
    img_bytes = make_image_bytes()
    data = {"file": (img_bytes, "sample.jpg")}
    resp = client.post("/predict", content_type="multipart/form-data", data=data)

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["predicted_class"] == "plastic"  # DummyModel always favours index 0
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["all_probabilities"].keys()) == {"plastic", "paper", "metal"}
    assert abs(sum(body["all_probabilities"].values()) - 1.0) < 0.01
