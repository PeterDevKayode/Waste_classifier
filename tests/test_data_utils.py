"""
Unit tests for src/data_utils.py — Chapter 3.10, "Unit Testing" row.

Run with: pytest tests/
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import get_class_names, validate_class_consistency, preprocess_image_array  # noqa: E402


@pytest.fixture
def fake_dataset(tmp_path):
    """Build a minimal train/val/test folder structure with a couple of
    tiny dummy images, so tests don't depend on the real dataset."""
    classes = ["plastic", "paper", "metal"]
    for split in ["train", "val", "test"]:
        for cls in classes:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            img = Image.new("RGB", (10, 10), color=(120, 120, 120))
            img.save(d / "sample.jpg")
    return tmp_path, classes


def test_get_class_names_returns_sorted_folders(fake_dataset):
    tmp_path, classes = fake_dataset
    result = get_class_names(str(tmp_path / "train"))
    assert result == sorted(classes)


def test_get_class_names_raises_on_missing_dir():
    with pytest.raises(FileNotFoundError):
        get_class_names("/path/that/does/not/exist")


def test_get_class_names_raises_on_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty_train"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        get_class_names(str(empty_dir))


def test_validate_class_consistency_passes_for_matching_splits(fake_dataset):
    tmp_path, _ = fake_dataset
    assert validate_class_consistency(
        str(tmp_path / "train"), str(tmp_path / "val"), str(tmp_path / "test")
    ) is True


def test_validate_class_consistency_fails_on_missing_class(fake_dataset):
    tmp_path, _ = fake_dataset
    # Remove one class from the val split only
    shutil.rmtree(tmp_path / "val" / "metal")
    with pytest.raises(ValueError):
        validate_class_consistency(
            str(tmp_path / "train"), str(tmp_path / "val"), str(tmp_path / "test")
        )


def test_preprocess_image_array_shape_and_range():
    img = Image.new("RGB", (500, 300), color=(255, 0, 0))
    arr = preprocess_image_array(img, img_size=(224, 224))

    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_preprocess_image_array_handles_grayscale_input():
    # Simulates a grayscale (single-channel) upload, which must not crash
    img = Image.new("L", (100, 100), color=128)
    arr = preprocess_image_array(img, img_size=(224, 224))
    assert arr.shape == (1, 224, 224, 3)
