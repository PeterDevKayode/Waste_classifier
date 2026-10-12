"""
Shared data-loading and preprocessing utilities for the waste classifier.

Used by both training scripts (train_baseline_cnn.py, train_mobilenet.py)
and by evaluate.py / the Flask app, so preprocessing is guaranteed to be
identical at train and inference time (Chapter 3.4 / 3.5).
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Resolve paths relative to the project root, regardless of the current
# working directory the script is launched from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def get_class_names(train_dir=TRAIN_DIR):
    """Return sorted class names from the subfolders under train_dir.

    Sorting keeps class-to-index mapping stable across separate training
    runs and between training and inference.
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}. "
            "Create data/train/<class_name>/ folders first (see README)."
        )
    classes = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
    )
    if not classes:
        raise ValueError(
            f"No class subfolders found in {train_dir}. "
            "Expected e.g. data/train/plastic/, data/train/paper/, ..."
        )
    return classes


def validate_class_consistency(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR):
    """Raise a clear error if train/val/test don't share the same class folders."""
    train_classes = set(get_class_names(train_dir))
    for name, d in [("val", val_dir), ("test", test_dir)]:
        if not os.path.isdir(d):
            continue
        split_classes = set(
            x for x in os.listdir(d)
            if os.path.isdir(os.path.join(d, x)) and not x.startswith(".")
        )
        missing = train_classes - split_classes
        extra = split_classes - train_classes
        if missing:
            raise ValueError(f"Classes missing from {name} split: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected extra classes in {name} split: {sorted(extra)}")
    return True


def build_generators(train_dir=TRAIN_DIR, val_dir=VAL_DIR, img_size=IMG_SIZE,
                      batch_size=BATCH_SIZE, seed=SEED):
    """Build augmented training and un-augmented validation generators.

    Augmentation (rotation, flip, zoom, brightness, shift) is applied only
    to the training generator, per Chapter 3.4 — validation data must stay
    representative of real, unaltered input.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.10,
        height_shift_range=0.10,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen


def build_test_generator(test_dir=TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Un-augmented generator for final evaluation. shuffle=False keeps
    predictions aligned with true labels for the confusion matrix."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def preprocess_image_array(img, img_size=IMG_SIZE):
    """Preprocess a single PIL Image the same way training images are
    preprocessed, for use at inference time in the Flask app.

    Returns a (1, H, W, 3) float32 array in [0, 1], ready for model.predict.
    """
    img = img.convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
