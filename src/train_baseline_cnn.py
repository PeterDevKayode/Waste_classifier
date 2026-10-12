"""
Train the baseline CNN (built from scratch) — Chapter 3.6.1.

Usage:
    python src/train_baseline_cnn.py
"""

import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks

from data_utils import (
    build_generators, get_class_names, validate_class_consistency,
    IMG_SIZE, MODELS_DIR,
)


def build_baseline_cnn(num_classes, input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    validate_class_consistency()
    class_names = get_class_names()
    print(f"Classes ({len(class_names)}): {class_names}")

    train_gen, val_gen = build_generators()

    model = build_baseline_cnn(num_classes=len(class_names), input_shape=(*IMG_SIZE, 3))
    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, "baseline_cnn_waste.keras")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=cb,
    )

    plot_history(history, os.path.join(MODELS_DIR, "baseline_cnn_training_curves.png"))

    with open(os.path.join(MODELS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
