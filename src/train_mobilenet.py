"""
Train the MobileNetV2 transfer-learning model — Chapter 3.6.2.

Two-phase training:
  Phase 1: base frozen, train only the new classification head.
  Phase 2: unfreeze the top of the base and fine-tune at a low learning rate.

Usage:
    python src/train_mobilenet.py
"""

import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2

from data_utils import (
    build_generators, get_class_names, validate_class_consistency,
    IMG_SIZE, MODELS_DIR,
)

FINE_TUNE_AT = 100  # unfreeze layers from this index onward in phase 2


def build_mobilenet_model(num_classes, input_shape=(224, 224, 3)):
    # An optional local file avoids an unnecessary network request on a
    # reproducible/offline run.  The default remains Keras' ImageNet weights.
    weights = os.environ.get("MOBILENETV2_WEIGHTS", "imagenet")
    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights=weights
    )
    base_model.trainable = False  # frozen for phase 1

    inputs = layers.Input(shape=input_shape)
    # data_utils supplies float pixels in [0, 1].  ImageNet MobileNetV2
    # weights expect values in [-1, 1], so keep this conversion inside the
    # saved model.  That also makes web-app inference consistent with training.
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_input_scaling")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history_phase1, history_phase2, out_path):
    acc = history_phase1.history["accuracy"] + history_phase2.history["accuracy"]
    val_acc = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
    loss = history_phase1.history["loss"] + history_phase2.history["loss"]
    val_loss = history_phase1.history["val_loss"] + history_phase2.history["val_loss"]
    switch_epoch = len(history_phase1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].axvline(switch_epoch, color="gray", linestyle="--", label="fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].axvline(switch_epoch, color="gray", linestyle="--", label="fine-tune start")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    validate_class_consistency()
    class_names = get_class_names()
    print(f"Classes ({len(class_names)}): {class_names}")

    train_gen, val_gen = build_generators()

    model, base_model = build_mobilenet_model(
        num_classes=len(class_names), input_shape=(*IMG_SIZE, 3)
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, "mobilenet_waste.keras")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ]

    print("\n--- Phase 1: training classification head (base frozen) ---")
    history1 = model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=cb)

    print("\n--- Phase 2: fine-tuning top layers of MobileNetV2 ---")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=cb)

    plot_history(history1, history2, os.path.join(MODELS_DIR, "mobilenet_training_curves.png"))

    with open(os.path.join(MODELS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
