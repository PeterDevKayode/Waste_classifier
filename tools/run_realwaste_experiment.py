"""Train and evaluate the two repository models on the documented RealWaste subset."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_utils import MODELS_DIR, build_generators, get_class_names, validate_class_consistency  # noqa: E402
from evaluate import evaluate  # noqa: E402
from train_baseline_cnn import build_baseline_cnn, plot_history as plot_baseline_history  # noqa: E402
from train_mobilenet import build_mobilenet_model, plot_history as plot_mobilenet_history  # noqa: E402


SEED = 42
# This is a CPU-only, reproducible pilot run.  The report records these
# bounded budgets rather than presenting it as exhaustive hyperparameter
# optimisation.  Early stopping may end a phase sooner.
BASELINE_EPOCHS = 5
MOBILENET_HEAD_EPOCHS = 4
MOBILENET_FINE_TUNE_EPOCHS = 2
FINE_TUNE_AT = 100


def set_seed():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def callbacks_for(path: Path):
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        callbacks.ModelCheckpoint(str(path), monitor="val_accuracy", save_best_only=True),
    ]


def main():
    weight_path = ROOT / ".datasets" / "mobilenet_v2_imagenet_no_top.h5"
    if not weight_path.exists() or weight_path.stat().st_size < 8_000_000:
        raise SystemExit("ImageNet MobileNetV2 weights are not ready; retry after the download completes.")
    os.environ["MOBILENETV2_WEIGHTS"] = str(weight_path)
    set_seed()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(4)
    except RuntimeError:
        pass

    validate_class_consistency()
    class_names = get_class_names()
    train_gen, val_gen = build_generators()
    MODELS = Path(MODELS_DIR)
    MODELS.mkdir(parents=True, exist_ok=True)

    with (MODELS / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f)

    baseline_path = MODELS / "baseline_cnn_waste.keras"
    baseline = build_baseline_cnn(len(class_names))
    baseline_history = baseline.fit(
        train_gen, validation_data=val_gen, epochs=BASELINE_EPOCHS,
        callbacks=callbacks_for(baseline_path), verbose=2,
    )
    plot_baseline_history(baseline_history, MODELS / "baseline_cnn_training_curves.png")
    baseline_results = evaluate(str(baseline_path))

    # Recreate generators so augmentation iterator state is identical at the
    # beginning of the second model's run.
    train_gen, val_gen = build_generators()
    mobilenet_path = MODELS / "mobilenet_waste.keras"
    mobilenet, base = build_mobilenet_model(len(class_names))
    mobilenet.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    head_history = mobilenet.fit(
        train_gen, validation_data=val_gen, epochs=MOBILENET_HEAD_EPOCHS,
        callbacks=callbacks_for(mobilenet_path), verbose=2,
    )
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    mobilenet.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    fine_history = mobilenet.fit(
        train_gen, validation_data=val_gen, epochs=MOBILENET_FINE_TUNE_EPOCHS,
        callbacks=callbacks_for(mobilenet_path), verbose=2,
    )
    plot_mobilenet_history(head_history, fine_history, MODELS / "mobilenet_training_curves.png")
    mobilenet_results = evaluate(str(mobilenet_path))

    metadata = {
        "seed": SEED,
        "classes": class_names,
        "input_size": [224, 224],
        "batch_size": 32,
        "baseline_max_epochs": BASELINE_EPOCHS,
        "mobilenet_head_max_epochs": MOBILENET_HEAD_EPOCHS,
        "mobilenet_fine_tune_max_epochs": MOBILENET_FINE_TUNE_EPOCHS,
        "mobilenet_pretraining": "ImageNet MobileNetV2 no-top weights",
        "baseline_results": baseline_results,
        "mobilenet_results": mobilenet_results,
        "tensorflow_version": tf.__version__,
    }
    (MODELS / "experiment_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
