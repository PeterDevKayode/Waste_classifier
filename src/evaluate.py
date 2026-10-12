"""
Evaluate a trained model on the held-out test set — Chapter 3.9 / Chapter 4.

Computes accuracy, precision, recall, F1-score (macro-averaged, since
classes may be imbalanced), a per-class classification report, a confusion
matrix image, and average per-image inference time.

Usage:
    python src/evaluate.py --model models/mobilenet_waste.keras
    python src/evaluate.py --model models/baseline_cnn_waste.keras
"""

import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from tensorflow.keras.models import load_model

from data_utils import build_test_generator, get_class_names, MODELS_DIR


def evaluate(model_path, test_dir=None):
    class_names = get_class_names()
    test_gen = build_test_generator(test_dir) if test_dir else build_test_generator()

    print(f"Loading model from {model_path} ...")
    model = load_model(model_path)

    y_true = test_gen.classes
    n_images = len(y_true)

    start = time.perf_counter()
    y_prob = model.predict(test_gen, verbose=1)
    elapsed = time.perf_counter() - start
    avg_inference_ms = (elapsed / n_images) * 1000

    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n=== Test Set Results ===")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision (macro):   {precision:.4f}")
    print(f"Recall (macro):      {recall:.4f}")
    print(f"F1-score (macro):    {f1:.4f}")
    print(f"Avg inference time:  {avg_inference_ms:.2f} ms/image")

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    print("\n--- Per-class report ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {os.path.basename(model_path)}")
    fig.tight_layout()

    out_name = os.path.splitext(os.path.basename(model_path))[0]
    cm_path = os.path.join(MODELS_DIR, f"{out_name}_confusion_matrix.png")
    fig.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

    results = {
        "model": model_path,
        "test_images": int(n_images),
        "class_names": class_names,
        "parameter_count": int(model.count_params()),
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "avg_inference_ms": avg_inference_ms,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
    }
    results_path = os.path.join(MODELS_DIR, f"{out_name}_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics summary saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a trained .keras model")
    parser.add_argument("--test_dir", default=None, help="Optional override for the test data directory")
    args = parser.parse_args()
    evaluate(args.model, args.test_dir)
