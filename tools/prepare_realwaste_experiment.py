"""Create a reproducible five-class experimental subset from RealWaste.

The selected labels preserve material meaning:
* cardboard + paper -> paper
* food organics + vegetation -> organic
* glass, metal and plastic remain unchanged

No source class is relabelled as an unrelated material.  A deterministic,
balanced sample is used so the CPU experiment is practical and each class has
the same support in the held-out test split.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / ".datasets" / "realwaste.zip"
DATA_DIR = ROOT / "data"
MANIFEST = ROOT / "models" / "realwaste_experiment_manifest.csv"
METADATA = ROOT / "models" / "realwaste_experiment_metadata.json"

SOURCE_TO_TARGET = {
    "Cardboard": "paper",
    "Paper": "paper",
    "Food Organics": "organic",
    "Vegetation": "organic",
    "Glass": "glass",
    "Metal": "metal",
    "Plastic": "plastic",
}
TARGETS = ("glass", "metal", "organic", "paper", "plastic")


def source_label(member: str) -> str | None:
    parts = Path(member).parts
    for part in parts:
        if part in SOURCE_TO_TARGET:
            return part
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not ZIP_PATH.exists():
        raise SystemExit(f"Missing dataset archive: {ZIP_PATH}")
    if args.per_class < 30:
        raise SystemExit("--per-class must be at least 30 for meaningful 80:10:10 splits")

    rng = random.Random(args.seed)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        available: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for member in zf.namelist():
            label = source_label(member)
            if label and member.lower().endswith((".jpg", ".jpeg", ".png")):
                available[SOURCE_TO_TARGET[label]].append((member, label))
        missing = [target for target in TARGETS if len(available[target]) < args.per_class]
        if missing:
            raise SystemExit(f"Not enough images for {missing}; found " +
                             ", ".join(f"{k}={len(v)}" for k, v in available.items()))

        selected = {target: rng.sample(available[target], args.per_class) for target in TARGETS}
        split_counts = {
            "train": int(args.per_class * 0.8),
            "val": int(args.per_class * 0.1),
        }
        split_counts["test"] = args.per_class - split_counts["train"] - split_counts["val"]

        for split in split_counts:
            for target in TARGETS:
                out = DATA_DIR / split / target
                out.mkdir(parents=True, exist_ok=True)
                if args.force:
                    for old in out.glob("realwaste_*"):
                        old.unlink()

        rows = []
        for target in TARGETS:
            rows_for_target = list(selected[target])
            rng.shuffle(rows_for_target)
            cursor = 0
            for split, count in split_counts.items():
                for i, (member, source) in enumerate(rows_for_target[cursor:cursor + count]):
                    suffix = Path(member).suffix.lower()
                    name = f"realwaste_{target}_{split}_{i:03d}{suffix}"
                    out = DATA_DIR / split / target / name
                    with zf.open(member) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    rows.append({
                        "file": str(out.relative_to(ROOT)).replace("\\", "/"),
                        "split": split,
                        "target_class": target,
                        "source_class": source,
                        "source_member": member,
                    })
                cursor += count

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": "RealWaste",
        "dataset_doi": "10.24432/C5SS4G",
        "dataset_license": "CC BY 4.0",
        "source": "UCI Machine Learning Repository",
        "selection": "Balanced deterministic sample from five target classes",
        "class_mapping": {
            "glass": ["Glass"], "metal": ["Metal"], "plastic": ["Plastic"],
            "paper": ["Paper", "Cardboard"],
            "organic": ["Food Organics", "Vegetation"],
        },
        "per_class": args.per_class,
        "split_counts_per_class": split_counts,
        "seed": args.seed,
        "total_images": len(rows),
        "limitation": "Images are from a landfill facility in Wollongong, Australia; results do not establish performance on Nigerian waste imagery.",
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
