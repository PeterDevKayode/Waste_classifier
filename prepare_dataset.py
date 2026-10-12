"""
Merge public-dataset images and locally captured photos, then split them
into data/train|val|test/<class>/ with a stratified 80:10:10 split — so
every split contains a mix of public and local images (Chapter 2, Gap 2:
evaluating only on clean public data would hide how the model performs
on real, locally sourced imagery).

USAGE
-----
1. Create a staging folder anywhere, one subfolder per class, and drop
   ALL images for that class into it — public dataset images and your
   own local photos together, it doesn't matter which came from where:

       staging/
       ├── plastic/    (public images + your local photos, all mixed in)
       ├── paper/
       ├── metal/
       ├── glass/
       └── organic/

2. Run:
       python prepare_dataset.py --staging staging/ --output data/

   This will validate every image (skipping anything corrupt/unreadable),
   resize a *copy* to a consistent max dimension, rename files
   sequentially per class to avoid collisions, shuffle, and split each
   class 80:10:10 into data/train|val|test/<class>/.

3. Re-run any time you add more images — pass --clean to wipe and
   rebuild the split from scratch, otherwise new images are appended to
   train only (so you don't have to worry about used quiz/test images
   being disturbed if you're mid-experiment).

Run with -h for all options.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def validate_and_copy(src: Path, dst: Path, max_dim: int) -> bool:
    """Open, verify, downscale-if-needed, and save as JPEG at dst.
    Returns False (and prints a warning) if the file can't be read."""
    try:
        with Image.open(src) as img:
            img.verify()  # cheap corruption check
        with Image.open(src) as img:  # re-open: verify() invalidates the handle
            img = img.convert("RGB")
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, "JPEG", quality=90)
        return True
    except (UnidentifiedImageError, OSError) as e:
        print(f"  SKIP (unreadable): {src.name} — {e}")
        return False


def split_indices(n, train_ratio, val_ratio, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", required=True, help="Folder containing one subfolder per class with all raw images")
    parser.add_argument("--output", default="data", help="Output data/ folder (default: data)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-dim", type=int, default=800, help="Max width/height after resize (default: 800px)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Wipe existing train/val/test folders before rebuilding")
    args = parser.parse_args()

    staging = Path(args.staging)
    output = Path(args.output)

    if not staging.is_dir():
        sys.exit(f"Staging folder not found: {staging}")

    classes = sorted(d.name for d in staging.iterdir() if d.is_dir())
    if not classes:
        sys.exit(f"No class subfolders found inside {staging}")

    print(f"Classes found: {classes}\n")

    if args.clean:
        for split in ["train", "val", "test"]:
            split_dir = output / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
        print("Cleaned existing train/val/test folders.\n")

    summary = []
    for cls in classes:
        src_dir = staging / cls
        images = find_images(src_dir)
        print(f"[{cls}] found {len(images)} raw files")

        valid_paths = []
        for i, img_path in enumerate(images):
            tmp_name = f"{cls}_{i:04d}.jpg"
            tmp_dst = output / "_staging_tmp" / cls / tmp_name
            if validate_and_copy(img_path, tmp_dst, args.max_dim):
                valid_paths.append(tmp_dst)

        n = len(valid_paths)
        splits = split_indices(n, args.train_ratio, args.val_ratio, args.seed)

        counts = {}
        for split_name, idxs in splits.items():
            split_dir = output / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for j, idx in enumerate(idxs):
                final_name = f"{cls}_{split_name}_{j:04d}.jpg"
                shutil.move(str(valid_paths[idx]), str(split_dir / final_name))
            counts[split_name] = len(idxs)

        print(f"  -> valid: {n} | train: {counts['train']}  val: {counts['val']}  test: {counts['test']}\n")
        summary.append((cls, n, counts["train"], counts["val"], counts["test"]))

    tmp_root = output / "_staging_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    print("=" * 60)
    print(f"{'Class':<12}{'Total':>8}{'Train':>8}{'Val':>8}{'Test':>8}")
    for cls, n, tr, va, te in summary:
        print(f"{cls:<12}{n:>8}{tr:>8}{va:>8}{te:>8}")
    print("=" * 60)
    print(f"\nDone. Data is ready in: {output}/train, {output}/val, {output}/test")


if __name__ == "__main__":
    main()
