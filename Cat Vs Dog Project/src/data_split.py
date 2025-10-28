#!/usr/bin/env python3
"""
split_dataset.py

Split a dataset organized like:
    raw/
      cat/
      dog/

into:
    data/
      train/
        cat/
        dog/
      val/        (optional)
        cat/
        dog/
      test/
        cat/
        dog/

Usage:
    python scripts/split_dataset.py --src raw --dst data --train 0.8 --val 0.1 --test 0.1 --seed 42 --mode copy

Options:
    --mode copy|move (default copy)
"""

import argparse
from pathlib import Path
import random
import shutil
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

def get_image_files(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS and p.is_file()]

def make_split_for_class(src_class_dir: Path, dst_root: Path, train_ratio: float, val_ratio: float, test_ratio: float,
                         seed: int, mode: str):
    files = get_image_files(src_class_dir)
    if not files:
        print(f"[WARN] no files found in {src_class_dir}")
        return

    random.Random(seed).shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # rest -> test
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    splits = [("train", train_files), ("val", val_files), ("test", test_files)]

    for split_name, file_list in splits:
        dst_dir = dst_root / split_name / src_class_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in file_list:
            dst_path = dst_dir / f.name
            if mode == "copy":
                shutil.copy2(f, dst_path)
            else:
                shutil.move(f, dst_path)
        print(f"Copied {len(file_list)} files to {dst_dir}")

def create_splits(src: Path, dst: Path, train_ratio: float, val_ratio: float, test_ratio: float,
                  seed: int = 42, mode: str = "copy"):
    # validate ratios sum approx 1
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    if not src.exists():
        raise FileNotFoundError(f"Source folder {src} not found")

    classes = [p for p in src.iterdir() if p.is_dir()]
    if not classes:
        raise ValueError(f"No class subfolders found in {src}. Expected folders like 'cat', 'dog'.")

    dst = dst.resolve()
    for class_dir in classes:
        make_split_for_class(class_dir, dst, train_ratio, val_ratio, test_ratio, seed, mode)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source folder with class subfolders (e.g., raw/)", type=Path)
    p.add_argument("--dst", required=True, help="Destination folder to create splits (e.g., data/)", type=Path)
    p.add_argument("--train", type=float, default=0.8, help="Train fraction (default 0.8)")
    p.add_argument("--val", type=float, default=0.1, help="Val fraction (default 0.1)")
    p.add_argument("--test", type=float, default=0.1, help="Test fraction (default 0.1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["copy", "move"], default="copy", help="Copy or move files")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_splits(args.src, args.dst, args.train, args.val, args.test, seed=args.seed, mode=args.mode)
    print("Done.")
