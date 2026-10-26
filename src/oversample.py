"""Oversample training images that contain rare classes (repeat factor sampling).

Each class c gets a repeat factor r_c = max(1, sqrt(t / f_c)), where f_c is
the fraction of training images containing c and t is a threshold. Each image
is repeated by the largest factor among the classes it contains, so images
with rare defects are seen more often while common ones are left alone. The
square root keeps the boost moderate, which limits memorisation of the few
rare examples. Method from Gupta et al. (2019), "LVIS: A Dataset for Large
Vocabulary Instance Segmentation".

Only the training split is changed; valid and test are copied unchanged so
results stay comparable with the baseline.

Run from the repo root:

    python src/oversample.py --data data/blocked --out data/blocked_oversampled
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml

from audit_dataset import IMAGE_EXTS, load_class_names, parse_label_file


def class_fractions(image_classes: list[set[int]], n_classes: int) -> np.ndarray:
    """Fraction of images that contain each class at least once.

    Args:
        image_classes: The set of class ids present in each image.
        n_classes: Number of classes.

    Returns:
        Array of length ``n_classes`` with values between 0 and 1.
    """
    counts = np.zeros(n_classes)
    for classes in image_classes:
        for c in classes:
            counts[c] += 1
    return counts / max(len(image_classes), 1)


def class_repeat_factors(fractions: np.ndarray, threshold: float) -> np.ndarray:
    """Repeat factor per class, r_c = max(1, sqrt(threshold / f_c)).

    Classes that appear in at least ``threshold`` of images get 1 (no boost).
    Classes absent from training also get 1, since there is nothing to repeat.
    """
    fractions = np.asarray(fractions, dtype=float)
    with np.errstate(divide="ignore"):
        r = np.sqrt(threshold / fractions)
    return np.where(fractions > 0, np.maximum(1.0, r), 1.0)


def image_repeat_factors(image_classes: list[set[int]], class_factors: np.ndarray) -> np.ndarray:
    """Repeat factor per image: the largest factor of any class it contains.

    Background images (no boxes) get 1.
    """
    return np.array([max((class_factors[c] for c in classes), default=1.0)
                     for classes in image_classes])


def copies_from_factors(factors: np.ndarray, seed: int = 0) -> np.ndarray:
    """Turn fractional repeat factors into whole numbers of copies.

    Uses stochastic rounding: a factor of 2.3 gives 2 copies, plus a third
    with probability 0.3. On average this matches the factor exactly, and a
    fixed seed makes the result reproducible.
    """
    factors = np.asarray(factors, dtype=float)
    rng = np.random.default_rng(seed)
    whole = np.floor(factors)
    return (whole + (rng.random(len(factors)) < factors - whole)).astype(int)


def oversample(
    data_dir: Path,
    out_dir: Path,
    threshold: float = 0.1,
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    """Build an oversampled copy of a dataset.

    Args:
        data_dir: Source dataset root (e.g. the blocked split).
        out_dir: Destination for the oversampled dataset.
        threshold: Classes in fewer than this fraction of training images
            get boosted.
        seed: Random seed for the stochastic rounding.
        overwrite: Replace ``out_dir`` if it exists.

    Returns:
        Summary used for the console report.

    Raises:
        ValueError: If ``out_dir`` is ``data_dir``.
        FileExistsError: If ``out_dir`` exists and ``overwrite`` is False.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if out_dir.resolve() == data_dir.resolve():
        raise ValueError("out_dir must differ from data_dir")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists, use --overwrite to replace it")
        shutil.rmtree(out_dir)

    names = load_class_names(data_dir)
    n = len(names)

    # Valid and test are copied unchanged.
    for split in ("valid", "test"):
        shutil.copytree(data_dir / split, out_dir / split)

    img_dir, lbl_dir = data_dir / "train" / "images", data_dir / "train" / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    boxes_per_image, image_classes = [], []
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        boxes = parse_label_file(lbl, n)[0] if lbl.exists() else []
        boxes_per_image.append(np.bincount([b[0] for b in boxes], minlength=n))
        image_classes.append({b[0] for b in boxes})

    fractions = class_fractions(image_classes, n)
    class_factors = class_repeat_factors(fractions, threshold)
    copies = copies_from_factors(image_repeat_factors(image_classes, class_factors), seed)

    out_img, out_lbl = out_dir / "train" / "images", out_dir / "train" / "labels"
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)
    for img, k in zip(images, copies):
        lbl = lbl_dir / f"{img.stem}.txt"
        for rep in range(k):
            # Copies keep the original name plus _rep<n>, so frame numbers and
            # source names still parse and audits still work.
            stem = img.stem if rep == 0 else f"{img.stem}_rep{rep}"
            shutil.copy2(img, out_img / f"{stem}{img.suffix}")
            if lbl.exists():
                shutil.copy2(lbl, out_lbl / f"{stem}.txt")

    cfg = {"train": "train/images", "val": "valid/images", "test": "test/images",
           "nc": n, "names": names}
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    box_matrix = np.array(boxes_per_image) if boxes_per_image else np.zeros((0, n))
    return {
        "names": names,
        "threshold": threshold,
        "fractions": fractions,
        "class_factors": class_factors,
        "images_before": len(images),
        "images_after": int(copies.sum()),
        "boxes_before": box_matrix.sum(axis=0).astype(int),
        "boxes_after": (box_matrix * copies[:, None]).sum(axis=0).astype(int),
    }


def print_summary(s: dict) -> None:
    """Print how much each class was boosted."""
    print(f"\nRepeat factor sampling, threshold t = {s['threshold']}")
    print(f"Training images: {s['images_before']:,} -> {s['images_after']:,}\n")
    print(f"{'class':<16}{'in images':>10}{'factor':>8}{'boxes before':>14}{'boxes after':>13}")
    for i, name in enumerate(s["names"]):
        print(f"{name:<16}{100 * s['fractions'][i]:>9.1f}%{s['class_factors'][i]:>8.2f}"
              f"{s['boxes_before'][i]:>14,}{s['boxes_after'][i]:>13,}")


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/blocked", help="source dataset root")
    parser.add_argument("--out", default="data/blocked_oversampled", help="output folder")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="boost classes found in fewer than this fraction of images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    summary = oversample(args.data, args.out, args.threshold, args.seed, args.overwrite)
    print_summary(summary)
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
