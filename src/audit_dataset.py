"""Audit a Roboflow YOLOv8 object detection dataset.

Checks split sizes, class balance, boxes per image, label validity and
leakage between splits (shared source images, exact duplicates and near
duplicates).

Run from the repo root:

    python src/audit_dataset.py --data data
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

Box = tuple[int, float, float, float, float]
FRAME_WINDOWS = (0, 1, 2, 5, 10, 25, 50)


@dataclass
class SplitStats:
    """Summary of one dataset split.

    Attributes:
        name: Split name, e.g. "train".
        image_paths: Every image file in the split, sorted by name.
        boxes_per_image: Box count for each image (0 if unlabelled).
        class_counts: Number of boxes per class id.
        missing_labels: Image stems with no matching label file.
        orphan_labels: Label stems with no matching image file.
        errors: Human readable descriptions of invalid label lines.
    """

    name: str
    image_paths: list[Path] = field(default_factory=list)
    boxes_per_image: list[int] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        """Number of images in the split."""
        return len(self.image_paths)


def load_class_names(data_dir: Path) -> list[str]:
    """Read class names from ``data.yaml``.

    Args:
        data_dir: Folder containing ``data.yaml``.

    Returns:
        Class names ordered by class id.
    """
    with open(Path(data_dir) / "data.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):  # newer YOLO format: {0: "name", ...}
        names = [names[k] for k in sorted(names)]
    return list(names)


def source_stem(filename: str) -> str:
    """Return the original image name a Roboflow file was derived from.

    Roboflow exports files as ``<original>_jpg.rf.<hash>.jpg``. Augmented
    copies of the same original share everything before ``.rf.``, so this
    is the key for detecting leakage between splits.

    Args:
        filename: File name (not a full path).

    Returns:
        The source identifier.
    """
    if ".rf." in filename:
        return filename.split(".rf.")[0]
    return Path(filename).stem


def parse_label_file(path: Path, n_classes: int) -> tuple[list[Box], list[str]]:
    """Parse a YOLO label file and validate each line.

    Each valid line is ``class x_centre y_centre width height`` with
    coordinates normalised to [0, 1].

    Args:
        path: Path to the ``.txt`` label file.
        n_classes: Number of classes in the dataset.

    Returns:
        A tuple of (valid boxes, error messages for invalid lines).
    """
    boxes: list[Box] = []
    errors: list[str] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        where = f"{Path(path).name}:{lineno}"
        if len(parts) != 5:
            errors.append(f"{where} expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{where} non numeric value")
            continue
        if not 0 <= cls < n_classes:
            errors.append(f"{where} class id {cls} out of range")
            continue
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w == 0 or h == 0:
            errors.append(f"{where} coordinates outside [0, 1] or zero size")
            continue
        boxes.append((cls, x, y, w, h))
    return boxes, errors


def audit_split(data_dir: Path, split: str, n_classes: int) -> SplitStats:
    """Collect statistics for one split.

    Images with no label file are counted as having 0 boxes, matching how
    YOLO treats them (background images).

    Args:
        data_dir: Dataset root containing the split folders.
        split: Split folder name.
        n_classes: Number of classes in the dataset.

    Returns:
        Statistics for the split.

    Raises:
        FileNotFoundError: If the split's images folder does not exist.
    """
    img_dir = Path(data_dir) / split / "images"
    lbl_dir = Path(data_dir) / split / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"No images folder at {img_dir}")

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}

    stats = SplitStats(name=split)
    stats.image_paths = [images[s] for s in sorted(images)]
    stats.missing_labels = sorted(set(images) - set(labels))
    stats.orphan_labels = sorted(set(labels) - set(images))

    for stem in sorted(images):
        if stem not in labels:
            stats.boxes_per_image.append(0)
            continue
        boxes, errors = parse_label_file(labels[stem], n_classes)
        stats.errors.extend(errors)
        stats.boxes_per_image.append(len(boxes))
        stats.class_counts.update(b[0] for b in boxes)
    return stats


def find_shared_sources(split_files: dict[str, list[str]]) -> dict[tuple[str, str], set[str]]:
    """Find source images that appear in more than one split.

    Args:
        split_files: Mapping of split name to its image file names.

    Returns:
        Mapping of each split pair to the source stems they share
        (an empty set means no leakage for that pair).
    """
    sources = {s: {source_stem(f) for f in files} for s, files in split_files.items()}
    return {(a, b): sources[a] & sources[b] for a, b in combinations(sources, 2)}


def frame_number(filename: str) -> int | None:
    """Extract the video frame number from a Roboflow file name.

    For example ``frame_with_red_2247_jpg.rf.abc.jpg`` gives 2247.

    Args:
        filename: File name (not a full path).

    Returns:
        The frame number, or None if the name contains no trailing number.
    """
    match = re.search(r"(\d+)(?:_(?:jpe?g|png|bmp))?$", source_stem(filename))
    return int(match.group(1)) if match else None


def nearest_frame_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each query frame, find the distance to the closest reference frame.

    Uses a sorted reference array and binary search (``np.searchsorted``),
    so it runs in O(n log m) rather than comparing every pair.

    Args:
        query: Frame numbers to check, e.g. the test split.
        reference: Frame numbers to compare against, e.g. the train split.

    Returns:
        Array of absolute frame distances, one per query frame
        (infinity if the reference is empty).
    """
    q = np.asarray(query, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float))
    if len(ref) == 0:
        return np.full(len(q), np.inf)
    idx = np.searchsorted(ref, q)
    left = ref[np.clip(idx - 1, 0, len(ref) - 1)]
    right = ref[np.clip(idx, 0, len(ref) - 1)]
    return np.minimum(np.abs(q - left), np.abs(q - right))


def file_md5(path: Path) -> str:
    """Return the MD5 hash of a file's bytes (detects exact copies)."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def find_exact_duplicates(split_paths: dict[str, list[Path]]) -> list[list[tuple[str, str]]]:
    """Find byte identical images that appear in more than one split.

    Args:
        split_paths: Mapping of split name to its image paths.

    Returns:
        Groups of (split, file name) pairs that share identical bytes and
        span at least two splits.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for split, paths in split_paths.items():
        for p in paths:
            groups.setdefault(file_md5(p), []).append((split, p.name))
    return [g for g in groups.values() if len({s for s, _ in g}) > 1]


def average_hash(path: Path, size: int = 8) -> np.ndarray:
    """Compute a perceptual average hash of an image.

    The image is shrunk to ``size`` x ``size`` greyscale and each pixel is
    compared to the mean, giving a boolean fingerprint that barely changes
    under recompression or small brightness shifts.

    Args:
        path: Image file.
        size: Side length of the shrunk image (hash has size**2 bits).

    Returns:
        Flat boolean array of length ``size**2``.
    """
    with Image.open(path) as img:
        small = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32)
    return (pixels > pixels.mean()).flatten()


def find_near_duplicates(
    hashes_a: np.ndarray, hashes_b: np.ndarray, max_distance: int = 5
) -> list[tuple[int, int, int]]:
    """Find image pairs whose hashes differ in at most ``max_distance`` bits.

    Args:
        hashes_a: Array of shape (n_a, bits).
        hashes_b: Array of shape (n_b, bits).
        max_distance: Largest Hamming distance counted as a near duplicate.

    Returns:
        List of (index in a, index in b, Hamming distance).
    """
    if len(hashes_a) == 0 or len(hashes_b) == 0:
        return []
    dist = (hashes_a[:, None, :] != hashes_b[None, :, :]).sum(axis=2)
    rows, cols = np.nonzero(dist <= max_distance)
    return [(int(i), int(j), int(dist[i, j])) for i, j in zip(rows, cols)]


def print_report(
    stats: dict[str, SplitStats], names: list[str], near_threshold: int, skip_hash: bool
) -> None:
    """Print the full audit report to the console."""
    total = sum(s.n_images for s in stats.values())

    print("\n== Split sizes ==")
    for s in stats.values():
        print(f"{s.name:<6} {s.n_images:>6,} images  ({100 * s.n_images / total:5.1f}%)")
    print(f"{'total':<6} {total:>6,}")

    print("\n== Boxes per image ==")
    print(f"{'split':<6} {'mean':>6} {'median':>7} {'min':>4} {'max':>4} {'no boxes':>9}")
    for s in stats.values():
        b = np.array(s.boxes_per_image)
        print(f"{s.name:<6} {b.mean():>6.2f} {np.median(b):>7.1f} "
              f"{b.min():>4} {b.max():>4} {(b == 0).sum():>9}")

    print("\n== Boxes per class ==")
    header = f"{'class':<16}" + "".join(f"{s:>10}" for s in stats) + f"{'total':>10}"
    print(header)
    for cid, cname in enumerate(names):
        counts = [stats[s].class_counts[cid] for s in stats]
        print(f"{cname:<16}" + "".join(f"{c:>10,}" for c in counts) + f"{sum(counts):>10,}")

    print("\n== Label problems ==")
    clean = True
    for s in stats.values():
        for label, items in (("images without labels", s.missing_labels),
                             ("labels without images", s.orphan_labels),
                             ("invalid label lines", s.errors)):
            if items:
                clean = False
                print(f"{s.name}: {len(items)} {label}, e.g. {items[:3]}")
    if clean:
        print("None found")

    print("\n== Leakage: shared source images ==")
    shared = find_shared_sources({k: [p.name for p in v.image_paths] for k, v in stats.items()})
    for (a, b), stems in shared.items():
        print(f"{a} / {b}: {len(stems)} shared" + (f", e.g. {sorted(stems)[:3]}" if stems else ""))

    print("\n== Leakage: neighbouring video frames ==")
    frames = {}
    for k, v in stats.items():
        nums = {frame_number(p.name) for p in v.image_paths}
        frames[k] = np.array(sorted(n for n in nums if n is not None))
        print(f"{k:<6} {len(frames[k]):>5} unique frames, range {frames[k].min()} to {frames[k].max()}")
    print("Share of the second split's frames with a frame from the first split within N frames:")
    print(f"{'pair':<14}" + "".join(f"{'<=' + str(w):>7}" for w in FRAME_WINDOWS) + f"{'median':>8}")
    for a, b in combinations(stats, 2):
        d = nearest_frame_distance(frames[b], frames[a])
        cells = "".join(f"{100 * np.mean(d <= w):>6.0f}%" for w in FRAME_WINDOWS)
        print(f"{a + ' > ' + b:<14}{cells}{np.median(d):>8.0f}")

    if skip_hash:
        return

    print("\n== Leakage: exact duplicate files ==")
    dupes = find_exact_duplicates({k: v.image_paths for k, v in stats.items()})
    print(f"{len(dupes)} groups" + (f", e.g. {dupes[:2]}" if dupes else ""))

    print(f"\n== Leakage: near duplicates (Hamming distance <= {near_threshold}) ==")
    hashes = {k: np.array([average_hash(p) for p in v.image_paths]) for k, v in stats.items()}
    for a, b in combinations(stats, 2):
        pairs = find_near_duplicates(hashes[a], hashes[b], near_threshold)
        example = ""
        if pairs:
            i, j, d = pairs[0]
            example = (f", e.g. {stats[a].image_paths[i].name} ~ "
                       f"{stats[b].image_paths[j].name} (d={d})")
        print(f"{a} / {b}: {len(pairs)} pairs{example}")
    print("Note: near duplicates are candidates only. Inspect a few by eye before "
          "concluding, since similar scenes on the same conveyor can hash alike.")


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data", help="dataset root folder")
    parser.add_argument("--near-threshold", type=int, default=5,
                        help="max Hamming distance for near duplicates (64 bit hash)")
    parser.add_argument("--skip-hash", action="store_true",
                        help="skip the slower image hashing checks")
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    names = load_class_names(data_dir)
    stats = {s: audit_split(data_dir, s, len(names)) for s in SPLITS}
    print_report(stats, names, args.near_threshold, args.skip_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())