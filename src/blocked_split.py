"""Create a leakage resistant train/valid/test split from blocks of video frames.

Roboflow split the frames of one video at random, so neighbouring (near
identical) frames ended up in different splits. This script regroups the same
images so that whole stretches of video go to a single split, then removes
frames near the boundaries so no valid or test frame has a close neighbour in
train.

The original ``data`` folder is left untouched, so both splits can be trained
and compared.

Run from the repo root:

    python src/blocked_split.py --data data --out data/blocked
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from audit_dataset import (
    IMAGE_EXTS,
    SPLITS,
    frame_number,
    load_class_names,
    nearest_frame_distance,
    parse_label_file,
)

Copy = tuple[str, Path, "Path | None"]  # (original split, image path, label path)


@dataclass
class Frame:
    """Every exported copy of one video frame.

    Attributes:
        number: Video frame number.
        copies: One entry per exported image of this frame. Train frames
            usually have several (augmented) copies.
        class_counts: Boxes per class id, counted from a single copy.
    """

    number: int
    copies: list[Copy] = field(default_factory=list)
    class_counts: np.ndarray | None = None

    def is_ambiguous(self, aug_copies: int = 3) -> bool:
        """Return True if this frame number probably covers more than one photo.

        Roboflow gave each training photo ``aug_copies`` exported versions, so
        more copies than that, or copies spread over several original splits,
        means the number was reused (e.g. by a second recording).

        Args:
            aug_copies: Exported versions per training photo.
        """
        train_copies = sum(1 for c in self.copies if c[0] == "train")
        return train_copies > aug_copies or len({c[0] for c in self.copies}) > 1

    def clean_copy(self) -> Copy:
        """Return one copy, preferring an unaugmented one from test or valid.

        Roboflow only augmented the training split, so a copy exported to
        test or valid is the closest thing to the untouched original.
        """
        for preferred in ("test", "valid", "train"):
            for copy in self.copies:
                if copy[0] == preferred:
                    return copy
        return self.copies[0]


def collect_frames(data_dir: Path, n_classes: int) -> tuple[dict[int, Frame], list[str]]:
    """Group every image in the dataset by its video frame number.

    Args:
        data_dir: Roboflow dataset root with train, valid and test folders.
        n_classes: Number of classes in the dataset.

    Returns:
        A tuple of (frames keyed by frame number, names of images skipped
        because they contain no frame number).
    """
    frames: dict[int, Frame] = {}
    skipped: list[str] = []
    for split in SPLITS:
        img_dir = Path(data_dir) / split / "images"
        lbl_dir = Path(data_dir) / split / "labels"
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            number = frame_number(img.name)
            if number is None:
                skipped.append(img.name)
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            frames.setdefault(number, Frame(number)).copies.append(
                (split, img, lbl if lbl.exists() else None)
            )

    # Augmentation only changed colour and exposure, so every copy has the
    # same boxes. Count them once to avoid inflating the totals.
    for frame in frames.values():
        counts = np.zeros(n_classes, dtype=int)
        label = frame.clean_copy()[2]
        if label is not None:
            boxes, _ = parse_label_file(label, n_classes)
            for box in boxes:
                counts[box[0]] += 1
        frame.class_counts = counts
    return frames, skipped


def make_blocks(numbers: np.ndarray, block_size: int) -> np.ndarray:
    """Assign each frame to a block of consecutive frame numbers.

    Args:
        numbers: Frame numbers.
        block_size: Width of each block in frames (e.g. 300 means frames
            0 to 299 form block 0, 300 to 599 block 1, ...).

    Returns:
        Block index per frame, renumbered 0, 1, 2... with no gaps.
    """
    _, block_ids = np.unique(np.asarray(numbers) // block_size, return_inverse=True)
    return block_ids


def score_assignment(
    assign: np.ndarray, block_frames: np.ndarray, block_classes: np.ndarray, targets: np.ndarray
) -> float:
    """Score how well a block to split assignment meets the targets (lower is better).

    The score adds three terms:

    1. how far each split's share of frames is from its target fraction;
    2. how far each class's share in each split is from that target;
    3. a penalty for every class missing from a split: 100 if missing from
       train (the model cannot learn it at all), 10 if missing from valid
       or test (it cannot be evaluated). So when a rare class is too
       clustered to share, train always keeps it.

    Args:
        assign: Split index (0 train, 1 valid, 2 test) for each block.
        block_frames: Number of frames in each block.
        block_classes: Box counts per class for each block, shape (blocks, classes).
        targets: Target fraction of frames for each split, summing to 1.

    Returns:
        The combined score.
    """
    n_splits = len(targets)
    frames = np.bincount(assign, weights=block_frames, minlength=n_splits)
    size_err = np.abs(frames / frames.sum() - targets).sum()

    per_split = np.zeros((n_splits, block_classes.shape[1]))
    np.add.at(per_split, assign, block_classes)
    totals = per_split.sum(axis=0)
    present = totals > 0
    shares = per_split[:, present] / totals[present]
    balance_err = np.abs(shares - targets[:, None]).mean(axis=1).sum()

    missing = (per_split[:, present] == 0).sum(axis=1)  # per split: train, valid, test
    penalty = 100 * missing[0] + 10 * missing[1:].sum()
    return float(size_err + balance_err + penalty)


def search_assignment(
    block_frames: np.ndarray,
    block_classes: np.ndarray,
    targets: np.ndarray,
    n_trials: int = 20000,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Find a good block assignment by random search.

    Each trial sends every block to a random split (weighted by the targets)
    and keeps the assignment with the lowest score. A fixed seed makes the
    result reproducible.

    Args:
        block_frames: Number of frames in each block.
        block_classes: Box counts per class for each block.
        targets: Target fraction of frames for each split.
        n_trials: Number of random assignments to try.
        seed: Random seed.

    Returns:
        A tuple of (best assignment, its score).
    """
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(n_trials):
        assign = rng.choice(len(targets), size=len(block_frames), p=targets)
        score = score_assignment(assign, block_frames, block_classes, targets)
        if score < best_score:
            best, best_score = assign, score
    return best, best_score


def purge_boundaries(
    numbers: np.ndarray, split_ids: np.ndarray, gap: int, protected: np.ndarray | None = None
) -> np.ndarray:
    """Mark frames to drop because they sit too close to another split.

    Train frames within ``gap`` frames of any valid or test frame are
    dropped, as are valid frames within ``gap`` of a test frame. Protected
    train frames are never dropped; instead, any valid or test frame within
    ``gap`` of them is dropped.

    Args:
        numbers: Frame numbers.
        split_ids: Split index per frame (0 train, 1 valid, 2 test).
        gap: Minimum allowed frame distance between different splits.
        protected: Optional boolean mask of train frames that must be kept.

    Returns:
        Boolean mask, True for frames to keep.
    """
    numbers = np.asarray(numbers)
    split_ids = np.asarray(split_ids)
    protected = np.zeros(len(numbers), dtype=bool) if protected is None else np.asarray(protected)
    keep = np.ones(len(numbers), dtype=bool)

    # Clear evaluation frames next to protected training frames first.
    evaluation = split_ids > 0
    if protected.any() and evaluation.any():
        dist = nearest_frame_distance(numbers[evaluation], numbers[protected])
        keep[np.flatnonzero(evaluation)[dist <= gap]] = False

    for split in range(len(SPLITS) - 1):
        mine = (split_ids == split) & keep & ~protected
        later = numbers[(split_ids > split) & keep]
        dist = nearest_frame_distance(numbers[mine], later)
        keep[np.flatnonzero(mine)[dist <= gap]] = False
    return keep


def write_dataset(
    out_dir: Path, by_split: dict[str, list[Frame]], names: list[str], blocks: dict[int, int]
) -> dict[str, int]:
    """Copy images and labels into the new split folders.

    Train keeps every augmented copy of its frames. Valid and test get one
    clean copy per frame, so evaluation never counts the same frame twice.

    Args:
        out_dir: Destination folder (created if needed).
        by_split: Frames assigned to each split.
        names: Class names, written to the new data.yaml.
        blocks: Block index per frame number, recorded in the manifest.

    Returns:
        Number of images written per split.
    """
    written = {}
    rows = []
    for split, frames in by_split.items():
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for frame in frames:
            copies = frame.copies if split == "train" else [frame.clean_copy()]
            for _, img, lbl in copies:
                shutil.copy2(img, img_dir / img.name)
                if lbl is not None:
                    shutil.copy2(lbl, lbl_dir / lbl.name)
                count += 1
            rows.append([frame.number, blocks[frame.number], split, len(copies),
                         *frame.class_counts.tolist()])
        written[split] = count

    cfg = {"train": "train/images", "val": "valid/images", "test": "test/images",
           "nc": len(names), "names": names}
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with open(out_dir / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "block", "split", "images_written", *names])
        writer.writerows(sorted(rows))
    return written


def blocked_split(
    data_dir: Path,
    out_dir: Path,
    block_size: int = 300,
    gap: int = 25,
    targets: tuple[float, float, float] = (0.7, 0.15, 0.15),
    n_trials: int = 20000,
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    """Run the full blocked split and write the new dataset.

    Args:
        data_dir: Original Roboflow dataset root.
        out_dir: Where to write the blocked dataset.
        block_size: Width of each block in frames.
        gap: Minimum frame distance allowed between different splits.
        targets: Target fraction of frames for train, valid and test.
        n_trials: Random search trials.
        seed: Random seed for reproducibility.
        overwrite: Replace ``out_dir`` if it already exists.

    Returns:
        Summary dictionary used for the console report.

    Raises:
        ValueError: If targets do not sum to 1 or ``out_dir`` is ``data_dir``.
        FileExistsError: If ``out_dir`` exists and ``overwrite`` is False.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    targets_arr = np.asarray(targets, dtype=float)
    if not np.isclose(targets_arr.sum(), 1.0):
        raise ValueError("targets must sum to 1")
    if out_dir.resolve() == data_dir.resolve():
        raise ValueError("out_dir must differ from data_dir, the original data is kept")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists, use --overwrite to replace it")
        shutil.rmtree(out_dir)

    names = load_class_names(data_dir)
    frames, skipped = collect_frames(data_dir, len(names))
    numbers = np.array(sorted(frames))
    class_matrix = np.array([frames[n].class_counts for n in numbers])

    block_ids = make_blocks(numbers, block_size)
    n_blocks = block_ids.max() + 1
    block_frames = np.bincount(block_ids, minlength=n_blocks)
    block_classes = np.zeros((n_blocks, len(names)))
    np.add.at(block_classes, block_ids, class_matrix)

    assign, score = search_assignment(block_frames, block_classes, targets_arr, n_trials, seed)
    split_ids = assign[block_ids].copy()

    # Frame numbers shared by two photos always go to train, so valid and test
    # only contain frames that are unambiguously one photo.
    ambiguous = np.array([frames[n].is_ambiguous() for n in numbers])
    split_ids[ambiguous] = 0
    keep = purge_boundaries(numbers, split_ids, gap, protected=ambiguous)

    by_split = {s: [frames[n] for n in numbers[(split_ids == i) & keep]]
                for i, s in enumerate(SPLITS)}
    written = write_dataset(out_dir, by_split, names,
                            dict(zip(numbers.tolist(), block_ids.tolist())))

    kept_numbers = {s: np.array([f.number for f in fs]) for s, fs in by_split.items()}
    return {
        "names": names,
        "n_frames": len(numbers),
        "n_blocks": int(n_blocks),
        "score": score,
        "skipped": skipped,
        "ambiguous": int(ambiguous.sum()),
        "purged": {s: int(((split_ids == i) & ~keep).sum()) for i, s in enumerate(SPLITS)},
        "frames": {s: len(fs) for s, fs in by_split.items()},
        "images": written,
        "classes": {s: (np.sum([f.class_counts for f in fs], axis=0) if fs
                        else np.zeros(len(names), dtype=int))
                    for s, fs in by_split.items()},
        "min_gap": {
            "train > valid": _min_gap(kept_numbers["valid"], kept_numbers["train"]),
            "train > test": _min_gap(kept_numbers["test"], kept_numbers["train"]),
            "valid > test": _min_gap(kept_numbers["test"], kept_numbers["valid"]),
        },
    }


def _min_gap(query: np.ndarray, reference: np.ndarray) -> float:
    """Smallest frame distance from any query frame to the reference (inf if either is empty)."""
    if len(query) == 0 or len(reference) == 0:
        return float("inf")
    return float(nearest_frame_distance(query, reference).min())


def print_summary(summary: dict, gap: int) -> None:
    """Print the outcome of the blocked split."""
    print(f"\n{summary['n_frames']} unique frames in {summary['n_blocks']} blocks, "
          f"search score {summary['score']:.3f}")
    print(f"{summary['ambiguous']} frame numbers shared by two photos, kept in train")
    if summary["skipped"]:
        print(f"Skipped {len(summary['skipped'])} images with no frame number")

    total = sum(summary["frames"].values())
    print("\n== New split ==")
    print(f"{'split':<6} {'frames':>7} {'share':>7} {'images':>7} {'purged':>7}")
    for s in SPLITS:
        print(f"{s:<6} {summary['frames'][s]:>7} {100 * summary['frames'][s] / total:>6.1f}% "
              f"{summary['images'][s]:>7} {summary['purged'][s]:>7}")

    print("\n== Boxes per class (one copy per frame) ==")
    print(f"{'class':<16}" + "".join(f"{s:>8}" for s in SPLITS))
    missing = []
    for cid, name in enumerate(summary["names"]):
        counts = [int(summary["classes"][s][cid]) for s in SPLITS]
        print(f"{name:<16}" + "".join(f"{c:>8}" for c in counts))
        missing += [f"{name} in {s}" for s, c in zip(SPLITS, counts) if c == 0]

    print(f"\n== Closest frames between splits (gap setting {gap}) ==")
    for pair, d in summary["min_gap"].items():
        print(f"{pair:<14} {d:>6.0f}  {'OK' if d > gap else 'TOO CLOSE'}")

    if missing:
        print("\nWARNING, classes with no boxes: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data", help="original dataset root")
    parser.add_argument("--out", default="data/blocked", help="output folder")
    parser.add_argument("--block-size", type=int, default=300, help="frames per block")
    parser.add_argument("--gap", type=int, default=25, help="min frames between splits")
    parser.add_argument("--targets", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        metavar=("TRAIN", "VALID", "TEST"), help="target frame fractions")
    parser.add_argument("--trials", type=int, default=20000, help="random search trials")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output")
    args = parser.parse_args(argv)

    summary = blocked_split(args.data, args.out, args.block_size, args.gap,
                            tuple(args.targets), args.trials, args.seed, args.overwrite)
    print_summary(summary, args.gap)
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
