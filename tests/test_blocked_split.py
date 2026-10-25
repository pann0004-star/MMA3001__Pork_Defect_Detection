"""Tests for blocked_split.py, using a small synthetic video dataset."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import audit_dataset as ad
import blocked_split as bs


def add_image(root: Path, split: str, frame: int, tag: str, label: str) -> None:
    """Write a tiny image and its label for one copy of a frame."""
    name = f"frame_with_red_{frame}_jpg.rf.{tag}"
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / split / "images" / f"{name}.jpg")
    (root / split / "labels" / f"{name}.txt").write_text(label)


@pytest.fixture
def video_dataset(tmp_path: Path) -> Path:
    """120 frames spaced 10 apart, randomly scattered like the Roboflow export.

    Class 0 appears everywhere, class 1 in every fourth frame, and train
    frames have two augmented copies each.
    """
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['a', 'b']\n")
    rng = np.random.default_rng(1)
    for frame in range(0, 1200, 10):
        label = "0 0.5 0.5 0.2 0.2\n" + ("1 0.3 0.3 0.1 0.1\n" if frame % 40 == 0 else "")
        split = rng.choice(ad.SPLITS, p=[0.8, 0.1, 0.1])
        copies = 2 if split == "train" else 1
        for c in range(copies):
            add_image(tmp_path, split, frame, f"x{c}", label)
    return tmp_path


def test_clean_copy_prefers_unaugmented():
    frame = bs.Frame(5, copies=[("train", Path("t.jpg"), None), ("valid", Path("v.jpg"), None)])
    assert frame.clean_copy()[0] == "valid"


def test_collect_frames_counts_boxes_once(video_dataset):
    frames, skipped = bs.collect_frames(video_dataset, n_classes=2)
    assert len(frames) == 120 and skipped == []
    assert frames[0].class_counts.tolist() == [1, 1]  # not doubled by two copies
    assert frames[10].class_counts.tolist() == [1, 0]


def test_make_blocks():
    assert bs.make_blocks(np.array([5, 250, 310, 900]), 300).tolist() == [0, 0, 1, 2]


def test_score_penalises_missing_class():
    frames = np.array([10, 10, 10])
    classes = np.array([[5, 1], [5, 1], [5, 1]])  # class b in every block
    targets = np.array([1 / 3, 1 / 3, 1 / 3])
    covered = bs.score_assignment(np.array([0, 1, 2]), frames, classes, targets)
    uncovered = bs.score_assignment(np.array([0, 0, 1]), frames, classes, targets)  # test empty
    assert uncovered > covered + 19  # both classes missing from test cost 10 each


def test_purge_boundaries():
    numbers = np.array([0, 10, 20, 100, 200])
    splits = np.array([0, 0, 2, 0, 1])
    keep = bs.purge_boundaries(numbers, splits, gap=10)
    assert keep.tolist() == [True, False, True, True, True]


def test_blocked_split_end_to_end(video_dataset, tmp_path):
    out = tmp_path / "blocked"
    summary = bs.blocked_split(video_dataset, out, block_size=100, gap=15,
                               targets=(0.6, 0.2, 0.2), n_trials=2000, seed=0)
    # every split keeps both classes and no cross split frames are too close
    for s in ad.SPLITS:
        assert (summary["classes"][s] > 0).all()
    assert all(d > 15 for d in summary["min_gap"].values())
    # valid and test hold exactly one image per frame, train keeps all copies
    for s in ("valid", "test"):
        assert summary["images"][s] == summary["frames"][s]
    assert (out / "data.yaml").exists() and (out / "split_manifest.csv").exists()
    assert ad.load_class_names(out) == ["a", "b"]
    # the new folder passes the audit with no label problems
    stats = ad.audit_split(out, "test", 2)
    assert stats.errors == [] and stats.missing_labels == []


def test_blocked_split_is_reproducible(video_dataset, tmp_path):
    a = bs.blocked_split(video_dataset, tmp_path / "a", block_size=100, n_trials=500, seed=3)
    b = bs.blocked_split(video_dataset, tmp_path / "b", block_size=100, n_trials=500, seed=3)
    assert a["frames"] == b["frames"]


def test_blocked_split_refuses_to_overwrite(video_dataset, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        bs.blocked_split(video_dataset, tmp_path / "out", n_trials=10)
    with pytest.raises(ValueError):
        bs.blocked_split(video_dataset, video_dataset, n_trials=10)


def test_score_keeps_rare_class_in_train():
    frames = np.array([10, 10, 10])
    classes = np.array([[5, 0], [5, 1], [5, 0]])  # class b in one block only
    targets = np.array([1 / 3, 1 / 3, 1 / 3])
    in_train = bs.score_assignment(np.array([1, 0, 2]), frames, classes, targets)
    in_test = bs.score_assignment(np.array([0, 2, 1]), frames, classes, targets)
    assert in_train < in_test


def test_is_ambiguous():
    one = bs.Frame(1, copies=[("train", Path(f"{i}.jpg"), None) for i in range(3)])
    two = bs.Frame(2, copies=[("train", Path(f"{i}.jpg"), None) for i in range(6)])
    spread = bs.Frame(3, copies=[("train", Path("a.jpg"), None), ("test", Path("b.jpg"), None)])
    assert not one.is_ambiguous()
    assert two.is_ambiguous() and spread.is_ambiguous()


def test_purge_keeps_protected_and_clears_neighbours():
    numbers = np.array([0, 10, 100, 200])
    splits = np.array([0, 2, 2, 1])
    protected = np.array([True, False, False, False])
    keep = bs.purge_boundaries(numbers, splits, gap=15, protected=protected)
    assert keep.tolist() == [True, False, True, True]


def test_ambiguous_frames_go_to_train(video_dataset, tmp_path):
    # give frame 500 a second photo (three extra train copies)
    for c in range(3):
        add_image(video_dataset, "train", 500, f"y{c}", "0 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "blocked"
    summary = bs.blocked_split(video_dataset, out, block_size=100, gap=15,
                               targets=(0.6, 0.2, 0.2), n_trials=500, seed=0)
    assert summary["ambiguous"] >= 1
    train_names = [p.name for p in (out / "train" / "images").iterdir()]
    assert any("frame_with_red_500_" in n for n in train_names)
    assert all(d > 15 for d in summary["min_gap"].values())
