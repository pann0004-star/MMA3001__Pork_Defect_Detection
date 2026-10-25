"""Tests for audit_dataset.py.

The real dataset is git ignored, so every test builds a tiny fake one in
a temporary folder.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import audit_dataset as ad


def save_image(path: Path, pattern: str) -> None:
    """Save a small 32x32 greyscale test image with a known pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((32, 32), dtype=np.uint8)
    if pattern == "left":
        arr[:, :16] = 255
    elif pattern == "top":
        arr[:16, :] = 255
    Image.fromarray(arr).save(path)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Build a fake dataset with one deliberate leak and one bad label."""
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['a', 'b']\n")

    files = {
        "train": [("img1_jpg.rf.aaa.jpg", "left", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"),
                  ("img1_jpg.rf.bbb.jpg", "left", "0 0.5 0.5 0.2 0.2\n"),
                  ("img2_jpg.rf.ccc.jpg", "top", None)],          # no label file
        "valid": [("img3_jpg.rf.ddd.jpg", "top", "7 0.5 0.5 0.2 0.2\n")],  # bad class
        "test": [("img1_jpg.rf.eee.jpg", "left", "1 0.5 0.5 0.2 0.2\n")],  # leaks img1
    }
    for split, items in files.items():
        for name, pattern, label in items:
            save_image(tmp_path / split / "images" / name, pattern)
            lbl_dir = tmp_path / split / "labels"
            lbl_dir.mkdir(parents=True, exist_ok=True)
            if label is not None:
                (lbl_dir / name.replace(".jpg", ".txt")).write_text(label)
    return tmp_path


def test_load_class_names_list(dataset):
    assert ad.load_class_names(dataset) == ["a", "b"]


def test_load_class_names_dict(tmp_path):
    (tmp_path / "data.yaml").write_text("names: {0: x, 1: y}\n")
    assert ad.load_class_names(tmp_path) == ["x", "y"]


@pytest.mark.parametrize("name, expected", [
    ("pork_01_jpg.rf.abc123.jpg", "pork_01_jpg"),
    ("plain.jpg", "plain"),
])
def test_source_stem(name, expected):
    assert ad.source_stem(name) == expected


@pytest.mark.parametrize("line, problem", [
    ("0 0.5 0.5 0.2", "expected 5 values"),
    ("0 0.5 abc 0.2 0.2", "non numeric"),
    ("5 0.5 0.5 0.2 0.2", "out of range"),
    ("0 1.5 0.5 0.2 0.2", "outside [0, 1]"),
    ("0 0.5 0.5 0.0 0.2", "zero size"),
])
def test_parse_label_file_rejects_bad_lines(tmp_path, line, problem):
    path = tmp_path / "x.txt"
    path.write_text(line + "\n")
    boxes, errors = ad.parse_label_file(path, n_classes=2)
    assert boxes == []
    assert problem in errors[0]


def test_parse_label_file_accepts_valid_and_blank_lines(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("1 0.5 0.5 0.2 0.2\n\n")
    boxes, errors = ad.parse_label_file(path, n_classes=2)
    assert boxes == [(1, 0.5, 0.5, 0.2, 0.2)]
    assert errors == []


def test_audit_split_counts(dataset):
    s = ad.audit_split(dataset, "train", n_classes=2)
    assert s.n_images == 3
    assert sorted(s.boxes_per_image) == [0, 1, 2]
    assert s.class_counts == {0: 2, 1: 1}
    assert s.missing_labels == ["img2_jpg.rf.ccc"]
    assert s.orphan_labels == []


def test_audit_split_reports_bad_label(dataset):
    s = ad.audit_split(dataset, "valid", n_classes=2)
    assert len(s.errors) == 1 and "out of range" in s.errors[0]


def test_audit_split_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        ad.audit_split(tmp_path, "train", n_classes=2)


def test_find_shared_sources_detects_leak(dataset):
    split_files = {s: [p.name for p in (dataset / s / "images").iterdir()]
                   for s in ad.SPLITS}
    shared = ad.find_shared_sources(split_files)
    assert shared[("train", "test")] == {"img1_jpg"}
    assert shared[("train", "valid")] == set()


def test_find_exact_duplicates(dataset):
    paths = {s: sorted((dataset / s / "images").iterdir()) for s in ad.SPLITS}
    groups = ad.find_exact_duplicates(paths)
    splits_in_groups = [{s for s, _ in g} for g in groups]
    # identical "left" images sit in train and test; identical "top" in train and valid
    assert {"train", "test"} in splits_in_groups
    assert {"train", "valid"} in splits_in_groups


def test_average_hash_same_vs_different(tmp_path):
    save_image(tmp_path / "a.png", "left")
    save_image(tmp_path / "b.png", "left")
    save_image(tmp_path / "c.png", "top")
    ha, hb, hc = (ad.average_hash(tmp_path / n) for n in ("a.png", "b.png", "c.png"))
    assert ha.shape == (64,)
    assert (ha != hb).sum() == 0
    assert (ha != hc).sum() > 5


def test_find_near_duplicates():
    a = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool)
    b = np.array([[0, 0, 0, 1]], dtype=bool)
    assert ad.find_near_duplicates(a, b, max_distance=1) == [(0, 0, 1)]
    assert ad.find_near_duplicates(a, np.empty((0, 4), dtype=bool)) == []


def test_main_runs_end_to_end(dataset, capsys):
    assert ad.main(["--data", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "Split sizes" in out and "train / test: 1 shared" in out


@pytest.mark.parametrize("name, expected", [
    ("frame_with_red_2247_jpg.rf.abc.jpg", 2247),
    ("frame_with_red_15.png", 15),
    ("no_number_here.jpg", None),
])
def test_frame_number(name, expected):
    assert ad.frame_number(name) == expected


def test_nearest_frame_distance():
    d = ad.nearest_frame_distance([5, 100, 0], [1, 10, 98])
    assert d.tolist() == [4, 2, 1]


def test_nearest_frame_distance_empty_reference():
    assert np.isinf(ad.nearest_frame_distance([3], [])).all()