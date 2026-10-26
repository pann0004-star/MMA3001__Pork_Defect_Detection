"""Tests for oversample.py (repeat factor sampling)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import audit_dataset as ad
import oversample as os_


def test_class_fractions():
    f = os_.class_fractions([{0}, {0, 1}, set(), {0}], n_classes=3)
    assert f.tolist() == [0.75, 0.25, 0.0]


def test_class_repeat_factors():
    r = os_.class_repeat_factors(np.array([0.5, 0.025, 0.0]), threshold=0.1)
    assert r[0] == 1.0                      # common class untouched
    assert r[1] == pytest.approx(2.0)       # sqrt(0.1 / 0.025)
    assert r[2] == 1.0                      # absent class, nothing to repeat


def test_image_repeat_factors_takes_max():
    r = os_.image_repeat_factors([{0}, {0, 1}, set()], np.array([1.0, 3.0]))
    assert r.tolist() == [1.0, 3.0, 1.0]


def test_copies_from_factors_unbiased_and_reproducible():
    factors = np.full(20000, 2.3)
    copies = os_.copies_from_factors(factors, seed=1)
    assert set(np.unique(copies)) == {2, 3}
    assert copies.mean() == pytest.approx(2.3, abs=0.02)
    assert (copies == os_.copies_from_factors(factors, seed=1)).all()
    assert (os_.copies_from_factors(np.array([1.0, 4.0])) == [1, 4]).all()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """20 train images: 19 with common class 0, one with rare class 1."""
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['common', 'rare']\n")
    for split, n in (("train", 20), ("valid", 2), ("test", 2)):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        for i in range(n):
            name = f"frame_with_red_{i}_jpg.rf.x{i}"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
                tmp_path / split / "images" / f"{name}.jpg")
            cls = 1 if (split == "train" and i == 0) else 0
            (tmp_path / split / "labels" / f"{name}.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n")
    return tmp_path


def test_oversample_end_to_end(dataset, tmp_path):
    out = tmp_path / "os"
    s = os_.oversample(dataset, out, threshold=0.2, seed=0)
    # rare class is in 1 of 20 images: factor sqrt(0.2 / 0.05) = 2
    assert s["class_factors"][1] == pytest.approx(2.0)
    assert s["images_after"] == 21
    assert s["boxes_after"].tolist() == [19, 2]
    # valid and test unchanged, and the new dataset passes the audit
    assert len(list((out / "test" / "images").iterdir())) == 2
    stats = ad.audit_split(out, "train", 2)
    assert stats.n_images == 21 and stats.errors == [] and stats.missing_labels == []
    assert stats.class_counts[1] == 2
    assert ad.load_class_names(out) == ["common", "rare"]


def test_oversample_refuses_overwrite(dataset, tmp_path):
    (tmp_path / "os").mkdir()
    with pytest.raises(FileExistsError):
        os_.oversample(dataset, tmp_path / "os")
    with pytest.raises(ValueError):
        os_.oversample(dataset, dataset)
