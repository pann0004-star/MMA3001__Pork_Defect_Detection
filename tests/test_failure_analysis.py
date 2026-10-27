"""Tests for failure_analysis.py using hand made boxes (no model needed)."""

from pathlib import Path

import numpy as np
from PIL import Image

import failure_analysis as fa

A = [0.1, 0.1, 0.4, 0.4]      # a box
A_SHIFT = [0.25, 0.25, 0.55, 0.55]  # overlaps A with IoU of about 0.22
FAR = [0.7, 0.7, 0.9, 0.9]    # no overlap with A


def outcomes(result, key):
    return [o for _, o, _ in result[key]]


def test_touches_edge():
    assert fa.touches_edge(np.array([0.0, 0.3, 0.2, 0.5]))
    assert fa.touches_edge(np.array([0.5, 0.5, 1.0, 0.8]))
    assert not fa.touches_edge(np.array([0.2, 0.2, 0.5, 0.5]))


def test_correct_detection_is_a_hit():
    r = fa.categorise_image([0], [0.9], [A], [0], [A])
    assert outcomes(r, "gt") == ["hit"] and outcomes(r, "pred") == ["hit"]


def test_wrong_class_recorded_both_ways():
    r = fa.categorise_image([1], [0.9], [A], [0], [A])
    assert r["gt"] == [(0, "wrong_class", 1)]
    assert r["pred"] == [(1, "wrong_class", 0)]


def test_poor_box():
    r = fa.categorise_image([0], [0.9], [A_SHIFT], [0], [A])
    assert outcomes(r, "gt") == ["poor_box"] and outcomes(r, "pred") == ["poor_box"]


def test_missed_and_background():
    r = fa.categorise_image([0], [0.9], [FAR], [0], [A])
    assert outcomes(r, "gt") == ["missed"] and outcomes(r, "pred") == ["background"]


def test_duplicate_box_counts_as_poor_box():
    r = fa.categorise_image([0, 0], [0.9, 0.5], [A, A], [0], [A])
    assert outcomes(r, "gt") == ["hit"]
    assert sorted(outcomes(r, "pred")) == ["hit", "poor_box"]


def test_no_predictions_or_no_truth():
    assert outcomes(fa.categorise_image([], [], np.zeros((0, 4)), [0], [A]), "gt") == ["missed"]
    assert outcomes(fa.categorise_image([0], [0.9], [A], [], np.zeros((0, 4))), "pred") == ["background"]


def test_summarise_counts_confusion_and_edges():
    names = ["a", "b"]
    im1 = fa.categorise_image([1], [0.9], [A], [0], [A])      # a called b
    im1["gt_edge"] = [True]
    im2 = fa.categorise_image([0], [0.9], [A], [0], [A])      # a found
    im2["gt_edge"] = [False]
    s = fa.summarise([im1, im2], names)
    assert s["true_boxes"]["a"]["hit"] == 1 and s["true_boxes"]["a"]["wrong_class"] == 1
    assert s["predictions"]["b"]["wrong_class"] == 1
    assert s["confusion"] == {"a -> b": 1}
    assert s["edge"]["fail_rate"] == 1.0 and s["inside"]["fail_rate"] == 0.0


def test_draw_mistakes_writes_image(tmp_path: Path):
    src = tmp_path / "in.jpg"
    Image.fromarray(np.zeros((54, 72, 3), dtype=np.uint8)).save(src)
    pred = {"cls": np.array([1]), "conf": np.array([0.8]), "boxes": np.array([A])}
    gt = {"cls": np.array([0]), "boxes": np.array([A])}
    fa.draw_mistakes(src, pred, gt, ["a", "b"], tmp_path / "out.jpg")
    with Image.open(tmp_path / "out.jpg") as im:
        assert im.size == (72, 54)
