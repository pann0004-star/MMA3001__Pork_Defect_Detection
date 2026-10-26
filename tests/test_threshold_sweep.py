"""Tests for threshold_sweep.py using hand made predictions (no model needed)."""

import numpy as np
import pytest

import threshold_sweep as ts


def test_xywh_to_xyxy():
    assert ts.xywh_to_xyxy([[0.5, 0.5, 0.2, 0.4]]).ravel() == pytest.approx([0.4, 0.3, 0.6, 0.7])


def test_iou_values():
    box = np.array([0, 0, 2, 2])
    others = np.array([[0, 0, 2, 2], [1, 1, 3, 3], [5, 5, 6, 6]])
    assert ts.iou(box, others) == pytest.approx([1.0, 1 / 7, 0.0])


def test_match_image_greedy_by_confidence():
    gt_cls, gt_boxes = [0], [[0, 0, 1, 1]]
    # two predictions on the same box: only the more confident one counts
    tp = ts.match_image([0, 0], [0.3, 0.9], [[0, 0, 1, 1], [0, 0, 1, 1]], gt_cls, gt_boxes)
    assert tp.tolist() == [False, True]


def test_match_image_needs_same_class_and_overlap():
    tp = ts.match_image([1, 0], [0.9, 0.9], [[0, 0, 1, 1], [3, 3, 4, 4]], [0], [[0, 0, 1, 1]])
    assert tp.tolist() == [False, False]


@pytest.fixture
def images():
    """Three trays: one defect caught confidently, one caught weakly, one clean
    tray with a weak false alarm."""
    return [
        {"cls": np.array([0]), "conf": np.array([0.9]), "tp": np.array([True]), "gt_cls": np.array([0])},
        {"cls": np.array([1]), "conf": np.array([0.2]), "tp": np.array([True]), "gt_cls": np.array([1])},
        {"cls": np.array([0]), "conf": np.array([0.3]), "tp": np.array([False]), "gt_cls": np.array([], dtype=int)},
    ]


def test_sweep_tradeoff(images):
    r = ts.sweep(images, n_classes=2, thresholds=[0.1, 0.5])
    # low threshold: both defects caught, one false alarm
    assert r["recall"][0] == 1.0 and r["precision"][0] == pytest.approx(2 / 3)
    assert r["tray_recall"][0] == 1.0 and r["tray_false_alarm"][0] == 1.0
    # high threshold: weak catch and false alarm both disappear
    assert r["recall"][1] == 0.5 and r["precision"][1] == 1.0
    assert r["tray_recall"][1] == 0.5 and r["tray_false_alarm"][1] == 0.0
    assert r["class_recall"][1].tolist() == [1.0, 0.0]


def test_f2_weights_recall(images):
    r = ts.sweep(images, n_classes=2, thresholds=[0.1, 0.5])
    # F2 = 5PR / (4P + R): recall counts twice as much as precision
    assert r["f2"][0] == pytest.approx(5 * (2 / 3) / (4 * (2 / 3) + 1))
    assert r["f2"][1] == pytest.approx(5 * 0.5 / (4 + 0.5))
    assert ts.best_threshold(r, "f2") == 0.1


def test_row_at_handles_missing_class(images):
    r = ts.sweep(images, n_classes=3, thresholds=[0.1, 0.5])
    row = ts.row_at(r, 0.5, ["a", "b", "c"])
    assert row["threshold"] == 0.5 and row["class_recall"]["c"] is None
