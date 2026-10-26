"""Tests for train.py that do not need a GPU or the ultralytics package.

Training itself is too slow for unit tests, so these check the config
preparation and the conversion of metrics into the saved results format.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import train


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Empty split folders plus a Roboflow style data.yaml."""
    (tmp_path / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\nnc: 3\nnames: ['a', 'b', 'c']\n")
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
    return tmp_path


def test_prepare_data_yaml_fixes_paths(dataset):
    out = train.prepare_data_yaml(dataset)
    cfg = yaml.safe_load(out.read_text())
    assert Path(cfg["path"]) == dataset.resolve()
    assert cfg["train"] == "train/images" and cfg["val"] == "valid/images"
    assert cfg["names"] == ["a", "b", "c"] and cfg["nc"] == 3


def test_prepare_data_yaml_missing_split(dataset):
    (dataset / "test" / "images").rmdir()
    with pytest.raises(FileNotFoundError):
        train.prepare_data_yaml(dataset)


class FakeBox:
    """Mimics the parts of Ultralytics' box metrics that train.py reads."""

    ap_class_index = [0, 2]  # class 1 has no examples in this split

    def mean_results(self):
        return [0.8, 0.6, 0.7, 0.4]

    def class_result(self, i):
        return [(0.9, 0.5, 0.8, 0.5), (0.0, 0.0, 0.0, 0.0)][i]


def test_summarise_metrics():
    s = train.summarise_metrics(SimpleNamespace(box=FakeBox()), ["a", "b", "c"])
    assert s["precision"] == 0.8 and s["mAP50_95"] == 0.4
    assert s["f1"] == pytest.approx(2 * 0.8 * 0.6 / 1.4)
    assert s["per_class"]["a"]["recall"] == 0.5
    assert s["per_class"]["b"] is None          # not evaluable, not zero
    assert s["per_class"]["c"]["f1"] == 0.0     # guarded against divide by zero


def test_resume_without_previous_run(dataset, monkeypatch, tmp_path):
    import sys, types
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=None))
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    with pytest.raises(FileNotFoundError):
        train.train_and_evaluate(dataset, "missing_run", resume=True)


def test_read_run_info(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "args.yaml").write_text("model: yolov8n.pt\nepochs: 50\nbatch: 8\nimgsz: 640\n")
    (run / "results.csv").write_text(
        "epoch,time,loss\n1,100.0,2.0\n46,4690.61,1.0\n")
    info = train.read_run_info(run)
    assert info["config"]["batch"] == 8 and info["config"]["epochs"] == 50
    assert info["epochs_run"] == 46
    assert info["train_minutes"] == pytest.approx(78.2)


def test_read_run_info_empty_folder(tmp_path):
    info = train.read_run_info(tmp_path)
    assert info == {"config": {}, "epochs_run": None, "train_minutes": None}


def test_evaluate_run_without_weights(dataset, monkeypatch, tmp_path):
    import sys, types
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=None))
    with pytest.raises(FileNotFoundError):
        train.evaluate_run(dataset, "x", run_dir=tmp_path / "nothing")