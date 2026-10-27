"""Train and evaluate a YOLOv8 detector on one version of the dataset.

Each run trains a pretrained model, picks the best epoch on the validation
split, then scores that model on both validation and test. Scores are saved
to ``results/<name>.json`` so runs can be compared later.

Run from the repo root, for example:

    python src/train.py --data data --name A_roboflow_split
    python src/train.py --data data/blocked --name B_blocked_split
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import yaml

from audit_dataset import load_class_names

RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs")


def prepare_data_yaml(data_dir: Path) -> Path:
    """Write a training config with unambiguous paths for Ultralytics.

    Roboflow's ``data.yaml`` points to ``../train/images``, which is relative
    to a folder above the dataset. This writes ``data_train.yaml`` next to
    it, using the dataset folder as the root so the paths always resolve.

    Args:
        data_dir: Dataset root containing train, valid and test folders.

    Returns:
        Path to the new config file.

    Raises:
        FileNotFoundError: If any split's images folder is missing.
    """
    data_dir = Path(data_dir).resolve()
    for split in ("train", "valid", "test"):
        if not (data_dir / split / "images").is_dir():
            raise FileNotFoundError(f"No images folder at {data_dir / split / 'images'}")
    names = load_class_names(data_dir)
    cfg = {"path": str(data_dir), "train": "train/images", "val": "valid/images",
           "test": "test/images", "nc": len(names), "names": names}
    out = data_dir / "data_train.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def read_run_info(run_dir: Path) -> dict:
    """Read the settings and progress Ultralytics saved in a run folder.

    Args:
        run_dir: A training run folder, e.g. ``runs/A_roboflow_split``.

    Returns:
        The training settings actually used (from ``args.yaml``), the number
        of epochs completed and the training time in minutes (from
        ``results.csv``). Missing files give empty or None values.
    """
    run_dir = Path(run_dir)
    info = {"config": {}, "epochs_run": None, "train_minutes": None}
    args_file = run_dir / "args.yaml"
    if args_file.exists():
        with open(args_file, encoding="utf-8") as f:
            args = yaml.safe_load(f) or {}
        keys = ("model", "epochs", "imgsz", "batch", "patience", "seed", "workers")
        info["config"] = {k: args.get(k) for k in keys}
    csv_file = run_dir / "results.csv"
    if csv_file.exists():
        lines = [ln for ln in csv_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) > 1:
            last = [v.strip() for v in lines[-1].split(",")]
            info["epochs_run"] = int(float(last[0]))
            info["train_minutes"] = round(float(last[1]) / 60, 1)
    return info


def summarise_metrics(metrics, names: list[str]) -> dict:
    """Turn an Ultralytics validation result into a plain dictionary.

    Args:
        metrics: Object returned by ``YOLO.val()``.
        names: Class names ordered by class id.

    Returns:
        Overall precision, recall, F1, mAP50 and mAP50-95, plus the same per
        class. Classes with no examples in the split are recorded as None
        (not evaluable) rather than as zero.
    """
    box = metrics.box
    precision, recall, map50, map50_95 = (float(v) for v in box.mean_results())

    def f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    per_class: dict[str, dict | None] = {name: None for name in names}
    for i, class_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = (float(v) for v in box.class_result(i))
        per_class[names[int(class_id)]] = {"precision": p, "recall": r, "f1": f1(p, r),
                                           "mAP50": ap50, "mAP50_95": ap}
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall),
            "mAP50": map50, "mAP50_95": map50_95, "per_class": per_class}


def train_and_evaluate(
    data_dir: Path,
    name: str,
    model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 15,
    seed: int = 0,
    device: str = "0",
    workers: int = 2,
) -> dict:
    """Train one model and score it on the validation and test splits.

    Args:
        data_dir: Dataset root.
        name: Run name, used for the output folders and results file.
        model: Starting weights. ``yolov8n.pt`` is the pretrained nano model.
        epochs: Maximum passes through the training set.
        imgsz: Image size the model trains at, in pixels.
        batch: Images per training step (lower this if the GPU runs out of memory).
        patience: Stop early if validation mAP has not improved for this many epochs.
        seed: Random seed for reproducibility.
        device: ``"0"`` for the first GPU, ``"cpu"`` for the processor.
        workers: Background processes loading images. Low values are more
            stable on Windows laptops.

    Returns:
        The results dictionary that is also saved to ``results/<name>.json``.
    """
    from ultralytics import YOLO  # imported here so tests run without it

    yaml_path = prepare_data_yaml(data_dir)
    project = str(RUNS_DIR.resolve())

    # Note: Ultralytics' resume option was tested and found unreliable here (it
    # restarted from default settings and overwrote the run), so it is not
    # offered. An interrupted run is scored with --evaluate-only instead.
    trainer_model = YOLO(model)
    trainer_model.train(data=str(yaml_path), epochs=epochs, imgsz=imgsz, batch=batch,
                        patience=patience, seed=seed, deterministic=True, device=device,
                        workers=workers, project=project, name=name, exist_ok=True)

    run_dir = Path(trainer_model.trainer.save_dir)
    expected = RUNS_DIR.resolve() / name
    if run_dir.resolve() != expected:
        print(f"WARNING: training saved to {run_dir}, expected {expected}")
    # Free the training process's memory before scoring. Without this, the
    # scoring step could run out of RAM on a 4 GB GPU laptop.
    del trainer_model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    return evaluate_run(data_dir, name, run_dir=run_dir, device=device, workers=workers)


def evaluate_run(data_dir: Path, name: str, run_dir: Path | None = None,
                 device: str = "0", workers: int = 2) -> dict:
    """Score a finished (or interrupted) run's best weights on valid and test.

    Settings and progress are read from the run folder, so the saved results
    always describe what was actually trained.

    Args:
        data_dir: Dataset root the model should be scored on.
        name: Run name, used for the results file.
        run_dir: Run folder; defaults to ``runs/<name>``.
        device: ``"0"`` for the first GPU, ``"cpu"`` for the processor.
        workers: Background processes loading images during scoring.

    Returns:
        The results dictionary that is also saved to ``results/<name>.json``.

    Raises:
        FileNotFoundError: If the run has no ``best.pt``.
    """
    from ultralytics import YOLO

    run_dir = Path(run_dir) if run_dir is not None else RUNS_DIR / name
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No trained weights at {weights}")

    yaml_path = prepare_data_yaml(data_dir)
    names = load_class_names(data_dir)
    info = read_run_info(run_dir)
    imgsz = info["config"].get("imgsz") or 640
    batch = info["config"].get("batch") or 8
    project = str(RUNS_DIR.resolve())

    best = YOLO(str(weights))
    results = {"name": name, "data": str(Path(data_dir)), **info, "weights": str(weights)}
    for split in ("val", "test"):
        metrics = best.val(data=str(yaml_path), split=split, imgsz=imgsz, batch=batch,
                           device=device, workers=workers, project=project,
                           name=f"{name}_{split}", exist_ok=True)
        results[split] = summarise_metrics(metrics, names)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def print_results(results: dict) -> None:
    """Print a compact score table for one run."""
    print(f"\n== {results['name']} ({results['epochs_run']} epochs, "
          f"{results['train_minutes']} min) ==")
    print(f"{'':<16}{'precision':>10}{'recall':>8}{'F1':>7}{'mAP50':>8}{'mAP50-95':>10}")
    for split in ("val", "test"):
        r = results[split]
        print(f"{split + ' overall':<16}{r['precision']:>10.3f}{r['recall']:>8.3f}"
              f"{r['f1']:>7.3f}{r['mAP50']:>8.3f}{r['mAP50_95']:>10.3f}")
    print("\nTest set per class:")
    for cname, r in results["test"]["per_class"].items():
        if r is None:
            print(f"  {cname:<16} not evaluable (no examples)")
        else:
            print(f"  {cname:<16}{r['precision']:>8.3f}{r['recall']:>8.3f}"
                  f"{r['f1']:>7.3f}{r['mAP50']:>8.3f}{r['mAP50_95']:>10.3f}")


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="dataset root, e.g. data or data/blocked")
    parser.add_argument("--name", required=True, help="run name, e.g. A_roboflow_split")
    parser.add_argument("--model", default="yolov8n.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0", help='"0" for GPU, "cpu" for processor')
    parser.add_argument("--workers", type=int, default=2, help="image loading processes")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="skip training, score runs/<name>/weights/best.pt")
    args = parser.parse_args(argv)

    if args.evaluate_only:
        results = evaluate_run(args.data, args.name, device=args.device, workers=args.workers)
        print_results(results)
        print(f"\nSaved to {RESULTS_DIR / (args.name + '.json')}")
        return 0
    results = train_and_evaluate(args.data, args.name, args.model, args.epochs, args.imgsz,
                                 args.batch, args.patience, args.seed, args.device,
                                 args.workers)
    print_results(results)
    print(f"\nSaved to {RESULTS_DIR / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())