"""Tune the confidence threshold of a trained detector.

Every prediction has a confidence score, and only predictions above a
threshold are acted on. A lower threshold catches more real defects (higher
recall) at the cost of more false alarms (lower precision).

The threshold is chosen on the validation split by maximising F2, a score
that weights recall twice as much as precision (a missed defect costs more
than a tray being double checked). It is then applied once to the test
split. Results are also reported per tray, as a pass/fail inspection.

Run from the repo root:

    python src/threshold_sweep.py --data data/blocked --name B_blocked_split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_dataset import IMAGE_EXTS, load_class_names, parse_label_file

RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs")
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
DEFAULT_CONF = 0.25  # Ultralytics' default prediction threshold


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (x_centre, y_centre, width, height) boxes to corner format."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    half = boxes[:, 2:] / 2
    return np.hstack([boxes[:, :2] - half, boxes[:, :2] + half])


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Intersection over union between one box and many (corner format).

    Args:
        box: Shape (4,) as (x1, y1, x2, y2).
        boxes: Shape (n, 4).

    Returns:
        Shape (n,) overlap scores between 0 and 1.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_image(pred_cls, pred_conf, pred_boxes, gt_cls, gt_boxes, iou_thr: float = 0.5):
    """Mark each prediction in one image as a true or false positive.

    Predictions are matched greedily from most to least confident. A
    prediction is a true positive if it overlaps an unmatched ground truth
    box of the same class with IoU of at least ``iou_thr``. Because higher
    confidence predictions claim boxes first, raising the threshold later
    never changes which of the remaining predictions are true positives.

    Args:
        pred_cls, pred_conf, pred_boxes: Predicted classes, confidences and
            corner boxes.
        gt_cls, gt_boxes: Ground truth classes and corner boxes.
        iou_thr: Minimum overlap to count as a match.

    Returns:
        Boolean array, True where the prediction is a true positive.
    """
    pred_cls = np.asarray(pred_cls, dtype=int)
    gt_cls = np.asarray(gt_cls, dtype=int)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    is_tp = np.zeros(len(pred_cls), dtype=bool)
    taken = np.zeros(len(gt_cls), dtype=bool)
    for i in np.argsort(-np.asarray(pred_conf, dtype=float), kind="stable"):
        candidates = (gt_cls == pred_cls[i]) & ~taken
        if not candidates.any():
            continue
        overlaps = np.where(candidates, iou(np.asarray(pred_boxes[i]), gt_boxes), -1.0)
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_thr:
            is_tp[i] = True
            taken[best] = True
    return is_tp


def sweep(images: list[dict], n_classes: int, thresholds=THRESHOLDS) -> dict:
    """Compute detection and tray level metrics at each threshold.

    Args:
        images: One dict per image with keys ``cls``, ``conf`` and ``tp``
            (predictions, already matched) and ``gt_cls`` (ground truth).
        n_classes: Number of classes.
        thresholds: Confidence thresholds to evaluate.

    Returns:
        Dict of arrays indexed by threshold: pooled precision, recall, F1
        and F2; per class recall; tray recall and tray false alarm rate.
    """
    gt_total = np.zeros(n_classes)
    for im in images:
        np.add.at(gt_total, np.asarray(im["gt_cls"], dtype=int), 1)
    defective = np.array([len(im["gt_cls"]) > 0 for im in images])

    out = {k: [] for k in ("precision", "recall", "f1", "f2", "class_recall",
                           "tray_recall", "tray_false_alarm")}
    for t in thresholds:
        tp = np.zeros(n_classes)
        fp = 0
        flagged = np.zeros(len(images), dtype=bool)
        for k, im in enumerate(images):
            keep = np.asarray(im["conf"], dtype=float) >= t
            flagged[k] = keep.any()
            hits = np.asarray(im["tp"], dtype=bool) & keep
            np.add.at(tp, np.asarray(im["cls"], dtype=int)[hits], 1)
            fp += int((keep & ~np.asarray(im["tp"], dtype=bool)).sum())
        p = tp.sum() / (tp.sum() + fp) if tp.sum() + fp > 0 else 1.0
        r = tp.sum() / gt_total.sum() if gt_total.sum() > 0 else 0.0
        out["precision"].append(p)
        out["recall"].append(r)
        out["f1"].append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        out["f2"].append(5 * p * r / (4 * p + r) if 4 * p + r > 0 else 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["class_recall"].append(np.where(gt_total > 0, tp / gt_total, np.nan))
        out["tray_recall"].append(flagged[defective].mean() if defective.any() else np.nan)
        out["tray_false_alarm"].append(flagged[~defective].mean() if (~defective).any() else np.nan)
    return {"thresholds": np.asarray(thresholds), **{k: np.asarray(v) for k, v in out.items()}}


def best_threshold(result: dict, metric: str = "f2") -> float:
    """Return the threshold with the highest value of ``metric``."""
    return float(result["thresholds"][int(np.nanargmax(result[metric]))])


def row_at(result: dict, threshold: float, names: list[str]) -> dict:
    """Extract every metric at one threshold as plain Python values."""
    i = int(np.argmin(np.abs(result["thresholds"] - threshold)))
    row = {k: float(result[k][i]) for k in ("precision", "recall", "f1", "f2",
                                             "tray_recall", "tray_false_alarm")}
    row["class_recall"] = {n: (None if np.isnan(v) else float(v))
                           for n, v in zip(names, result["class_recall"][i])}
    return {"threshold": float(result["thresholds"][i]), **row}


def collect_predictions(model, data_dir: Path, split: str, device: str) -> list[dict]:
    """Run the model on one split and match its predictions to the labels.

    Predictions are kept down to a confidence of 0.001 so every threshold
    can be evaluated afterwards from a single pass over the images.
    """
    img_dir = Path(data_dir) / split / "images"
    lbl_dir = Path(data_dir) / split / "labels"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    images = []
    for res in model.predict(source=[str(p) for p in paths], conf=0.001, device=device,
                             stream=True, verbose=False):
        boxes = res.boxes
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxyn.cpu().numpy()
        label = lbl_dir / f"{Path(res.path).stem}.txt"
        gt = parse_label_file(label, 10**6)[0] if label.exists() else []
        gt_cls = np.array([b[0] for b in gt], dtype=int)
        gt_boxes = xywh_to_xyxy(np.array([b[1:] for b in gt])) if gt else np.zeros((0, 4))
        tp = match_image(cls, conf, xyxy, gt_cls, gt_boxes)
        images.append({"cls": cls, "conf": conf, "tp": tp, "gt_cls": gt_cls})
    return images


def plot_sweep(result: dict, chosen: float, out_path: Path) -> None:
    """Save a precision, recall and F2 against threshold chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f2", "F2"),
                       ("tray_recall", "Tray recall")):
        ax.plot(result["thresholds"], result[key], label=label)
    ax.axvline(DEFAULT_CONF, color="grey", linestyle=":", label=f"Default ({DEFAULT_CONF})")
    ax.axvline(chosen, color="black", linestyle="--", label=f"Chosen ({chosen})")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score (validation)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_comparison(rows: dict[str, dict], names: list[str]) -> None:
    """Print test metrics at the default and chosen thresholds side by side."""
    labels = list(rows)
    print(f"\n{'Test set':<22}" + "".join(f"{lab:>16}" for lab in labels))
    for key, title in (("threshold", "threshold"), ("precision", "precision"),
                       ("recall", "recall"), ("f2", "F2"), ("tray_recall", "trays caught"),
                       ("tray_false_alarm", "good trays flagged")):
        print(f"{title:<22}" + "".join(f"{rows[lab][key]:>16.2f}" for lab in labels))
    print("recall per class")
    for n in names:
        vals = [rows[lab]["class_recall"][n] for lab in labels]
        print(f"  {n:<20}" + "".join(f"{'n/a' if v is None else f'{v:.2f}':>16}" for v in vals))


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="dataset root the model was trained on")
    parser.add_argument("--name", required=True, help="run name, e.g. B_blocked_split")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    weights = RUNS_DIR / args.name / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No trained weights at {weights}")
    names = load_class_names(args.data)
    model = YOLO(str(weights))

    val = sweep(collect_predictions(model, args.data, "valid", args.device), len(names))
    test = sweep(collect_predictions(model, args.data, "test", args.device), len(names))
    chosen = best_threshold(val, "f2")

    print(f"\nThreshold chosen on validation (max F2): {chosen}")
    rows = {f"default {DEFAULT_CONF}": row_at(test, DEFAULT_CONF, names),
            f"chosen {chosen}": row_at(test, chosen, names)}
    print_comparison(rows, names)

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_path = RESULTS_DIR / f"{args.name}_threshold_sweep.png"
    plot_sweep(val, chosen, plot_path)
    summary = {"name": args.name, "chosen_threshold": chosen, "selection": "max F2 on validation",
               "validation": [row_at(val, t, names) for t in THRESHOLDS],
               "test": rows}
    out = RESULTS_DIR / f"{args.name}_threshold_sweep.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out} and {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
