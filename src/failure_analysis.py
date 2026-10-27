"""Classify a detector's mistakes on one split into failure modes.

Every missed defect and every false alarm is put into one category:

* wrong class: the right area was found but given the wrong label;
* poor box: right label and area, but overlap with the true box is too low
  (IoU between 0.1 and 0.5), which also covers duplicate boxes;
* missed entirely / background: nothing overlapping was predicted, or a
  prediction was made where there is no defect.

It also checks whether misses are more common for boxes touching the image
edge (trays only partly in view), and saves annotated images of mistakes:
true boxes in green, predictions in red.

Run from the repo root:

    python src/failure_analysis.py --data data/blocked --name B_blocked_split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_dataset import IMAGE_EXTS, load_class_names, parse_label_file
from threshold_sweep import DEFAULT_CONF, greedy_match, iou, xywh_to_xyxy

RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs")
NEAR_IOU = 0.1   # below this, boxes are treated as unrelated
EDGE_MARGIN = 0.01  # box within 1% of the border counts as touching the edge


def touches_edge(box: np.ndarray, margin: float = EDGE_MARGIN) -> bool:
    """True if a normalised corner box reaches the image border."""
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= 1 - margin or y2 >= 1 - margin


def categorise_image(pred_cls, pred_conf, pred_boxes, gt_cls, gt_boxes) -> dict:
    """Sort every unmatched true box and prediction in one image into a failure mode.

    Args:
        pred_cls, pred_conf, pred_boxes: Predictions already filtered by the
            confidence threshold (corner boxes, normalised).
        gt_cls, gt_boxes: Ground truth classes and corner boxes.

    Returns:
        Dict with lists ``gt`` and ``pred``. Each entry is
        ``(class_id, outcome, other_class)``, where outcome is ``"hit"``,
        ``"wrong_class"``, ``"poor_box"``, ``"missed"`` (true boxes) or
        ``"background"`` (predictions), and other_class is the confused
        class for wrong class cases, otherwise None.
    """
    pred_cls = np.asarray(pred_cls, dtype=int)
    gt_cls = np.asarray(gt_cls, dtype=int)
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)

    matched = greedy_match(pred_cls, pred_conf, pred_boxes, gt_cls, gt_boxes)
    gt_hit = np.zeros(len(gt_cls), dtype=bool)
    gt_hit[matched[matched >= 0]] = True
    free_preds = matched < 0

    # Overlap of every prediction with every true box: rows preds, cols gts.
    overlap = np.array([iou(b, gt_boxes) for b in pred_boxes]).reshape(len(pred_cls), len(gt_cls))

    gt_out = []
    for j in range(len(gt_cls)):
        if gt_hit[j]:
            gt_out.append((int(gt_cls[j]), "hit", None))
            continue
        ov = overlap[:, j] if len(pred_cls) else np.zeros(0)
        other = free_preds & (pred_cls != gt_cls[j]) & (ov >= 0.5)
        same = (pred_cls == gt_cls[j]) & (ov >= NEAR_IOU)
        if other.any():
            k = int(np.flatnonzero(other)[np.argmax(ov[other])])
            gt_out.append((int(gt_cls[j]), "wrong_class", int(pred_cls[k])))
        elif same.any():
            gt_out.append((int(gt_cls[j]), "poor_box", None))
        else:
            gt_out.append((int(gt_cls[j]), "missed", None))

    pred_out = []
    for i in range(len(pred_cls)):
        if matched[i] >= 0:
            pred_out.append((int(pred_cls[i]), "hit", None))
            continue
        ov = overlap[i] if len(gt_cls) else np.zeros(0)
        other = (gt_cls != pred_cls[i]) & (ov >= 0.5)
        same = (gt_cls == pred_cls[i]) & (ov >= NEAR_IOU)
        if other.any():
            k = int(np.flatnonzero(other)[np.argmax(ov[other])])
            pred_out.append((int(pred_cls[i]), "wrong_class", int(gt_cls[k])))
        elif same.any():
            pred_out.append((int(pred_cls[i]), "poor_box", None))
        else:
            pred_out.append((int(pred_cls[i]), "background", None))
    return {"gt": gt_out, "pred": pred_out}


def summarise(images: list[dict], names: list[str]) -> dict:
    """Count failure modes per class and test whether edge boxes are missed more.

    Args:
        images: One dict per image with ``gt`` and ``pred`` outcome lists
            (from ``categorise_image``) and ``gt_edge``, a list of booleans
            saying whether each true box touches the image edge.
        names: Class names.

    Returns:
        Per class counts for true boxes and predictions, a confusion list
        (true class to predicted class) and edge statistics.
    """
    gt_modes = ("hit", "wrong_class", "poor_box", "missed")
    pred_modes = ("hit", "wrong_class", "poor_box", "background")
    gt_counts = {n: dict.fromkeys(gt_modes, 0) for n in names}
    pred_counts = {n: dict.fromkeys(pred_modes, 0) for n in names}
    confusion: dict[str, int] = {}
    edge_total = edge_missed = inner_total = inner_missed = 0

    for im in images:
        for (c, outcome, other), on_edge in zip(im["gt"], im["gt_edge"]):
            gt_counts[names[c]][outcome] += 1
            if outcome == "wrong_class":
                key = f"{names[c]} -> {names[other]}"
                confusion[key] = confusion.get(key, 0) + 1
            failed = outcome != "hit"
            if on_edge:
                edge_total += 1
                edge_missed += failed
            else:
                inner_total += 1
                inner_missed += failed
        for c, outcome, _ in im["pred"]:
            pred_counts[names[c]][outcome] += 1

    def rate(a: int, b: int) -> float | None:
        return a / b if b else None

    return {
        "true_boxes": gt_counts,
        "predictions": pred_counts,
        "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
        "edge": {"boxes": edge_total, "failed": edge_missed,
                 "fail_rate": rate(edge_missed, edge_total)},
        "inside": {"boxes": inner_total, "failed": inner_missed,
                   "fail_rate": rate(inner_missed, inner_total)},
    }


def draw_mistakes(img_path: Path, pred, gt, names: list[str], out_path: Path) -> None:
    """Save a copy of an image with true boxes in green and predictions in red."""
    from PIL import Image, ImageDraw

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        for c, box in zip(gt["cls"], gt["boxes"]):
            x1, y1, x2, y2 = box * [w, h, w, h]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
            draw.text((x1 + 3, y1 + 3), f"TRUE {names[c]}", fill=(0, 255, 0))
        for c, conf, box in zip(pred["cls"], pred["conf"], pred["boxes"]):
            x1, y1, x2, y2 = box * [w, h, w, h]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=2)
            draw.text((x1 + 3, max(y2 - 14, 0)), f"{names[c]} {conf:.2f}", fill=(255, 80, 80))
        im.save(out_path)


def print_summary(s: dict, names: list[str], threshold: float) -> None:
    """Print the failure mode tables."""
    print(f"\nTest set failure modes at confidence {threshold}")
    print(f"\nTrue defects     {'total':>6}{'found':>7}{'wrong cls':>11}{'poor box':>10}{'missed':>8}")
    for n in names:
        c = s["true_boxes"][n]
        total = sum(c.values())
        print(f"{n:<17}{total:>6}{c['hit']:>7}{c['wrong_class']:>11}{c['poor_box']:>10}{c['missed']:>8}")
    print(f"\nPredictions      {'total':>6}{'correct':>9}{'wrong cls':>11}{'poor box':>10}{'background':>12}")
    for n in names:
        c = s["predictions"][n]
        total = sum(c.values())
        print(f"{n:<17}{total:>6}{c['hit']:>9}{c['wrong_class']:>11}{c['poor_box']:>10}{c['background']:>12}")
    print("\nClass confusions (true -> predicted):")
    for pair, k in s["confusion"].items() or []:
        print(f"  {pair}: {k}")
    if not s["confusion"]:
        print("  none")
    e, i = s["edge"], s["inside"]
    fmt = lambda r: "n/a" if r is None else f"{100 * r:.0f}%"  # noqa: E731
    print(f"\nBoxes touching the image edge: {e['failed']} of {e['boxes']} not found ({fmt(e['fail_rate'])})")
    print(f"Boxes fully inside the image:  {i['failed']} of {i['boxes']} not found ({fmt(i['fail_rate'])})")


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="dataset root")
    parser.add_argument("--name", required=True, help="run name, e.g. B_blocked_split")
    parser.add_argument("--split", default="test", help="split to analyse")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--max-images", type=int, default=60, help="annotated images to save")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    weights = RUNS_DIR / args.name / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No trained weights at {weights}")
    names = load_class_names(args.data)
    model = YOLO(str(weights))

    img_dir = Path(args.data) / args.split / "images"
    lbl_dir = Path(args.data) / args.split / "labels"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    out_dir = RUNS_DIR / "failures" / f"{args.name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    images, saved = [], 0
    for res in model.predict(source=[str(p) for p in paths], conf=args.conf, device=args.device,
                             stream=True, verbose=False):
        pred = {"cls": res.boxes.cls.cpu().numpy().astype(int),
                "conf": res.boxes.conf.cpu().numpy(),
                "boxes": res.boxes.xyxyn.cpu().numpy()}
        label = lbl_dir / f"{Path(res.path).stem}.txt"
        boxes = parse_label_file(label, len(names))[0] if label.exists() else []
        gt = {"cls": np.array([b[0] for b in boxes], dtype=int),
              "boxes": xywh_to_xyxy(np.array([b[1:] for b in boxes])) if boxes else np.zeros((0, 4))}
        result = categorise_image(pred["cls"], pred["conf"], pred["boxes"], gt["cls"], gt["boxes"])
        result["gt_edge"] = [touches_edge(b) for b in gt["boxes"]]
        images.append(result)

        modes = {o for _, o, _ in result["gt"] + result["pred"]} - {"hit"}
        if modes and saved < args.max_images:
            tag = "_".join(sorted(modes))
            draw_mistakes(Path(res.path), pred, gt, names, out_dir / f"{tag}__{Path(res.path).name}")
            saved += 1

    summary = summarise(images, names)
    print_summary(summary, names, args.conf)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}_failures_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "split": args.split, "conf": args.conf, **summary}, f, indent=2)
    print(f"\nSaved {out}, and {saved} annotated images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
