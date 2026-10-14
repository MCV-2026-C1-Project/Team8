import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from remove_background import foreground_mask


def evaluate_mask(predicted, ground_truth):
    """Evaluate binary masks, treating nonzero pixels as foreground."""
    pred = predicted > 0
    gt = ground_truth > 0

    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def load_image(path):
    # Supports paths containing non-ASCII characters.
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", default="results/background_masks")
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--border-ratio", type=float, default=0.08)
    parser.add_argument("--mad-multiplier", type=float, default=3.0)
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--grid-search", action="store_true", help="Test thresholds and select the best one.")
    args = parser.parse_args()

    thresholds = [10, 15, 20, 25] if args.grid_search else [args.threshold]

    image_dir = Path(args.images)
    gt_dir = Path(args.gt)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"}
    )

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    # Find matching ground-truth masks.
    pairs = []
    for image_path in image_paths:
        candidates = [
            gt_dir / f"{image_path.stem}{ext}"
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif")
        ]
        gt_path = next((p for p in candidates if p.is_file()), None)

        if gt_path is not None:
            pairs.append((image_path, gt_path))
        else:
            print(f"Skipping {image_path.name}: ground truth not found.")

    if not pairs:
        raise ValueError("No matching image and ground-truth mask pairs found.")

    results = []

    for threshold in thresholds:
        print(f"\nEvaluating threshold = {threshold}")

        # Separate output directories for each threshold.
        if args.grid_search:
            mask_dir = output_dir / f"threshold_{threshold}" / "masks"
        else:
            mask_dir = output_dir / "masks"

        mask_dir.mkdir(parents=True, exist_ok=True)
        image_scores = []

        for image_path, gt_path in tqdm(pairs):
            image = load_image(image_path)
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

            if gt is None:
                print(f"Could not read mask: {gt_path}")
                continue

            if image.shape[:2] != gt.shape[:2]:
                raise ValueError(
                    f"Image/mask size mismatch: {image_path.name}"
                )

            pred = foreground_mask(
                image,
                border_ratio=args.border_ratio,
                threshold=threshold,
                mad_multiplier=args.mad_multiplier,
                sample_step=args.sample_step,
            )

            scores = evaluate_mask(pred, gt)
            image_scores.append(scores)

            cv2.imwrite(
                str(mask_dir / f"{image_path.stem}.png"), pred
            )

        if not image_scores:
            continue

        results.append({
            "threshold": threshold,
            "precision": np.mean([s["precision"] for s in image_scores]),
            "recall": np.mean([s["recall"] for s in image_scores]),
            "f1": np.mean([s["f1"] for s in image_scores]),
        })

    if not results:
        raise ValueError("No images could be evaluated.")

    # Sort by mean F1, highest first.
    results.sort(key=lambda r: r["f1"], reverse=True)

    if args.grid_search:
        csv_path = output_dir / "threshold_comparison.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["threshold", "precision", "recall", "f1"]
            )
            writer.writeheader()
            writer.writerows(results)

        print("\n=== Grid search results (best F1 first) ===")
        print(f"{'Threshold':>10} {'Precision':>12} {'Recall':>12} {'F1':>12}")

        for r in results:
            print(
                f"{r['threshold']:>10} "
                f"{r['precision']:>12.4f} "
                f"{r['recall']:>12.4f} "
                f"{r['f1']:>12.4f}"
            )

        print(f"\nBest threshold: {results[0]['threshold']}")
        print(f"Best mean F1:   {results[0]['f1']:.4f}")
        print(f"Comparison saved to: {csv_path}")

    else:
        r = results[0]
        print(f"\nThreshold: {r['threshold']}")
        print(f"Mean precision: {r['precision']:.4f}")
        print(f"Mean recall:    {r['recall']:.4f}")
        print(f"Mean F1:        {r['f1']:.4f}")


if __name__ == "__main__":
    main()
