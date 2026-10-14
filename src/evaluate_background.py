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
    parser = argparse.ArgumentParser(
        description="Generate and evaluate color-based background masks."
    )
    parser.add_argument("--images", required=True,
                        help="Folder containing QS2 images.")
    parser.add_argument("--gt", required=True,
                        help="Folder containing ground-truth masks.")
    parser.add_argument("--output", default="results/background_masks",
                        help="Folder for predicted masks and metrics.csv.")
    parser.add_argument("--border-ratio", type=float, default=0.08)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--mad-multiplier", type=float, default=3.0)
    parser.add_argument("--sample-step", type=int, default=4)
    args = parser.parse_args()

    image_dir = Path(args.images)
    gt_dir = Path(args.gt)
    output_dir = Path(args.output)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    rows = []
    total_tp = total_fp = total_fn = 0

    for image_path in tqdm(image_paths, desc="Evaluating masks"):
        # Find the matching ground-truth mask by image stem.
        candidates = [
            gt_dir / f"{image_path.stem}{ext}"
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif")
        ]
        gt_path = next((p for p in candidates if p.is_file()), None)

        if gt_path is None:
            print(f"Skipping {image_path.name}: no ground truth found.")
            continue

        image = load_image(image_path)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if gt is None:
            print(f"Skipping unreadable ground truth: {gt_path}")
            continue

        if image.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"Size mismatch for {image_path.name}: "
                f"image={image.shape[:2]}, GT={gt.shape[:2]}"
            )

        pred = foreground_mask(
            image,
            border_ratio=args.border_ratio,
            threshold=args.threshold,
            mad_multiplier=args.mad_multiplier,
            sample_step=args.sample_step,
        )

        scores = evaluate_mask(pred, gt)
        rows.append({"image": image_path.name, **scores})

        total_tp += scores["tp"]
        total_fp += scores["fp"]
        total_fn += scores["fn"]

        # Save each predicted mask.
        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), pred)

    if not rows:
        raise ValueError("No images could be evaluated. Check image/mask names.")

    # Macro average: every image contributes equally.
    macro = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("precision", "recall", "f1")
    }

    # Global pixel scores: pool TP, FP and FN across all images.
    global_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0.0
    )
    global_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0.0
    )
    global_f1 = (
        2 * global_precision * global_recall /
        (global_precision + global_recall)
        if global_precision + global_recall else 0.0
    )

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluated images: {len(rows)}")
    print("\nMacro-average (mean per image):")
    for key, value in macro.items():
        print(f"  {key.capitalize():<10}: {value:.4f}")

    print("\nGlobal pixel-level scores:")
    print(f"  Precision : {global_precision:.4f}")
    print(f"  Recall    : {global_recall:.4f}")
    print(f"  F1        : {global_f1:.4f}")

    print(f"\nPredicted masks: {mask_dir}")
    print(f"Per-image metrics: {csv_path}")


if __name__ == "__main__":
    main()
