"""Compute and save 1D colour histogram descriptors for an image folder.

Examples:
    py src/compute_descriptors.py --data data/BBDD --method all
    py src/compute_descriptors.py --data data/qsd1_w1 --method ycbcr --chroma-range 64 192
"""
import argparse
from pathlib import Path

from descriptors import METHODS, DEFAULTS, compute_folder, method_tag
from retrieval import save_descriptor_cache

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder with the .jpg images")
    ap.add_argument("--name", default=None,
                    help="Dataset prefix for output files (default: folder name before '_', lowercased)")
    ap.add_argument("--method", choices=[*METHODS, "all"], default="all")
    ap.add_argument("--ycbcr-bins", type=int, nargs=3, default=DEFAULTS["ycbcr"]["bins"], metavar=("Y", "CR", "CB"))
    ap.add_argument("--chroma-range", type=int, nargs=2, default=DEFAULTS["ycbcr"]["chroma_range"], metavar=("LO", "HI"))
    ap.add_argument("--chroma-overflow", action="store_true",
                    help="Add below/above bins counting Cr/Cb pixels outside --chroma-range")
    ap.add_argument("--hsv-bins", type=int, nargs=3, default=DEFAULTS["hsv"]["bins"], metavar=("H", "S", "V"))
    ap.add_argument("--s-min", type=int, default=DEFAULTS["hsv"]["s_min"])
    ap.add_argument("--v-min", type=int, default=DEFAULTS["hsv"]["v_min"])
    ap.add_argument("--hue-valid-weight", action=argparse.BooleanOptionalAction, default=DEFAULTS["hsv"]["hue_valid_weight"],
                    help="HSV only: scale Hue by the fraction of valid pixels")
    ap.add_argument("--hue-smoothing", action=argparse.BooleanOptionalAction, default=DEFAULTS["hsv"]["hue_smoothing"],
                    help="HSV only: circularly smooth the Hue histogram")
    ap.add_argument("--rgb-bins", type=int, nargs=3, default=DEFAULTS["rgb"]["bins"], metavar=("R", "G", "B"))
    ap.add_argument("--lab-bins", type=int, nargs=3, default=DEFAULTS["lab"]["bins"], metavar=("L", "A", "B"))
    ap.add_argument("--out", default=str(ROOT / "descriptors"))
    args = ap.parse_args()

    name = args.name or Path(args.data).name.split("_")[0].lower()
    params = {
        "ycbcr": dict(bins=tuple(args.ycbcr_bins), chroma_range=tuple(args.chroma_range),
                      chroma_overflow=args.chroma_overflow),
        "hsv": dict(bins=tuple(args.hsv_bins), s_min=args.s_min, v_min=args.v_min,
                    hue_valid_weight=args.hue_valid_weight, hue_smoothing=args.hue_smoothing),
        "hsv_baseline": dict(bins=tuple(DEFAULTS["hsv_baseline"]["bins"])),
        "rgb": dict(bins=tuple(args.rgb_bins)),
        "lab": dict(bins=tuple(args.lab_bins)),
    }
    methods = list(METHODS) if args.method == "all" else [args.method]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in methods:
        ids, desc = compute_folder(args.data, m, **params[m])
        path = out / f"{name}_{method_tag(m, **params[m])}.npz"
        save_descriptor_cache(path, ids, desc, m, params[m])
        print(f"Saved {path}  ids={ids.shape}  desc={desc.shape} {desc.dtype}")


if __name__ == "__main__":
    main()
