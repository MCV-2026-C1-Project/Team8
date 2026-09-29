"""Compute and save 1D colour histogram descriptors for an image folder.

Examples:
    py src/compute_descriptors.py --data data/BBDD --method all
    py src/compute_descriptors.py --data data/qsd1_w1 --method ycbcr --chroma-range 64 192
"""
import argparse
from pathlib import Path

import numpy as np

from descriptors import compute_folder, method_tag

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder with the .jpg images")
    ap.add_argument("--name", default=None,
                    help="Dataset prefix for output files (default: folder name before '_', lowercased)")
    ap.add_argument("--method", choices=["ycbcr", "hsv", "all"], default="all")
    ap.add_argument("--ycbcr-bins", type=int, nargs=3, default=[8, 32, 32], metavar=("Y", "CR", "CB"))
    ap.add_argument("--chroma-range", type=int, nargs=2, default=[0, 256], metavar=("LO", "HI"))
    ap.add_argument("--hsv-bins", type=int, nargs=3, default=[32, 16, 8], metavar=("H", "S", "V"))
    ap.add_argument("--s-min", type=int, default=40)
    ap.add_argument("--v-min", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "descriptors"))
    args = ap.parse_args()

    name = args.name or Path(args.data).name.split("_")[0].lower()
    params = {
        "ycbcr": dict(bins=tuple(args.ycbcr_bins), chroma_range=tuple(args.chroma_range)),
        "hsv": dict(bins=tuple(args.hsv_bins), s_min=args.s_min, v_min=args.v_min),
    }
    methods = ["ycbcr", "hsv"] if args.method == "all" else [args.method]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in methods:
        ids, desc = compute_folder(args.data, m, **params[m])
        path = out / f"{name}_{method_tag(m, **params[m])}.npz"
        np.savez(path, ids=ids, desc=desc)
        print(f"Saved {path}  ids={ids.shape}  desc={desc.shape} {desc.dtype}")


if __name__ == "__main__":
    main()
