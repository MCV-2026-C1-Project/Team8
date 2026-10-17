"""Descriptor arguments shared by precomputation, retrieval and experiments."""
import argparse
import json
from pathlib import Path

from descriptor_config import resolve_descriptor_config
from descriptors import METHODS, DEFAULTS, MASKABLE_METHODS, method_tag


def add_descriptor_arguments(parser, method_flag="--descriptor", allow_all=False):
    parser.add_argument(method_flag, dest="descriptor", choices=[*METHODS, *(["all"] if allow_all else [])])
    parser.add_argument("--config", type=Path, help="JSON descriptor/export (experiment runner: JSON suite)")
    parser.add_argument("--bins", type=int, nargs="+", help="Two joint bins or three 1D/3D bins")
    parser.add_argument("--normalization", choices=("none", "photometric"), default=None,
                        help="Lab/YCbCr preprocessing before partitioning (foreground statistics with masks); default: none")
    for method in ("hsv", "ycbcr", "rgb", "lab"):
        parser.add_argument(f"--{method}-bins", type=int, nargs=3, help="Legacy per-channel bin option")
    for key in ("s-min", "v-min", "chroma-weight"):
        parser.add_argument(f"--{key}", type=float)
    for key in ("v-bins", "l-bins"):
        parser.add_argument(f"--{key}", type=int)
    for key in ("y-range", "chroma-range"):
        parser.add_argument(f"--{key}", type=float, nargs=2)
    for key in ("hue-valid-weight", "hue-smoothing", "filter-hue", "achromatic-bin",
                "include-v", "include-l", "chroma-overflow"):
        parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--spatial-mode", choices=("global", "grid", "pyramid"))
    parser.add_argument("--grids", type=int, nargs="+", help="Grid resolutions, e.g. 1 2 4")
    parser.add_argument("--level-weights", type=float, nargs="+")


def load_descriptor_document(path):
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    config = resolve_descriptor_config(document)
    if "tag" in document and document["tag"] != method_tag(config["descriptor"], **config["params"]):
        raise ValueError("Exported descriptor tag does not match its configuration")
    return config, document.get("measure")


def config_from_args(args, base=None, method=None):
    method = method or args.descriptor or (base["descriptor"] if base else None)
    if method not in METHODS:
        raise ValueError("Specify --descriptor/--method or a descriptor --config")
    if base and base["descriptor"] != method:
        raise ValueError("CLI descriptor must match the loaded configuration")
    params = dict(base["params"]) if base else {}
    if base or args.descriptor not in (None, "all"):
        for alias in ("hsv", "ycbcr", "rgb", "lab"):
            if alias != method and getattr(args, alias + "_bins", None) is not None:
                raise ValueError(f"--{alias}-bins applies only to {alias}; use --bins for {method}")
    for key in set().union(*(p.keys() for p in DEFAULTS.values())) | {"normalization"}:
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    legacy_bins = getattr(args, method + "_bins", None)
    if legacy_bins is not None:
        if args.bins is not None:
            raise ValueError("Use either --bins or the legacy method-specific bins flag")
        params["bins"] = legacy_bins
    if any(getattr(args, key, None) is not None for key in ("spatial_mode", "grids", "level_weights")):
        spatial = dict(params.get("spatial", {}))
        # A new mode starts a new layout; weights are defaulted for new grids.
        if args.spatial_mode is not None:
            spatial = {"mode": args.spatial_mode}
        if args.grids is not None:
            spatial["grids"] = args.grids
            spatial.pop("level_weights", None)
            spatial.setdefault("mode", "grid" if len(args.grids) == 1 else "pyramid")
        if args.level_weights is not None:
            spatial["level_weights"] = args.level_weights
        params["spatial"] = spatial
    config = resolve_descriptor_config(dict(descriptor=method, params=params))
    if getattr(args, "remove_background", False) and method not in MASKABLE_METHODS:
        raise ValueError(f"Foreground masks are supported only for {sorted(MASKABLE_METHODS)}, not {method}")
    return config
