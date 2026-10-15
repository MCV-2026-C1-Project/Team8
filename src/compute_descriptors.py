"""Precompute colour/spatial descriptors using the same settings as retrieval."""
import argparse
from pathlib import Path

from descriptor_cli import add_descriptor_arguments, config_from_args, load_descriptor_document
from descriptors import compute_folder, method_tag
from retrieval import DESC_DIR, save_descriptor_cache

# Preserve --method all as the original Week 1 set, rather than silently
# precomputing every new joint descriptor too.
LEGACY_METHODS = ["ycbcr", "hsv", "hsv_baseline", "rgb", "lab"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_descriptor_arguments(ap, "--method", allow_all=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--name", help="Cache prefix; default: folder name before underscore")
    ap.add_argument("--out", "--cache-dir", dest="out", type=Path, default=DESC_DIR)
    args = ap.parse_args(argv)
    try:
        base, _ = load_descriptor_document(args.config) if args.config else (None, None)
        if base and args.descriptor == "all":
            raise ValueError("--config describes one method and cannot be used with --method all")
        methods = (LEGACY_METHODS if args.descriptor == "all" or (not args.descriptor and not base)
                   else [args.descriptor or base["descriptor"]])
        configs = []
        for method in methods:
            selected = argparse.Namespace(**vars(args))
            if len(methods) > 1:
                from descriptors import DEFAULTS
                # Legacy --method all applies colour options only to their methods.
                for key in set().union(*(p.keys() for p in DEFAULTS.values())) - set(DEFAULTS[method]):
                    if hasattr(selected, key):
                        setattr(selected, key, None)
            configs.append(config_from_args(selected, base, method))
        name = args.name or args.data.name.split("_")[0].lower()
        for config in configs:
            method, params = config["descriptor"], config["params"]
            ids, desc = compute_folder(args.data, method, **params)
            path = args.out / f"{name}_{method_tag(method, **params)}.npz"
            save_descriptor_cache(path, ids, desc, method, params, args.data)
            print(f"Saved {path} ids={ids.shape} desc={desc.shape} {desc.dtype}")
    except (ValueError, TypeError, OSError, KeyError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
