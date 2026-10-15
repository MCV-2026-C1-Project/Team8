"""Expand Python or JSON experiment dictionaries through the same validation."""
import math
from itertools import product

from descriptor_config import resolve_descriptor_config, measure_channels
from descriptors import method_tag

DEFAULT_MEASURES = ["l1", "chi2", "hellinger"]
MAX_COMBINATIONS = 5000


def suite_settings(experiments):
    if isinstance(experiments, list):
        return dict(experiments=experiments)
    if not isinstance(experiments, dict):
        raise ValueError("An experiment suite must be a dictionary or a list of experiments")
    unknown = set(experiments) - {"id", "experiments", "spatial", "measures", "db_dir", "query_dir", "gt"}
    if unknown:
        raise ValueError(f"Unknown suite settings: {sorted(unknown)}")
    if not isinstance(experiments.get("experiments"), list) or not experiments["experiments"]:
        raise ValueError("A suite requires a nonempty experiments list")
    return experiments


def expand_experiments(experiments, max_combinations=MAX_COMBINATIONS):
    suite = suite_settings(experiments)
    if not suite["experiments"]:
        raise ValueError("An experiments list cannot be empty")
    jobs, total = {}, 0
    for index, entry in enumerate(suite["experiments"]):
        if not isinstance(entry, dict) or set(entry) - {"id", "descriptor", "params", "grid", "spatial", "measures"}:
            raise ValueError(f"Invalid experiment at index {index}")
        grid = entry.get("grid", {})
        params = entry.get("params", {})
        if not isinstance(grid, dict) or not isinstance(params, dict):
            raise ValueError("grid and params must be dictionaries")
        if any(not isinstance(choices, (list, tuple)) or not choices for choices in grid.values()):
            raise ValueError("Each grid parameter requires a nonempty list of candidate values")
        if "spatial" in grid:
            raise ValueError("Use the spatial list for layouts, not a spatial entry inside grid")
        spatial = entry.get("spatial", suite.get("spatial", [None]))
        if isinstance(spatial, dict):
            spatial = [spatial]
        if not isinstance(spatial, (list, tuple)) or not spatial or any(s is not None and not isinstance(s, dict) for s in spatial):
            raise ValueError("spatial must be a configuration or a nonempty list of configurations")
        measures = entry.get("measures", suite.get("measures", DEFAULT_MEASURES))
        if not isinstance(measures, (list, tuple)) or not measures:
            raise ValueError("measures must be a nonempty list")
        count = math.prod(len(choices) for choices in grid.values()) * len(spatial)
        total += count * len(measures)
        if total > max_combinations:
            raise ValueError(f"Grid expands beyond {max_combinations} configuration/measure combinations; split the suite")
        if "spatial" in params and ("spatial" in entry or "spatial" in suite):
            raise ValueError("Specify spatial inside params or as a layout list, not both")
        for values in product(*grid.values()):
            for layout in spatial:
                effective = dict(params)
                effective.update(zip(grid, values))
                if layout is not None:
                    effective["spatial"] = layout
                config = resolve_descriptor_config(dict(descriptor=entry.get("descriptor"), params=effective))
                method, effective = config["descriptor"], config["params"]
                for measure in measures:
                    measure_channels(method, effective, measure)
                tag = method_tag(method, **effective)
                job = jobs.setdefault(tag, dict(config=config, measures=[], ids=[]))
                job["measures"].extend(m for m in measures if m not in job["measures"])
                identifier = str(entry.get("id", suite.get("id", f"experiment_{index + 1}")))
                if identifier not in job["ids"]:
                    job["ids"].append(identifier)
    return list(jobs.values())
