"""Deterministic Week 4 grid expansion; paths are relative to the specification."""
import copy
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from local_matching_config import ROOT, resolve_run_config, section, number


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def expand_local_experiments(spec, directory=ROOT):
    spec = section(spec, dict(base_config={}, shared={}, search={}, evaluation={}), "experiment specification")
    base = spec["base_config"]
    if isinstance(base, str):
        base = json.loads((Path(directory) / base).read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(spec["shared"], dict) or not isinstance(spec["search"], dict):
        raise ValueError("base_config/shared/search must be objects (base_config may also be a JSON path)")
    base = merge(base, spec["shared"])
    if base.get("segmentation") is None:
        base["segmentation"] = json.loads((ROOT / "configs/week3/task3_selected.json").read_text(encoding="utf-8"))
    evaluation = section(spec["evaluation"], dict(k_values=[1,5], select_rejection_threshold=True), "evaluation")
    ks = evaluation["k_values"]
    if not isinstance(ks, list) or not ks or any(number(k, "evaluation K", integer=True) < 1 for k in ks):
        raise ValueError("k_values must be positive integers")
    evaluation["k_values"] = sorted(set(ks) | {1,5})  # Selection always needs both metrics.
    if not isinstance(evaluation["select_rejection_threshold"], bool):
        raise ValueError("select_rejection_threshold must be boolean")
    keys = sorted(spec["search"])
    choices = [spec["search"][k] for k in keys]
    if any(not isinstance(v,list) or not v for v in choices):
        raise ValueError("Every search dimension needs a nonempty choices list")
    if math.prod(map(len,choices)) > 5000:
        raise ValueError("Grid exceeds 5000 combinations; split the specification")
    if any(b.startswith(a+".") for a in keys for b in keys if a != b):
        raise ValueError("Overlapping search paths are ambiguous")
    jobs = {}
    for values in product(*choices):
        config = copy.deepcopy(base)
        for key,value in zip(keys,values):
            parts = key.split(".")
            if len(parts) > 2 or any(not p for p in parts):
                raise ValueError(f"Unsupported search path: {key}")
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part,{})
                if not isinstance(target,dict):
                    raise ValueError(f"Non-object search parent: {key}")
            target[parts[-1]] = value
        config = resolve_run_config(config)
        if config["retrieval"]["k"] < max(evaluation["k_values"]):
            raise ValueError("retrieval.k must cover every evaluation K (including 1 and 5)")
        if not evaluation["select_rejection_threshold"] and config["rejection"]["mode"] != "min_inliers":
            raise ValueError("Fixed-threshold experiments require rejection.mode=min_inliers")
        if evaluation["select_rejection_threshold"]:
            config["rejection"] = dict(mode="disabled", min_inliers=None)
        identity = dict(version=1, config=config, evaluation=evaluation)
        key = hashlib.sha256(canonical(identity).encode()).hexdigest()[:24]
        jobs[key] = dict(id=key, config=config, evaluation=copy.deepcopy(evaluation))
    return [jobs[key] for key in sorted(jobs)]
