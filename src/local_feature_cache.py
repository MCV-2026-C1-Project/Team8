"""Complete-database Week 4 NPZ caches: numeric arrays, offsets, JSON; no pickle."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile
import zlib

import cv2
import numpy as np
from tqdm import tqdm

from descriptor_config import canonical_json
from descriptors import list_images, load_image
from local_features import LocalFeatures, extract_local_features, resolve_local_config
from retrieval import DESC_DIR, source_identity


SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = DESC_DIR / "week4"


@dataclass
class DatabaseFeatures:
    ids: list
    filenames: list
    features: list
    cache_path: Path
    cache_status: str
    cache_reason: str


def local_feature_cache_path(db_dir, method="sift", cache_dir=DEFAULT_CACHE_DIR, **params):
    config = resolve_local_config(method, params)
    identity = [str(Path(db_dir).resolve()), config, SCHEMA_VERSION, cv2.__version__]
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:24]
    return Path(cache_dir) / f"local_v{SCHEMA_VERSION}_{method}_{digest}.npz"


def _read_cache(path, metadata, ids, filenames):
    with np.load(path, allow_pickle=False) as data:
        required = {"metadata", "ids", "filenames", "shapes", "offsets", "keypoints", "descriptors"}
        if set(data.files) != required:
            raise ValueError("Cache fields are incomplete or incompatible")
        if json.loads(data["metadata"].item()) != metadata:
            raise ValueError("Cache metadata/source identity changed")
        saved_ids, names, shapes, offsets = (data[k] for k in ("ids", "filenames", "shapes", "offsets"))
        kp, desc = data["keypoints"], data["descriptors"]
        if saved_ids.dtype != np.int64 or saved_ids.tolist() != ids or names.tolist() != filenames:
            raise ValueError("Cache image IDs/order do not cover the complete database")
        if shapes.dtype != np.int64 or shapes.shape != (len(ids), 2) or np.any(shapes < 1):
            raise ValueError("Invalid image shapes in cache")
        if (offsets.dtype != np.int64 or offsets.shape != (len(ids)+1,) or offsets[0] != 0
                or np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(kp) or len(desc) != len(kp)):
            raise ValueError("Invalid feature offsets/row alignment in cache")
        features = [LocalFeatures(metadata["method"], metadata["params"], tuple(shape),
                                  kp[start:end].copy(), desc[start:end].copy())
                    for shape, start, end in zip(shapes, offsets[:-1], offsets[1:])]
    return features


def _write_cache(path, metadata, ids, filenames, features):
    arrays = dict(metadata=np.asarray(canonical_json(metadata)), ids=np.asarray(ids, np.int64),
                  filenames=np.asarray(filenames), shapes=np.asarray([f.image_shape for f in features], np.int64),
                  offsets=np.asarray([0, *np.cumsum([len(f.keypoints) for f in features])], np.int64),
                  keypoints=np.concatenate([f.keypoints for f in features]),
                  descriptors=np.concatenate([f.descriptors for f in features]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)  # Same-volume atomic replacement; readers never see a partial NPZ.
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_or_compute_database_features(db_dir, method="sift", cache_dir=DEFAULT_CACHE_DIR, **params):
    """Load/extract ALL JPEGs. No subset argument: debug subsets cannot poison caches."""
    config = resolve_local_config(method, params)
    paths, ids = list_images(db_dir)
    if not paths:
        raise ValueError(f"No JPEG database images found in {db_dir}")
    filenames = [p.name for p in paths]
    metadata = dict(schema_version=SCHEMA_VERSION, opencv_version=cv2.__version__,
                    source_identity=source_identity(db_dir), **config)
    path = local_feature_cache_path(db_dir, method, cache_dir, **config["params"])
    status, reason = "miss", "cache file absent"
    if path.exists():
        try:
            features = _read_cache(path, metadata, ids, filenames)
            return DatabaseFeatures(ids, filenames, features, path, "hit", "validated complete database")
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile, zlib.error) as error:
            status, reason = "recomputed", str(error)
    features = [extract_local_features(load_image(p), method, **config["params"])
                for p in tqdm(paths, desc=f"BBDD/{method}")]
    if source_identity(db_dir) != metadata["source_identity"]:
        raise ValueError("Database changed during feature extraction; cache was not written")
    _write_cache(path, metadata, ids, filenames, features)
    return DatabaseFeatures(ids, filenames, features, path, status, reason)
