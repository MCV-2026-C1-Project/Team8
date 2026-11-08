"""Week 4 local descriptors at native input resolution; no matching or ranking."""
from dataclasses import dataclass
from numbers import Integral, Real

import cv2
import numpy as np


METHODS = ("sift", "orb", "akaze")
DEFAULTS = {
    "sift": dict(nfeatures=1000, nOctaveLayers=3, contrastThreshold=0.04,
                 edgeThreshold=10.0, sigma=1.6),
    "orb": dict(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                patchSize=31, fastThreshold=20),
    "akaze": dict(threshold=0.001, nOctaves=4, nOctaveLayers=4),
}
KEYPOINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("size", "<f4"), ("angle", "<f4"),
    ("response", "<f4"), ("octave", "<i4"), ("class_id", "<i4"),
])


def resolve_local_config(method="sift", params=None):
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"Unknown local feature method {method!r}; choose from {METHODS}")
    params = {} if params is None else params
    if not isinstance(params, dict) or set(params) - set(DEFAULTS[method]):
        raise ValueError(f"Supported {method} parameters: {', '.join(DEFAULTS[method])}")
    effective = dict(DEFAULTS[method], **params)
    for name, value in effective.items():
        integer = isinstance(DEFAULTS[method][name], int)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral if integer else Real):
            raise ValueError(f"{method}.{name} must be {'an integer' if integer else 'a number'}")
        minimum = 0 if name in {"nfeatures", "fastThreshold", "edgeThreshold"} else 1 if integer else 0
        if not np.isfinite(value) or value < minimum:
            raise ValueError(f"{method}.{name} must be finite and >= {minimum}")
        if not integer and value <= 0:
            raise ValueError(f"{method}.{name} must be positive")
        if integer and value > np.iinfo(np.int32).max:
            raise ValueError(f"{method}.{name} exceeds OpenCV's integer range")
        effective[name] = int(value) if integer else float(value)
    if method == "orb" and (effective["nfeatures"] < 1 or effective["scaleFactor"] <= 1 or effective["patchSize"] < 2):
        raise ValueError("ORB needs nfeatures >= 1, scaleFactor > 1 and patchSize >= 2")
    return dict(method=method, params=effective)


def feature_spec(method):
    """Descriptor width in elements (bytes for binary descriptors), dtype, norm."""
    resolve_local_config(method)
    return {"sift": (128, np.dtype("float32"), cv2.NORM_L2),
            "orb": (32, np.dtype("uint8"), cv2.NORM_HAMMING),
            "akaze": (61, np.dtype("uint8"), cv2.NORM_HAMMING)}[method]


def create_detector(method="sift", **params):
    config = resolve_local_config(method, params)
    constructor = getattr(cv2, {"sift": "SIFT_create", "orb": "ORB_create", "akaze": "AKAZE_create"}[method], None)
    if constructor is None:
        raise ValueError(f"OpenCV {cv2.__version__} lacks {method.upper()}_create; "
                         "use the isolated environment from requirements-week4.txt")
    fixed = {}
    if method == "orb":
        fixed = dict(WTA_K=2, firstLevel=0, scoreType=cv2.ORB_HARRIS_SCORE)
    elif method == "akaze":
        fixed = dict(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0,
                     descriptor_channels=3, diffusivity=cv2.KAZE_DIFF_PM_G2)
    return constructor(**config["params"], **fixed)


def validate_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not image.size):
        raise ValueError("Expected a nonempty HxWx3 uint8 BGR image")


@dataclass
class LocalFeatures:
    """Row i describes keypoint i. Coordinates/size are in input-image pixels.

    Structured keypoints preserve integer octave/class_id without float coercion.
    No resizing, feature normalisation, sorting or descriptor conversion is done.
    """
    method: str
    params: dict
    image_shape: tuple
    keypoints: np.ndarray
    descriptors: np.ndarray

    def __post_init__(self):
        self.params = resolve_local_config(self.method, self.params)["params"]
        width, dtype, _ = feature_spec(self.method)
        if (len(self.image_shape) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1
                                            for v in self.image_shape)):
            raise ValueError("image_shape must be positive (height, width)")
        self.image_shape = tuple(int(v) for v in self.image_shape)
        kp, desc = self.keypoints, self.descriptors
        if not isinstance(kp, np.ndarray) or kp.ndim != 1 or kp.dtype != KEYPOINT_DTYPE:
            raise ValueError("Invalid structured keypoint array")
        if (not isinstance(desc, np.ndarray) or desc.dtype != dtype or desc.shape != (len(kp), width)
                or not np.isfinite(desc).all()):
            raise ValueError("Descriptor dimensions, dtype or row alignment are invalid")
        for name in ("x", "y", "size", "angle", "response"):
            if not np.isfinite(kp[name]).all():
                raise ValueError("Keypoints must be finite")
        h, w = self.image_shape
        if (np.any(kp["x"] < 0) or np.any(kp["x"] >= w) or np.any(kp["y"] < 0) or np.any(kp["y"] >= h)
                or np.any(kp["size"] <= 0) or np.any(kp["angle"] < -1) or np.any(kp["angle"] >= 360)):
            raise ValueError("Keypoint coordinates, sizes or angles are invalid")

    @property
    def points(self):
        return np.column_stack((self.keypoints["x"], self.keypoints["y"]))

    @property
    def norm(self):
        return feature_spec(self.method)[2]

    @property
    def descriptor_type(self):
        return "float" if self.method == "sift" else "binary"

    def to_cv_keypoints(self):
        return [cv2.KeyPoint(float(k["x"]), float(k["y"]), float(k["size"]), float(k["angle"]),
                             float(k["response"]), int(k["octave"]), int(k["class_id"])) for k in self.keypoints]


def extract_local_features(image_bgr, method="sift", **params):
    validate_image(image_bgr)
    detector = create_detector(method, **params)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # A one-pixel-wide image has no 2D neighbourhood; ORB/AKAZE cannot build its pyramid.
    keypoints, descriptors = ((), None) if min(gray.shape) < 2 else detector.detectAndCompute(gray, None)
    keypoints = () if keypoints is None else keypoints
    array = np.array([(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
                      for k in keypoints], dtype=KEYPOINT_DTYPE)
    if descriptors is None and not keypoints:
        width, dtype, _ = feature_spec(method)
        descriptors = np.empty((0, width), dtype=dtype)
    return LocalFeatures(method, params, image_bgr.shape[:2], array, descriptors)
