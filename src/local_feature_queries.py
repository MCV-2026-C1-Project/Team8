"""Reuse Week 3 detection and crop inset, with explicit crop/query coordinates."""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from local_features import LocalFeatures, extract_local_features, validate_image
from painting_detection import detect_paintings, resolve_detection_config
from retrieval import inset_box
from pathlib import Path
import cv2


@dataclass
class PaintingFeatures:
    query_id: int
    painting_index: int
    image_shape: tuple
    detection_bbox: tuple
    crop_bbox: tuple
    features: LocalFeatures

    @property
    def query_points(self):
        """Coordinates in the original query; descriptors remain crop-relative."""
        return self.features.points + np.asarray(self.crop_bbox[:2], dtype=np.float32)


@dataclass
class PaintingCrop:
    query_id: int
    painting_index: int
    image_shape: tuple
    detection_bbox: tuple
    crop_bbox: tuple
    image: np.ndarray

    def extract(self, method="sift", **params):
        return PaintingFeatures(self.query_id, self.painting_index, self.image_shape,
                                self.detection_bbox, self.crop_bbox,
                                extract_local_features(self.image, method, **params))


def prepare_painting_crops(image_bgr, query_id, segmentation_config=None, *, crop_inset=0.03, preprocess=None):
    """Detect once; original rectangular pixels, in detector order; no GT or cache.

    preprocess can be the existing query_preprocessor(config) callable. It runs
    after cropping and must retain shape/dtype, so coordinate offsets stay valid.
    Returns [] for no detections. Reuse the returned crops across feature methods.
    """
    validate_image(image_bgr)
    if isinstance(query_id, (bool, np.bool_)) or not isinstance(query_id, Integral) or query_id < 0:
        raise ValueError("query_id must be a nonnegative integer")
    if (isinstance(crop_inset, (bool, np.bool_)) or not isinstance(crop_inset, Real)
            or not np.isfinite(crop_inset) or not 0 <= crop_inset < .5):
        raise ValueError("crop_inset must lie in [0, 0.5)")
    if preprocess is not None and not callable(preprocess):
        raise ValueError("preprocess must be callable or None")
    _, paintings = detect_paintings(image_bgr, resolve_detection_config(segmentation_config))

    # ### debug 
    # mask, paintings = detect_paintings(
    # image_bgr, resolve_detection_config(segmentation_config))

    # # Save the detector's final mask, even when no paintings are detected.
    # debug_dir = Path("results/week4/debug_masks")
    # debug_dir.mkdir(parents=True, exist_ok=True)

    # cv2.imwrite(str(debug_dir / f"{query_id:05d}_mask.png"), mask)

    # # Highlight foreground in green over the original image.
    # overlay = image_bgr.copy()
    # overlay[mask > 0] = (0, 255, 0)
    # overlay = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)

    # cv2.imwrite(str(debug_dir / f"{query_id:05d}_overlay.png"), overlay)

    # print(f"Query {query_id:05d}: {len(paintings)} paintings detected")
    # ### debug
    
    crops = []
    h, w = image_bgr.shape[:2]
    for index, painting in enumerate(paintings):
        x, y, right, bottom = painting.bbox
        if painting.order != index or not (0 <= x < right <= w and 0 <= y < bottom <= h):
            raise ValueError("Painting detector returned invalid order or bounds")
        box = inset_box(painting.bbox, crop_inset)
        x, y, right, bottom = box
        if x >= right or y >= bottom:
            raise ValueError(f"Inset produces an empty crop for query {query_id}, painting {index}")
        # A copy lets a caller's optional preprocessing operate without altering the original.
        crop = image_bgr[y:bottom, x:right].copy()
        if preprocess is not None:
            crop = preprocess(crop)
            validate_image(crop)
            if crop.shape[:2] != (bottom-y, right-x):
                raise ValueError("Query preprocessing must preserve crop dimensions")
        crops.append(PaintingCrop(int(query_id), index, (h, w), tuple(painting.bbox), tuple(box), crop))
    return crops
