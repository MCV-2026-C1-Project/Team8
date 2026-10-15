"""Joint colour histograms; no resizing, equalisation or foreground processing."""
import cv2
import numpy as np


def joint_hist(image, channels, bins, ranges, mask=None):
    """Unnormalised multidimensional counts, with channels in the requested order."""
    return cv2.calcHist([image], list(channels), mask, list(bins),
                        [float(x) for interval in ranges for x in interval])


def distribution(counts):
    counts = counts.ravel().astype(np.float32)
    mass = counts.sum()
    if mass <= 0:
        raise ValueError("Cannot normalise an empty colour distribution")
    return counts / mass


def combine(chroma, image, channel, bins, include, weight):
    if not include:
        return chroma
    light = distribution(joint_hist(image, [channel], [bins], [(0, 256)]))
    return np.concatenate([weight * chroma, (1 - weight) * light])


def hsv_hs2d(img_bgr, bins=(16, 16), s_min=10, v_min=40,
             filter_hue=True, achromatic_bin=True, hue_smoothing=False,
             include_v=True, v_bins=8, chroma_weight=0.7):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    valid = ((hsv[:, :, 1] > s_min) & (hsv[:, :, 2] > v_min)
             if filter_hue else np.ones(hsv.shape[:2], dtype=bool))
    hs = joint_hist(hsv, [0, 1], bins, [(0, 180), (0, 256)], valid.astype(np.uint8))
    if hue_smoothing:
        hs = 0.25 * np.roll(hs, 1, axis=0) + 0.5 * hs + 0.25 * np.roll(hs, -1, axis=0)
    if achromatic_bin:
        chroma = distribution(np.append(hs.ravel(), valid.size - np.count_nonzero(valid)))
    elif hs.sum() > 0:
        chroma = distribution(hs)
    else:
        # Explicit ablation policy: no valid Hue and no achromatic bin means an
        # uninformative uniform HS distribution, preserving component mass.
        chroma = np.full(hs.size, 1.0 / hs.size, dtype=np.float32)
    return combine(chroma, hsv, 2, v_bins, include_v, chroma_weight)


def lab_ab2d(img_bgr, bins=(16, 16), include_l=True, l_bins=8, chroma_weight=0.7):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    chroma = distribution(joint_hist(lab, [1, 2], bins, [(0, 256)] * 2))
    return combine(chroma, lab, 0, l_bins, include_l, chroma_weight)


def hsv_3d(img_bgr, bins=(8, 8, 4)):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return distribution(joint_hist(hsv, [0, 1, 2], bins, [(0, 180), (0, 256), (0, 256)]))


def lab_3d(img_bgr, bins=(4, 8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return distribution(joint_hist(lab, [0, 1, 2], bins, [(0, 256)] * 3))
