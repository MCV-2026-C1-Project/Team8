"""Optional whole-image photometric normalisation for Lab/YCrCb histograms."""
import cv2
import numpy as np


def color_image(img_bgr, method, normalization="none", mask=None):
    """Convert BGR to the histogram colour space, optionally centring/scaling it.

    The legacy path retains OpenCV's uint8 conversion exactly. Photometric mode
    uses float conversion, each channel's own mean and a shared luminance scale.
    Statistics use the complete image's foreground when a mask is supplied.
    """
    codes = {"lab": cv2.COLOR_BGR2Lab, "ycbcr": cv2.COLOR_BGR2YCrCb}
    if method not in codes:
        raise ValueError("Photometric conversion supports only lab and ycbcr")
    if normalization not in ("none", "photometric"):
        raise ValueError("normalization must be 'none' or 'photometric'")
    if normalization == "none":
        return cv2.cvtColor(img_bgr, codes[method])
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != img_bgr.shape[:2]:
            raise ValueError("Mask and image sizes differ")
        mask = (mask > 0).astype(np.uint8) * 255
        if not np.any(mask):
            raise ValueError("Foreground mask is empty; cannot compute a painting descriptor")

    image = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, codes[method])
    if method == "lab":
        image = image * np.float32([255 / 100, 1, 1]) + np.float32([0, 128, 128])
    else:
        image *= 255.0
    statistics = cv2.meanStdDev(image) if mask is None else cv2.meanStdDev(image, mask=mask)
    mean, std = (part.ravel() for part in statistics)
    scale = np.float32(50.0 / max(std[0], 1e-6))
    image = (image - mean.astype(np.float32)) * scale + np.float32(128.0)
    return np.clip(image, 0, 255.99).astype(np.float32, copy=False)
