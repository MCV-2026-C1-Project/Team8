import cv2
import numpy as np


def foreground_mask(
    image,
    border_ratio=0.08,
    threshold=15.0,
    mad_multiplier=3.0,
    sample_step=4,
):
    """
    Estimate a smoothly varying wall color from image borders.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.
    border_ratio : float
        Width of the border region used to estimate the background.
    threshold : float
        Minimum Lab color distance required for foreground.
    mad_multiplier : float
        Controls the adaptive threshold based on border residuals.
    sample_step : int
        Spacing between sampled border pixels.

    Returns
    -------
    mask : np.ndarray
        Binary uint8 mask: 255 = foreground, 0 = background.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    # Normalize pixel coordinates.
    yy, xx = np.mgrid[0:h, 0:w]
    x = 2.0 * xx / max(w - 1, 1) - 1.0
    y = 2.0 * yy / max(h - 1, 1) - 1.0

    # Quadratic surface models gradual color/illumination changes.
    X = np.stack(
        [np.ones_like(x), x, y, x*x, x*y, y*y], axis=-1
    ).reshape(-1, 6)

    # Define the four border regions.
    bh = max(2, int(h * border_ratio))
    bw = max(2, int(w * border_ratio))

    border = np.zeros((h, w), dtype=bool)
    border[:bh, :] = True
    border[-bh:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True

    # Subsample the border while retaining its spatial distribution.
    sampled = np.zeros((h, w), dtype=bool)
    sampled[::sample_step, ::sample_step] = True
    sampled &= border

    indices = np.flatnonzero(sampled)
    A = X[indices]
    colors = lab.reshape(-1, 3)[indices]

    # Robust weighted least-squares fitting.
    weights = np.ones(len(colors), dtype=np.float64)

    for _ in range(8):
        sqrt_weights = np.sqrt(weights)[:, None]

        coefficients = np.linalg.lstsq(
            A * sqrt_weights,
            colors * sqrt_weights,
            rcond=None,
        )[0]

        residuals = np.linalg.norm(
            colors - A @ coefficients, axis=1
        )

        median = np.median(residuals)
        mad = 1.4826 * np.median(
            np.abs(residuals - median)
        )

        # Downweight pixels that disagree with the background model.
        cutoff = max(1.0, 1.5 * mad)
        weights = np.minimum(
            1.0, cutoff / np.maximum(residuals, 1e-6)
        )

    # Estimate background-model residual variation.
    residuals = np.linalg.norm(
        colors - A @ coefficients, axis=1
    )
    median = np.median(residuals)
    mad = 1.4826 * np.median(
        np.abs(residuals - median)
    )

    adaptive_threshold = max(
        threshold, median + mad_multiplier * mad
    )

    # Predict the background color at every pixel.
    background = (X @ coefficients).reshape(h, w, 3)

    # Pixels sufficiently different from the predicted wall are foreground.
    distance = np.linalg.norm(lab - background, axis=2)
    mask = (distance > adaptive_threshold).astype(np.uint8) * 255

    return mask


def remove_background(image, **kwargs):
    """Return the foreground image and its binary mask."""
    mask = foreground_mask(image, **kwargs)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    return foreground, mask
