"""Heuristic noise scoring for uint8 colour/grayscale images.
Identify when an image requires noise filtering.

No filtering or file loading is performed here. Tune the decision threshold on
 development queries; this score does not identify the noise distribution.
"""
import cv2
import numpy as np

from filter_noise import _image


def compute_noise_score(image, *, smooth_sigma=2.0, max_gradient=3.0,
                        min_smooth_fraction=0.05, min_smooth_pixels=100):
    """Measure high-frequency residuals away from visible edges.

    Input: uint8 HxW or HxWx3, either RGB or BGR. Channels are processed
    independently; their maximum score detects noise confined to one channel.
    No resizing is performed. Scores use the input's 0..255 intensity scale.

    Smooth regions have small gradients in EVERY Gaussian-smoothed channel.
    max_gradient is in intensity units per pixel (normalised Sobel derivative).
    The mask is eroded so residual samples avoid nearby edges.

    Returns a dictionary with score, per-channel scores, smooth_fraction,
    smooth_pixels and reliable. An insufficient smooth area gives score=None.
    'reliable' means sufficient samples, not guaranteed noise identification.
    Fine painting texture, JPEG artefacts and periodic patterns can confound
    this heuristic. Sparse impulse noise may not increase its robust MAD score.
    """
    _image(image)
    if not np.isfinite(smooth_sigma) or smooth_sigma <= 0:
        raise ValueError('smooth_sigma must be finite and positive')
    if not np.isfinite(max_gradient) or max_gradient < 0:
        raise ValueError('max_gradient must be finite and nonnegative')
    if not np.isfinite(min_smooth_fraction) or not 0 < min_smooth_fraction <= 1:
        raise ValueError('min_smooth_fraction must be in (0, 1]')
    if not isinstance(min_smooth_pixels, (int, np.integer)) or min_smooth_pixels < 1:
        raise ValueError('min_smooth_pixels must be a positive integer')
    if min(image.shape[:2]) < 7:
        raise ValueError('Images must be at least 7x7 pixels')

    channels = image[..., None] if image.ndim == 2 else image
    channels = channels.astype(np.float32)
    mask = np.ones(image.shape[:2], dtype=bool)
    residuals = []
    kernel = cv2.getGaussianKernel(3, 1.0)
    highpass = -(kernel @ kernel.T)
    highpass[1, 1] += 1.0
    # Under independent additive Gaussian noise, a linear residual changes its
    # standard deviation by ||highpass||_2. Correct for that known gain.
    gain = float(np.sqrt(np.sum(highpass ** 2)))
    for channel in np.moveaxis(channels, -1, 0):
        smooth = cv2.GaussianBlur(channel, (0, 0), smooth_sigma)
        dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
        dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
        mask &= np.hypot(dx, dy) <= max_gradient
        residuals.append(channel - cv2.GaussianBlur(channel, (3, 3), 1.0))
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    mask[:3] = mask[-3:] = False
    mask[:, :3] = mask[:, -3:] = False
    count = int(mask.sum())
    fraction = count / mask.size
    reliable = count >= min_smooth_pixels and fraction >= min_smooth_fraction
    scores = []
    if reliable:
        for residual in residuals:
            samples = residual[mask]
            mad = np.median(np.abs(samples - np.median(samples)))
            scores.append(float(1.4826 * mad / gain))
    return dict(score=max(scores) if scores else None, channel_scores=scores,
                smooth_fraction=float(fraction), smooth_pixels=count,
                reliable=bool(reliable))


def needs_filtering(image, *, threshold, **score_options):
    """Return score metadata plus needs_filtering: True, False or None.

    threshold is REQUIRED: there is no universal calibrated cutoff.
    None means insufficient smooth pixels; inspect that image manually.
    True means score > threshold, not proof filtering improves retrieval.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and nonnegative')
    result = compute_noise_score(image, **score_options)
    result['threshold'] = float(threshold)
    result['needs_filtering'] = (result['score'] > threshold
                                 if result['reliable'] else None)
    return result
