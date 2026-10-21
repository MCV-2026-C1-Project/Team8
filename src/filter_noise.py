"""Noise removal for uint8 grayscale or three-channel colour images.

Filters include: Average filter, Gaussian filter, Median filter, Ideal 
low-pass filter, Frequency rejection filter

All filters preserve image size, dtype and channel order.
Frequency parameters are measured in DFT bins of the original image.
"""
import cv2
import numpy as np


def _image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError('Expected a uint8 numpy image (values 0..255)')
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError('Expected grayscale HxW or colour HxWx3')
    if not image.size:
        raise ValueError('Image must not be empty')
    return image


def _kernel(ksize):
    if not isinstance(ksize, (int, np.integer)) or ksize < 1 or ksize % 2 != 1:
        raise ValueError('ksize must be a positive odd integer')
    return int(ksize)


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def average_filter(image, ksize=3):
    """Uniform local average; reflect borders. Larger windows blur more detail."""
    return cv2.blur(_image(image), (_kernel(ksize), ksize), borderType=cv2.BORDER_REFLECT_101)


def gaussian_filter(image, ksize=5, sigma=1.0):
    """Gaussian weighted average; sigma is measured in spatial pixels."""
    return cv2.GaussianBlur(_image(image), (_kernel(ksize), ksize),
                            _positive(sigma, 'sigma'), borderType=cv2.BORDER_REFLECT_101)


def median_filter(image, ksize=3):
    """Channel-wise median, useful for impulse noise; OpenCV replicates borders."""
    _image(image)
    ksize = _kernel(ksize)
    return image.copy() if ksize == 1 else cv2.medianBlur(image, ksize)


def _frequency_filter(image, transfer):
    """Filter each channel; retain real signed reconstruction before clipping.

    Do not take absolute values or min-max normalise: those change intensities.
    FFT assumes periodic boundaries; edge discontinuities and sharp frequency
    cutoffs can cause ringing.
    """
    image = _image(image)
    spectrum = np.fft.fft2(image.astype(np.float32), axes=(0, 1))
    weights = transfer if image.ndim == 2 else transfer[..., None]
    result = np.fft.ifft2(spectrum * weights, axes=(0, 1)).real
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _frequency_grid(shape):
    h, w = shape[:2]
    return np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing='ij')


def ideal_low_pass_filter(image, cutoff=30.0):
    """Keep a circular radius of cutoff DFT bins around DC; may cause ringing.

    The same cutoff smooths different resolutions differently. Tune at a fixed
    image size, or scale it with image dimensions when comparing resolutions.
    """
    _image(image)
    cutoff = _positive(cutoff, 'cutoff')
    fy, fx = _frequency_grid(image.shape)
    return _frequency_filter(image, np.hypot(fy, fx) <= cutoff)


def frequency_rejection_filter(image, notches, radius=3.0):
    """Gaussian notch rejection for periodic noise, preserving the DC component.

    notches: iterable of (dy, dx) frequency-bin offsets. Supply only one peak
    from each conjugate pair; its negative is rejected automatically. radius
    controls notch width (Gaussian scale in bins). Coordinates wrap on the DFT
    grid. Inspect the spectrum first; arbitrary notches remove painting detail.
    """
    _image(image)
    radius = _positive(radius, 'radius')
    fy, fx = _frequency_grid(image.shape)
    h, w = image.shape[:2]
    transfer = np.ones((h, w), dtype=np.float64)
    for dy, dx in notches:
        if not np.isfinite([dy, dx]).all() or not (-h / 2 <= dy <= h / 2 and -w / 2 <= dx <= w / 2):
            raise ValueError('Notch offsets must be finite and within the Nyquist limits')
        if dy == 0 and dx == 0:
            raise ValueError('Do not reject DC (the mean image intensity)')
        for sign in (1, -1):
            # Wrapped distance also handles peaks near the Nyquist boundary.
            ddy = (fy - sign * dy + h / 2) % h - h / 2
            ddx = (fx - sign * dx + w / 2) % w - w / 2
            transfer *= -np.expm1(-(ddy**2 + ddx**2) / (2 * radius**2))
    transfer[0, 0] = 1.0
    return _frequency_filter(image, transfer)
