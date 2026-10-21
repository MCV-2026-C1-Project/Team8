"""Noise removal for uint8 grayscale or three-channel colour images.

Filters include: Average filter, Gaussian filter, Median filter, Ideal 
low-pass filter, Frequency rejection filter

All filters preserve image size, dtype and channel order.
Frequency distances are in DFT bins of the original image. Notch coordinates
are (dy, dx) offsets from the centred spectrum's DC component, not image pixels.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from functools import partial
from itertools import product
import json


def _image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError('Expected a uint8 numpy image (values 0..255)')
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError('Expected grayscale HxW or BGR HxWx3')
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


def log_spectrum(image):
    """Centred log magnitude for peak inspection (float array, not a filter)."""
    _image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))



FILTER_NAMES = (
    "none", "average", "gaussian", "median", "ideal", "rejection"
)


def _make_unconditional_filter(
    name="none",
    *,
    ksize=3,
    sigma=1.0,
    cutoff=30.0,
    notches=(),
    radius=3.0,
):
    """Return an image -> filtered image function with fixed parameters."""
    if name == "none":
        return lambda image: image.copy()

    if name == "average":
        return partial(average_filter, ksize=ksize)

    if name == "gaussian":
        return partial(gaussian_filter, ksize=ksize, sigma=sigma)

    if name == "median":
        return partial(median_filter, ksize=ksize)

    if name == "ideal":
        return partial(ideal_low_pass_filter, cutoff=cutoff)

    if name == "rejection":
        notches = tuple(tuple(pair) for pair in notches)
        if not notches:
            raise ValueError(
                "Frequency rejection requires at least one notch."
            )
        return partial(
            frequency_rejection_filter,
            notches=notches,
            radius=radius,
        )

    raise ValueError(
        f"Unknown filter {name!r}. Choose from {FILTER_NAMES}."
    )

def make_filter(name="none", *, noise_threshold=None, **params):
    """Filter all images, or only those whose noise score exceeds a threshold."""
    base_filter = _make_unconditional_filter(name, **params)

    # JSON null means apply filtering to every image.
    if noise_threshold is None or name == "none":
        return base_filter

    if (
        isinstance(noise_threshold, bool)
        or not np.isfinite(noise_threshold)
        or noise_threshold < 0
    ):
        raise ValueError("noise_threshold must be finite and nonnegative")

    # Lazy import avoids a circular import with noise_detection.py.
    from noise_detection import needs_filtering

    def selective_filter(image):
        result = needs_filtering(image, threshold=noise_threshold)

        if result["needs_filtering"] is True:
            return base_filter(image)

        # Keep clean images and images with an uncertain noise estimate.
        return image.copy()

    return selective_filter

def expand_filter_grid(entries=None):
    """Expand filter settings into individual configurations."""
    allowed = {
        "none": set(),
        "average": {"ksize"},
        "gaussian": {"ksize", "sigma"},
        "median": {"ksize"},
        "ideal": {"cutoff"},
        "rejection": {"notches", "radius"},
    }

    for name in allowed:
        if name != "none":
            allowed[name].add("noise_threshold")

    defaults = {
        "ksize": 3,
        "sigma": 1.0,
        "cutoff": 30.0,
        "radius": 3.0,
    }

    if entries is None:
        entries = [{"name": "none"}]

    if not isinstance(entries, list) or not entries:
        raise ValueError("query_filters must be a nonempty list")

    configurations = {}

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each filter entry must be a dictionary")

        if set(entry) - {"name", "grid"}:
            raise ValueError("Filter entries accept only name and grid")

        name = entry.get("name")
        grid = entry.get("grid", {})

        if name not in allowed:
            raise ValueError(f"Unknown filter: {name!r}")

        if not isinstance(grid, dict):
            raise ValueError("Filter grid must be a dictionary")

        unknown = set(grid) - allowed[name]
        if unknown:
            raise ValueError(
                f"Unsupported parameters for {name}: {sorted(unknown)}"
            )

        for parameter, choices in grid.items():
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"{parameter} requires a nonempty list of choices"
                )

        count = 1
        for choices in grid.values():
            count *= len(choices)

        if len(configurations) + count > 5000:
            raise ValueError("Filter grid exceeds 5000 configurations")

        for values in product(*grid.values()):
            params = {
                key: defaults[key]
                for key in allowed[name]
                if key in defaults
            }
            params.update(zip(grid.keys(), values))

            # Check that the factory accepts this configuration.
            make_filter(name, **params)

            config = {"name": name, **params}
            key = json.dumps(config, sort_keys=True)
            configurations[key] = config

    return list(configurations.values())

def main(argv=None):
    from descriptors import list_images, load_image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JPEG image or folder')
    parser.add_argument('--output', type=Path, required=True, help='Separate output folder')
    parser.add_argument('--filter', choices=['none', 'average', 'gaussian', 'median', 'ideal', 'rejection'], required=True)
    parser.add_argument('--ksize', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--cutoff', type=float, default=30.0)
    parser.add_argument('--notch', type=float, nargs=2, action='append', metavar=('DY', 'DX'), default=[])
    parser.add_argument('--radius', type=float, default=3.0)
    parser.add_argument('--save-spectrum', action='store_true', help='Save input log spectra as .npy arrays')
    args = parser.parse_args(argv)
    try:
        if args.filter == 'rejection' and not args.notch:
            raise ValueError('rejection requires at least one --notch DY DX')
        if args.output.resolve() == (args.input if args.input.is_dir() else args.input.parent).resolve():
            raise ValueError('Use a separate output folder to preserve the original images')
        paths = list_images(args.input)[0] if args.input.is_dir() else [args.input]
        if not paths:
            raise ValueError('No JPEG images found')
        args.output.mkdir(parents=True, exist_ok=True)
        for path in paths:
            image = load_image(path)
            functions = {
                'none': lambda: image.copy(),
                'average': lambda: average_filter(image, args.ksize),
                'gaussian': lambda: gaussian_filter(image, args.ksize, args.sigma),
                'median': lambda: median_filter(image, args.ksize),
                'ideal': lambda: ideal_low_pass_filter(image, args.cutoff),
                'rejection': lambda: frequency_rejection_filter(image, args.notch, args.radius),
            }
            result = functions[args.filter]()
            # PNG avoids adding JPEG compression noise. Retrieval currently reads
            # JPEG only; use these through load_image or a texture-specific loader.
            target = args.output / f'{path.stem}.png'
            if not cv2.imwrite(str(target), result):
                raise OSError(f'Failed to write {target}')
            if args.save_spectrum:
                np.save(args.output / f'{path.stem}_spectrum.npy', log_spectrum(image))
        print(f'Saved {len(paths)} filtered images to {args.output}')
    except (ValueError, OSError, AssertionError, cv2.error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
