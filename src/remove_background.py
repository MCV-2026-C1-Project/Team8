import cv2
import numpy as np


def suppress_sparse_lines(mask, min_row_foreground=0.5, min_col_foreground=0.5):
    """Clear rows/columns with foreground fraction below their cutoff.

    Both decisions use the same input mask, once, without cascading updates.
    Fractions refer to the full image width/height. Set either cutoff to zero
    to disable that axis. This can remove narrow or small genuine paintings.
    """
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Expected a nonempty 2-D mask.")
    for value in (min_row_foreground, min_col_foreground):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Foreground fraction cutoffs must be in [0, 1].")
    foreground = mask > 0
    keep_rows = foreground.mean(axis=1) >= min_row_foreground
    keep_cols = foreground.mean(axis=0) >= min_col_foreground
    return (foreground & keep_rows[:, None] & keep_cols[None, :]).astype(np.uint8) * 255


def fill_painting_interior(mask, closing_size=5, fill_rows=True, fill_columns=True, min_component_ratio=0.001):
    """Fill painting interiors using the colour mask, without contour detection.

    Closing seals small gaps. Optional row filling handles wider leaks: for
    each connected foreground component separately, fill between its leftmost
    and rightmost pixels in each row. This assumes row-solid paintings; it may
    fill real concavities or spaces between paintings joined by a false bridge.
    It never fits a rectangle and does not extend the component's row spans.
    Enclosed holes are then filled by flooding the exterior background.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array.")
    if (not isinstance(closing_size, (int, np.integer)) or closing_size < 0
            or (closing_size > 1 and closing_size % 2 == 0)):
        raise ValueError("closing_size must be 0, 1, or a positive odd integer.")
    result = (mask > 0).astype(np.uint8) * 255
    if closing_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_size, closing_size)
        )
        # Padding avoids reflecting foreground beyond the image boundary.
        pad = closing_size
        work = np.pad(result, pad, constant_values=0)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
        result = work[pad:-pad, pad:-pad]
    if not 0 <= min_component_ratio <= 1:
        raise ValueError("min_component_ratio must be in [0, 1].")
    # Remove small marks before filling, so hollow labels are not expanded.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_component_ratio * result.size
    keep[0] = False
    result = keep[labels].astype(np.uint8) * 255
    if fill_columns:
        # Reuse the same per-component span operation on the transposed mask.
        result = fill_painting_interior(
            result.T, closing_size=0, fill_rows=True, fill_columns=False,
            min_component_ratio=0
        ).T.copy()
    if fill_rows:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
        for label in range(1, count):
            x, y, width, height, _ = stats[label]
            component = labels[y:y + height, x:x + width] == label
            present = component.any(axis=1)
            left = component.argmax(axis=1)
            right = width - 1 - component[:, ::-1].argmax(axis=1)
            columns = np.arange(width)[None, :]
            interior = (present[:, None] & (columns >= left[:, None])
                        & (columns <= right[:, None]))
            roi = result[y:y + height, x:x + width]
            roi[interior] = 255
    # Fill only background regions that cannot reach the padded exterior.
    padded = np.pad(result, 1, constant_values=0)
    exterior = padded.copy()
    cv2.floodFill(exterior, None, (0, 0), 255)
    return cv2.bitwise_or(padded, cv2.bitwise_not(exterior))[1:-1, 1:-1]


def foreground_mask(
    image,
    border_ratio=0.04,
    threshold=20.0,
    mad_multiplier=3.0,
    sample_step=4,
    l_weight=0.5,
    fill_interior=True,
    fill_rows=True,
    closing_size=5,
    fill_columns=True,
    min_component_ratio=0.001,
    method="color_distribution",
    color_sigma=5.5,
    min_row_foreground=0.5,
    min_col_foreground=0.5,
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
    min_row_foreground, min_col_foreground : float
        Final full-image row/column foreground fraction cutoffs. Below-cutoff
        lines become background; zero disables an axis. Defaults are 0.5.
    method : str
        color_distribution uses brightness-conditioned wall colours;
        quadratic retains the spatial baseline.
    color_sigma : float
        Mahalanobis tolerance for color_distribution (default 3).
    fill_columns : bool
        Also fill vertical spans within each connected foreground component.
    min_component_ratio : float
        Minimum component area / image area before interior filling.
    l_weight : float
        Weight of lightness (L) in the Lab distance. Values below 1 make
        illumination changes on the wall matter less than changes in colour.

    fill_interior : bool
        Refine the colour mask with closing and interior filling.
    fill_rows : bool
        Fill each component's horizontal spans to repair open interior leaks.
        Disable to use enclosed-hole filling only.
    closing_size : int
        Odd closing kernel size; 0 disables closing.

    Returns
    -------
    mask : np.ndarray
        Binary uint8 mask: 255 = foreground, 0 = background.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if method not in {"quadratic", "color_distribution"}:
        raise ValueError("Unknown method.")
    if method == "color_distribution":
        mask = _color_distribution_mask(image, border_ratio, sample_step, color_sigma)
        if fill_interior:
            mask = fill_painting_interior(mask, closing_size, fill_rows,
                                         fill_columns, min_component_ratio)
        return suppress_sparse_lines(mask, min_row_foreground, min_col_foreground)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= l_weight
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

    # Compute each pixel's Lab distance from the estimated wall color.
    distance = np.linalg.norm(lab - background, axis=2)

    # Candidate background pixels are sufficiently similar to the wall.
    candidate_background = (
        distance <= adaptive_threshold
    ).astype(np.uint8)

    # Label connected regions of candidate background pixels.
    num_labels, labels = cv2.connectedComponents(
        candidate_background,
        connectivity=8,
    )

    # Only background regions connected to the image border
    # are considered actual wall/background.
    border_labels = np.unique(np.concatenate([
        labels[0, :],
        labels[-1, :],
        labels[:, 0],
        labels[:, -1],
    ]))

    background_mask = np.isin(labels, border_labels)
    background_mask &= candidate_background.astype(bool)

    # Everything else is foreground, including enclosed regions
    # whose colors resemble the wall.
    mask = (~background_mask).astype(np.uint8) * 255

    if fill_interior:
        mask = fill_painting_interior(mask, closing_size, fill_rows,
                                     fill_columns, min_component_ratio)
    return suppress_sparse_lines(mask, min_row_foreground, min_col_foreground)



def remove_background(image, **kwargs):
    """Return the foreground image and its binary mask."""
    mask = foreground_mask(image, **kwargs)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    return foreground, mask


def _distribution_model(
    image,
    border_ratio=0.04,
    threshold=15.0,
    mad_multiplier=3.0,
    sample_step=4,
    lightness_weight=0.5,
    opening_size=3,
    min_foreground_ratio=0.0,
    method="color_distribution",
    chroma_threshold=3.0,
    fill_holes=False,
    color_sigma=3.0,
    closing_size=5,
):
    """
    Segment using only a background colour model and colour-mask connectivity.

    No gradients, contours, object detectors or rectangle/shape fitting are used.
    Border connectivity and optional morphology operate on the colour mask only.
    closing_size controls foreground closing (odd pixels; 0 disables it).
    color_distribution uses brightness-conditioned a,b means and robust residual
    covariance; color_sigma is its Mahalanobis-distance threshold (default 3).
    The older chromaticity and quadratic methods remain available.

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

    lightness_weight : float
        Weight of squared L differences (0 to 1); lower tolerates shadows.
    opening_size : int
        Odd background-opening kernel size in pixels; 0 or 1 disables it.
    min_foreground_ratio : float
        Remove foreground components smaller than this fraction of image area.
        Set to 0 to disable; substantial separate paintings are retained.

    method : str
        "chromaticity" fits wall a,b as functions of lightness, allowing local
        spotlights and shadows. "quadratic" retains the previous spatial model.
    chroma_threshold : float
        Minimum a,b residual tolerance in OpenCV's uint8 Lab units. Used only
        by chromaticity; threshold/lightness_weight affect quadratic only.
    fill_holes : bool
        Fill enclosed background holes, matching masks that include panel gaps.

    Returns
    -------
    mask : np.ndarray
        Binary uint8 mask: 255 = foreground, 0 = background.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if not 0 < border_ratio <= 0.5:
        raise ValueError("border_ratio must be in (0, 0.5].")
    if not isinstance(sample_step, (int, np.integer)) or sample_step < 1:
        raise ValueError("sample_step must be a positive integer.")
    if threshold < 0 or mad_multiplier < 0:
        raise ValueError("Threshold parameters must be nonnegative.")
    if not 0 <= lightness_weight <= 1:
        raise ValueError("lightness_weight must be in [0, 1].")
    if opening_size < 0 or (opening_size > 1 and opening_size % 2 == 0):
        raise ValueError("opening_size must be 0, 1, or a positive odd integer.")
    if (not isinstance(closing_size, (int, np.integer)) or closing_size < 0
            or (closing_size > 1 and closing_size % 2 == 0)):
        raise ValueError("closing_size must be 0, 1, or a positive odd integer.")
    if not 0 <= min_foreground_ratio <= 1:
        raise ValueError("min_foreground_ratio must be in [0, 1].")

    if method not in {"quadratic", "chromaticity", "color_distribution"}:
        raise ValueError("Unknown background method.")
    if not np.isfinite(color_sigma) or color_sigma <= 0:
        raise ValueError("color_sigma must be positive and finite.")
    if chroma_threshold <= 0:
        raise ValueError("chroma_threshold must be positive.")

    # Use the same weighted metric for fitting residuals and classification.
    channel_scale = np.array([np.sqrt(lightness_weight), 1.0, 1.0])
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

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
    colors = lab.reshape(-1, 3)[indices]

    if method in {"chromaticity", "color_distribution"}:
        # Learn how the wall's chromatic channels change with brightness.
        # No spatial extrapolation of a spotlight into the painting is needed.
        lightness = colors[:, 0] / 255.0
        design = np.column_stack([np.ones_like(lightness), lightness, lightness**2])
        target = colors[:, 1:]
        weights = np.ones(len(colors))
        for _ in range(8):
            sw = np.sqrt(weights)[:, None]
            coefficients = np.linalg.lstsq(design * sw, target * sw, rcond=None)[0]
            residuals = np.linalg.norm(target - design @ coefficients, axis=1)
            med = np.median(residuals)
            scale = max(0.5, 1.4826 * np.median(np.abs(residuals - med)))
            weights = np.minimum(1.0, 1.5 * scale / np.maximum(residuals, 1e-6))
        # Limit extrapolation beyond the sampled brightness range.
        L = np.clip(lab[:, :, 0] / 255.0, lightness.min(), lightness.max())
        prediction = (coefficients[0] + L[:, :, None] * coefficients[1]
                      + L[:, :, None]**2 * coefficients[2])
        distance = np.linalg.norm(lab[:, :, 1:] - prediction, axis=2)
        # Cap adaptive expansion to avoid border contamination opening the mask.
        adaptive_threshold = max(chroma_threshold,
                                 min(2 * chroma_threshold, med + mad_multiplier * scale))
        if method == "color_distribution":
            # Model anisotropic colour noise: the wall may vary more along
            # one chromatic direction than another. Robust weights suppress
            # border outliers such as labels or small pieces of artwork.
            errors = target - design @ coefficients
            center = np.median(errors, axis=0)
            centered = errors - center
            covariance = ((centered * weights[:, None]).T @ centered
                          / max(weights.sum(), 1.0))
            covariance += np.eye(2) * 1.0  # floor for quantisation / flat walls
            inverse = np.linalg.inv(covariance)
            delta = lab[:, :, 1:] - prediction - center
            distance = np.sqrt(np.maximum(0, np.einsum(
                "...i,ij,...j->...", delta, inverse, delta
            )))
            adaptive_threshold = color_sigma
            # Ignoring brightness entirely confuses dark neutral paintings
            # with pale neutral walls. Allow shadows, but reject extreme darks.
            lower = max(0.0, np.percentile(colors[:, 0], 2) - 30.0)
            distance[lab[:, :, 0] < lower] = np.inf
    else:
        # Only the legacy quadratic model needs a spatial design matrix.
        yy, xx = np.mgrid[0:h, 0:w]
        x = 2.0 * xx / max(w - 1, 1) - 1.0
        y = 2.0 * yy / max(h - 1, 1) - 1.0
        X = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=-1).reshape(-1, 6)
        A = X[indices]
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
                (colors - A @ coefficients) * channel_scale, axis=1
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
            (colors - A @ coefficients) * channel_scale, axis=1
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

        # Compute each pixel's Lab distance from the estimated wall color.
        distance = np.linalg.norm((lab - background) * channel_scale, axis=2)

    # Candidate background pixels are sufficiently similar to the wall.
    candidate_background = (
        distance <= adaptive_threshold
    ).astype(np.uint8)

    # Break narrow background paths that could leak into painting interiors.
    # Opening may also remove thin genuine wall gaps; disable for comparison.
    if opening_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (opening_size, opening_size)
        )
        candidate_background = cv2.morphologyEx(
            candidate_background, cv2.MORPH_OPEN, kernel
        )

    # Label connected regions of candidate background pixels.
    num_labels, labels = cv2.connectedComponents(
        candidate_background,
        connectivity=8,
    )

    # Only background regions connected to the image border
    # are considered actual wall/background.
    border_labels = np.unique(np.concatenate([
        labels[0, :],
        labels[-1, :],
        labels[:, 0],
        labels[:, -1],
    ]))

    background_mask = np.isin(labels, border_labels)
    background_mask &= candidate_background.astype(bool)

    # Everything else is foreground, including enclosed regions
    # whose colors resemble the wall.
    mask = (~background_mask).astype(np.uint8) * 255

    # Close narrow gaps / small holes in the colour-derived foreground mask.
    # This assumes no rectangular shape and uses no image edges or contours.
    if closing_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_size, closing_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove small isolated foreground marks without assuming one painting.
    if min_foreground_ratio > 0:
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        keep = stats[:, cv2.CC_STAT_AREA] >= min_foreground_ratio * h * w
        keep[0] = False  # label 0 is background
        mask = keep[component_labels].astype(np.uint8) * 255

    if fill_holes:
        # Padding guarantees an exterior seed even when foreground touches edges.
        padded = np.pad(mask, 1, constant_values=0)
        exterior = padded.copy()
        cv2.floodFill(exterior, None, (0, 0), 255)
        mask = cv2.bitwise_or(padded, cv2.bitwise_not(exterior))[1:-1, 1:-1]

    return mask


def _color_distribution_mask(image, border_ratio, sample_step, color_sigma):
    return _distribution_model(
        image, method="color_distribution", border_ratio=border_ratio,
        sample_step=sample_step, color_sigma=color_sigma, opening_size=0,
        closing_size=0, min_foreground_ratio=0, fill_holes=False
    )
