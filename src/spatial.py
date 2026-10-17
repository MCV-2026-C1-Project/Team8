"""Relative image partitions and mass-preserving spatial concatenation."""
import numpy as np


def resolve_spatial(config=None):
    config = {} if config is None else dict(config)
    if set(config) - {"mode", "grids", "level_weights"}:
        raise ValueError("Spatial settings accept only mode, grids and level_weights")
    mode = config.get("mode", "global")
    if mode not in ("global", "grid", "pyramid"):
        raise ValueError("Spatial mode must be global, grid or pyramid")
    grids = config.get("grids", [1] if mode == "global" else [2] if mode == "grid" else [1, 2])
    if (not isinstance(grids, (list, tuple)) or not grids or
            any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in grids)):
        raise ValueError("Spatial grids must be a nonempty list of positive integers")
    grids = [int(n) for n in grids]
    if mode == "global" and grids != [1]:
        raise ValueError("Global mode requires grids=[1]")
    if mode == "grid" and len(grids) != 1:
        raise ValueError("Grid mode requires exactly one grid resolution")
    weights = np.asarray(config.get("level_weights", [1.0] * len(grids)), dtype=float)
    if (weights.ndim != 1 or len(weights) != len(grids) or
            not np.isfinite(weights).all() or np.any(weights < 0) or
            not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError("Level weights must be finite, nonnegative, match grids and have positive sum")
    # Avoid renormalising already canonical values on every resolve/cache read.
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-15):
        weights = weights / weights.sum()
    return dict(mode=mode, grids=grids, level_weights=weights.tolist())


def block_slices(shape, grid):
    """Yield row-major slices covering every pixel once, including uneven sizes."""
    height, width = shape[:2]
    if grid <= 0 or grid > min(height, width):
        raise ValueError(f"Grid {grid} exceeds image dimensions {height}x{width} or is not positive")
    ys = np.linspace(0, height, grid + 1, dtype=int)
    xs = np.linspace(0, width, grid + 1, dtype=int)
    for row in range(grid):
        for col in range(grid):
            yield slice(ys[row], ys[row + 1]), slice(xs[col], xs[col + 1])


def trim_background_blocks(image, mask, grid, min_fraction=0.01):
    """Delete block rows/columns that are (almost) only background, then repeat on the result.

    A block row or column is background when less than min_fraction of its pixels
    are foreground. Its pixel rows/columns are removed and the rest stitched together;
    the grid is laid again on the new image until nothing more can be removed.
    """
    fg = mask > 0
    while True:
        height, width = fg.shape
        if min(height, width) < grid:
            break
        ys = np.linspace(0, height, grid + 1, dtype=int)
        xs = np.linspace(0, width, grid + 1, dtype=int)
        keep_rows = [fg[ys[i]:ys[i + 1]].mean() >= min_fraction for i in range(grid)]
        keep_cols = [fg[:, xs[j]:xs[j + 1]].mean() >= min_fraction for j in range(grid)]
        if all(keep_rows) and all(keep_cols):
            break
        if not any(keep_rows) or not any(keep_cols):
            break  # Never delete everything; keep the current image.
        rows = np.concatenate([np.arange(ys[i], ys[i + 1]) for i in range(grid) if keep_rows[i]])
        cols = np.concatenate([np.arange(xs[j], xs[j + 1]) for j in range(grid) if keep_cols[j]])
        if min(len(rows), len(cols)) < grid:
            break
        image, mask, fg = (a[rows][:, cols] for a in (image, mask, fg))
    return image, mask


def spatial_descriptor(image, extract_color, config, mask=None):
    """A level contributes its weight in total, divided equally among its cells.

    With a mask, extract_color(block, block_mask) counts only foreground pixels and
    a block without foreground contributes an all-zero histogram.
    """
    blocks = []
    for grid, weight in zip(config["grids"], config["level_weights"]):
        for ys, xs in block_slices(image.shape, grid):
            if mask is None:
                block = extract_color(image[ys, xs]).astype(np.float32)
            else:
                block = extract_color(image[ys, xs], mask[ys, xs]).astype(np.float32)
            mass = block.sum()
            if mask is not None and mass == 0:
                blocks.append(block)
                continue
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("Spatial colour descriptor must have positive finite mass")
            blocks.append((block / mass) * (weight / (grid * grid)))
    return np.concatenate(blocks)
