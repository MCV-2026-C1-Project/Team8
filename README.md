# Team 8 — Museum Painting Retrieval (C1)

Colour-histogram-based image retrieval for the UAB museum painting project (Weeks 1–2).

## Setup

```sh
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Extract the course datasets into `data/` (git-ignored), including `data/BBDD/` and `data/qsd1_w1/` (with `gt_corresps.pkl`). Add the test sets when available. **QSD1-W2 uses the same cropped images as QSD1-W1.** Run commands from the repository root.

## Retrieval

```sh
# Run a descriptor and distance directly
python src/run_retrieval.py --descriptor hsv --measure l1

# Run a configuration selected during experiments
python src/run_retrieval.py --config results/week2/best_config.json --query-set data/qsd1_w1

# Generate top-10 predictions for the test set
python src/run_retrieval.py --config results/week2/best_config.json --query-set data/qst1_w2 --no-gt --k 10 --output results/week2/QST1/method1/result.pkl
```

- `--descriptor`: select a colour descriptor (listed below).
- `--measure`: `euclidean`, `l1`, `chi2`, `intersection`, `hellinger`, `wasserstein` (global 1D only).
- `--config`: load a configuration exported by `run_experiments.py`.
- `--query-set`, `--db-dir`, `--k`, `--output`: select the data and retrieval output.

Retrieval ranks database paintings by their similarity to each query. The output `.pkl` contains one ordered list of painting IDs per query.

## Colour descriptors

| Type | Descriptors | Description |
|---|---|---|
| **1D (Week 1)** | `hsv`, `hsv_baseline`, `rgb`, `lab`, `ycbcr` | Independent histograms for each colour channel |
| **2D (Week 2)** | `hsv_hs2d`, `lab_ab2d` | Joint H–S or a–b histograms, optionally including brightness |
| **3D (Week 2)** | `hsv_3d`, `lab_3d` | Joint histograms across three colour channels |

Histogram bins and other parameters are configurable through Python, JSON or CLI options.

### Photometric normalisation

Optional photometric normalisation supports `lab`, `ycbcr`, `lab_ab2d` and `lab_3d` in global, grid and pyramid layouts.
It centres each channel around 128 and scales all channels using the luminance standard deviation (target 50), then clips.
The transformation runs once per complete image, before spatial partitioning; histograms use the prepared float32 pixels.
The default `--normalization none` preserves existing values and caches; exported configurations retain the selected mode.
RGB/HSV remain unnormalised baselines.

```bash
python src/run_retrieval.py --descriptor lab_ab2d --normalization photometric --spatial-mode pyramid --grids 1 2 --measure l1
```

With `--remove-background`, Lab/YCbCr statistics and histogram counts use only foreground pixels; preparation runs once before trimming or partitioning. All four Lab/YCbCr families support masks with `none` or `photometric`; RGB keeps its existing mask support and HSV masks remain unsupported. Empty foreground masks raise an error.
Experiments also accept `--mask-source pred|gt`, `--trim-mode block|bbox` or `--no-trim`. GT masks are development diagnostics only. Database caches are reused; masked queries are never cached.
Masked CSVs and exported provenance record the mask source/trim policy and `replay_flags`; replay those flags explicitly with `run_retrieval.py --config` (loading the descriptor alone does not enable masking).

```sh
python src/run_experiments.py --config configs/week2/week2_qsd2_background_photometric.json --remove-background --mask-source pred --trim-mode block --cache-dir descriptors --output results/week2/qsd2_pred.csv --export-best results/week2/qsd2_pred_best.json
```

### Spatial histograms (Week 2)

- **`global`**: one descriptor for the entire image.
- **`grid`**: histograms from non-overlapping image blocks.
- **`pyramid`**: combine histograms at multiple spatial resolutions, with configurable level weights.

Select a representation using `--spatial-mode`, `--grids` and `--level-weights`. Background removal is handled separately from the cropped-image retrieval experiments.

### Background removal (Week 2)
`src/remove_background.py` estimates the wall colour from the image borders and returns a binary mask at the original image resolution: 255 = painting, 0 = background. Segmentation uses colour, connectivity and mask postprocessing; it does not use edge/contour detection or object detectors.

**Method**
The default `color_distribution` method learns the wall’s Lab colour distribution from border pixels, accounting for brightness changes. Only wall-like regions connected to the image boundary are removed, preserving enclosed painting regions with similar colours.

Postprocessing closes gaps, removes small components, fills painting interiors and suppresses rows or columns with little foreground. The output is a binary mask: 255 = painting, 0 = background.

The alternative `quadratic` method models spatial variations in wall colour using a quadratic surface. Both methods use colour and mask connectivity without edge detection, contours or object detectors.

### Evaluate segmentations
Place development JPEG images and their matching ground-truth PNG masks in `data/qsd2_w2/` (for example, `00001.jpg` and `00001.png`). Use `--gt` to select a separate mask directory if needed.

```sh
# Evaluate the colour-distribution model using the retrieval tolerance
python src/evaluate_background.py --images data/qsd2_w2 --gt data/qsd2_w2 --method color_distribution --color-sigma 5.5 --output results/background_masks

# Search colour-distribution tolerances on the development set
python src/evaluate_background.py --images data/qsd2_w2 --gt data/qsd2_w2 --method color_distribution --grid-search --output results/background_masks_sweep

# Evaluate the spatial quadratic baseline
python src/evaluate_background.py --images data/qsd2_w2 --gt data/qsd2_w2 --method quadratic --threshold 20 --output results/background_masks_quadratic
```

A single run saves `<output>/masks/<query_id>.png` and per-image precision, recall, F1 and pixel counts in `<output>/metrics.csv`. Printed scores are macro averages over images. Grid search saves masks under `<output>/threshold_<value>/masks/` and a `threshold_comparison.csv` ranked by mean F1. It tests colour tolerances from 2 to 6 (step 0.5) for `color_distribution`, or Lab thresholds 10, 15, 20 and 25 for `quadratic`.

| Option | Purpose |
|---|---|
| `--color-sigma` | Mahalanobis tolerance for `color_distribution`; larger values classify more colours as wall |
| `--threshold`, `--mad-multiplier`, `--l-weight` | Minimum Lab threshold, adaptive scale and lightness weight for `quadratic` |
| `--border-ratio`, `--sample-step` | Border width fraction and sampling spacing |
| `--closing-size`, `--min-component-ratio` | Closing kernel size and minimum component area fraction |
| `--enclosed-only` | Disable both horizontal and vertical span filling; keep closing, component filtering and enclosed-hole filling |
| `--no-fill-columns` | Disable vertical span filling only |
| `--no-fill-interior` | Disable interior refinement; final row/column filtering still runs |
| `--min-row-foreground`, `--min-col-foreground` | Final foreground-fraction cutoffs; use 0 to disable an axis |

## Retrieve with foreground masks and export QST2 results

Enable predicted masks explicitly, including when loading an exported descriptor configuration:

```sh
# Evaluate foreground-only retrieval on the development set
python src/run_retrieval.py --config results/week2/qsd2_best.json --query-set data/qsd2_w2 --remove-background --mask-source pred --trim-mode block

# Export top-10 rankings and masks for blind QST2 evaluation
python src/run_retrieval.py --config results/week2/qsd2_best.json --query-set data/qst2_w2 --remove-background --mask-source pred --trim-mode block --no-gt --k 10 --output results/week2/QST2/method1/result.pkl --save-masks results/week2/QST2/method1
```

The QST2 output directory contains `result.pkl` and one mask per query, with the same basename and a `.png` extension (for example, `00001.jpg` → `00001.png`). The pickle contains one ranked list of 10 integer database IDs per query. `--save-masks` saves the original-resolution masks before descriptor trimming. Ground-truth masks (`--mask-source gt`) are for development diagnostics only.

## Experiments

Use `src/run_experiments.py` to compare descriptors, distances and spatial representations. Experiments can be configured in three ways:

**Python / original grids** — edit `EXPERIMENTS` in `src/run_experiments.py`, or use the existing grids:

```sh
python src/run_experiments.py --grid default --output config_results/default.csv
python src/run_experiments.py --grid lab --output config_results/lab.csv
python src/run_experiments.py --grid ycbcr --output config_results/ycbcr.csv
python src/sweep_hsv.py --output config_results/hsv_sweep.csv
```

**JSON** — edit the experiment suites in `configs/week1/` and `configs/week2/`:

```sh
python src/run_experiments.py --config configs/week1/week1_1d_none.json --cache-dir descriptors --output results/week1/1d_none.csv --export-best results/week1/1d_none_best.json
python src/run_experiments.py --config configs/week1/week1_1d_photometric.json --cache-dir descriptors --output results/week1/1d_photometric.csv --export-best results/week1/1d_photometric_best.json
python src/run_experiments.py --config configs/week2/week2_joint_spatial_normalization.json --cache-dir descriptors --output results/week2/joint_spatial.csv --export-best results/week2/best_config.json
```

The example suites compare colour histograms, spatial layouts and pyramid weights. Their settings can be modified freely.

**CLI** — run an experiment directly without editing a configuration file:

```sh
python src/run_experiments.py --descriptor hsv_hs2d --spatial-mode grid --grids 2 --measures l1 chi2 hellinger --output results/week2/quick.csv
```

Use `--dry-run` to preview experiments, `--overwrite` to replace existing outputs and `-h` for more options. `--export-best` saves a configuration that can be reused with `run_retrieval.py`.

## Plots

Scripts in `src/visualizations/` write to `plots/`:

- `plot_best_comparison.py`: all measures for the best configuration of each descriptor.
- `plot_hsv_histograms.py <id>` / `plot_hue_mask.py <id>`: per-image HSV histograms and Hue mask.
- `plot_results.py`, `plot_example_descriptors.py`: report figures.
