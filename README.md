# Team 8 — Museum Painting Retrieval (C1)

Colour-histogram-based image retrieval for the UAB museum painting project (Weeks 1–2).

## Setup

```sh
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Extract the course datasets into `data/` (git-ignored): `data/BBDD/`, `data/qsd1_w1/` and `data/qsd2_w2/` (each with `gt_corresps.pkl`; QSD2 also has the ground-truth masks `<id>.png`), and the test sets `data/qst1_w2/` and `data/qst2_w2/` (from `qst_w2.zip`). **QSD1-W2 uses the same cropped images as QSD1-W1.** Run commands from the repository root.

## Best results

| Query set | Descriptor (exported configuration) | mAP@1 | mAP@5 |
|---|---|---|---|
| QSD1, Week 1 best | HSV 16-16-16, global, intersection (`results/week1/1d_none_best.json`) | 0.600 | 0.672 |
| QSD1, Week 2 best | YCbCr 64-64-64, photometric, 4×4 grid, L1 (`results/week2/qsd1_best_with_norm.json`) | 0.967 | 0.978 |
| QSD2, whole photo | QSD2 best descriptor without background removal | 0.167 | 0.217 |
| QSD2, Week 2 best | Lab 3D 16-16-8, photometric, 2×2 grid, L1, predicted masks (`results/week2/qsd2_best.json`) | 0.900 | 0.928 |

QSD2 background masks (default `foreground_mask`): mean precision 0.969, recall 0.985, F1 0.976 per image. Ground-truth masks give the same QSD2 retrieval scores as the predicted ones.

## Retrieval

```sh
# Run a descriptor and distance directly
python src/run_retrieval.py --descriptor hsv --measure l1

# Run a configuration selected during experiments
python src/run_retrieval.py --config results/week2/qsd1_best_with_norm.json --query-set qsd1

# QSD2: remove the wall and describe only the painting's pixels
python src/run_retrieval.py --config results/week2/qsd2_best.json --query-set qsd2_w2 --remove-background
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

<<<<<<< HEAD
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

The default colour tolerance is `color_sigma=5.5`, selected experimentally for achieving the highest mean F1 on the development set. This value is used consistently for background-removal evaluation and retrieval.

## Retrieve with foreground masks and export QST2 results

Enable predicted masks explicitly, including when loading an exported descriptor configuration:

```sh
# Evaluate foreground-only retrieval on the development set
python src/run_retrieval.py --config results/week2/qsd2_best.json --query-set data/qsd2_w2 --remove-background --mask-source pred --trim-mode block

# Export top-10 rankings and masks for blind QST2 evaluation
python src/run_retrieval.py --config results/week2/qsd2_best.json --query-set data/qst2_w2 --remove-background --mask-source pred --trim-mode block --no-gt --k 10 --output results/week2/QST2/method1/result.pkl --save-masks results/week2/QST2/method1
```

The QST2 output directory contains `result.pkl` and one mask per query, with the same basename and a `.png` extension (for example, `00001.jpg` → `00001.png`). The pickle contains one ranked list of 10 integer database IDs per query. `--save-masks` saves the original-resolution masks before descriptor trimming. Ground-truth masks (`--mask-source gt`) are for development diagnostics only.

=======

## Background removal (Week 2)

`src/remove_background.py` (`foreground_mask`) separates the painting from the wall using colour only (no edges, contours or rectangle fitting):

1. Sample a strip of 4 % of the image around the border, assumed to be wall.
2. Fit the wall's chroma (Lab a, b) as a quadratic function of lightness, with robust re-weighting, so spotlights and shadows on the wall are tolerated.
3. Mark pixels within Mahalanobis distance 5.5 of the predicted wall colour as wall-like (very dark pixels never are); background is the wall-like pixels connected to the image border.
4. Clean up: closing, remove specks under 0.1 % of the image, fill each region row by row and column by column, fill holes, and clear rows/columns with less than 50 % foreground (kept if that would empty the mask).

`method="quadratic"` keeps the earlier spatial wall model. Evaluate masks against the ground truth (writes masks and per-image `metrics.csv`):

```sh
python src/evaluate_background.py --images data/qsd2_w2 --gt data/qsd2_w2 --color-sigma 5.5 --output results/background_masks
```

Pass `--color-sigma 5.5` to match `foreground_mask` (the script's own default is 3.0; `--grid-search` tries several values).

With `--remove-background`, query descriptors count only foreground pixels. Before the grid is laid, `--trim-mode block` (default) deletes block rows/columns with less than 10 % foreground and `--trim-mode bbox` crops to the foreground's bounding box; `--no-trim` keeps the whole image. `--save-masks DIR` writes the predicted masks as `<id>.png`. Do not use background removal on QSD1/QST1: those images are already cropped.

## Week 2 delivery

```sh
# QST1: cropped paintings, whole image
python src/run_retrieval.py --config results/week2/qsd1_best_with_norm.json --query-set qst1_w2 --no-gt --k 10 --output results/week2/QST1/method1/result.pkl

# QST2: paintings on a wall, predicted masks saved next to the results
python src/run_retrieval.py --config results/week2/qsd2_best.json --query-set qst2_w2 --no-gt --k 10 --remove-background --save-masks results/week2/QST2/method1 --output results/week2/QST2/method1/result.pkl
```

Each `result.pkl` is a list with one list of 10 BBDD IDs per query, in query order.
>>>>>>> bb79d34 (Add Week 2 plots, QST deliveries and README update)

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
python src/run_experiments.py --config configs/week2/week2_joint_spatial_normalization.json --cache-dir descriptors --output results/week2/qsd1_pred_with_norm.csv --export-best results/week2/qsd1_best_with_norm.json
```

The example suites compare colour histograms, spatial layouts and pyramid weights. Their settings can be modified freely.

**CLI** — run an experiment directly without editing a configuration file:

```sh
python src/run_experiments.py --descriptor hsv_hs2d --spatial-mode grid --grids 2 --measures l1 chi2 hellinger --output results/week2/quick.csv
```

Use `--dry-run` to preview experiments, `--overwrite` to replace existing outputs and `-h` for more options. `--export-best` saves a configuration that can be reused with `run_retrieval.py`.

## Plots

Each script in `src/visualizations/` produces one figure in `plots/` (`--output` to change it; `-h` for options):

**Week 1**
- `plot_best_comparison.py`: all measures for the best configuration of each descriptor.
- `plot_hsv_histograms.py <id>` / `plot_hue_mask.py <id>`: per-image HSV histograms and Hue mask.
- `plot_results.py`, `plot_example_descriptors.py`: report figures.

**Week 2: descriptors (QSD1)**
- `plot_descriptor_types.py`: best global mAP@5 of each descriptor type, with and without photometric normalisation.
- `plot_block_grid_example.py`: a query split into a 4×4 grid and its concatenated descriptor.
- `plot_grid_size_sweep.py`: mAP@5 against the number of blocks per side.
- `plot_pyramid_vs_grid.py`: spatial pyramids against the grid of their finest level.
- `plot_week1_vs_week2.py`: Week 1 best, Week 1 with normalisation and Week 2 best.
- `plot_week1_vs_week2_example.py`: a query Week 1 misses and Week 2 finds, top 3 of each.

**Week 2: background removal and QSD2**
- `plot_mask_pipeline.py`: the background-removal steps on one query.
- `plot_wall_model.py`: the fitted wall colour against lightness.
- `plot_mask_examples.py`: predicted masks on harder queries, with errors against the ground truth.
- `plot_block_foreground.py`: foreground percentage per block of the grid.
- `plot_trim_limit.py`: retrieval and lost painting for each trim mode and limit.
- `plot_foreground_descriptor.py`: how a query descriptor uses the mask.
- `plot_qsd2_retrieval.py`: QSD2 mAP without removal, with predicted masks and with GT masks.
- `plot_qsd2_failures.py`: the QSD2 queries wrong at rank 1.
