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
