# Team 8 — Museum Painting Retrieval (C1)

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Extract the course zips into `data/` (git-ignored): `data/BBDD/`, `data/qsd1_w1/` (with `gt_corresps.pkl`) and `data/qst1_w1/`.

## Retrieval

```
python src/run_retrieval.py --descriptor hsv --measure hellinger
python src/run_retrieval.py --descriptor hsv --measure hellinger --query-set qst1 --no-gt --output results/QST1/method2/result.pkl
```

- `--descriptor`: `hsv`, `hsv_baseline`, `ycbcr`, `rgb`, `lab`
- `--measure`: `euclidean`, `l1`, `chi2`, `intersection`, `hellinger`, `wasserstein`

Each descriptor concatenates one normalised 1D histogram per channel. Default bins:

| Descriptor     | Bins        | Other defaults |
|----------------|-------------|----------------|
| `hsv` (method 2) | 16-16-16    | Hue counts only pixels with S > 10 and V > 40; Hue not rescaled, no smoothing |
| `hsv_baseline` (method 1) | 180-256-256 | One bin per value, no mask |
| `ycbcr`        | 8-32-32     | Full 0-256 range, no chroma overflow bins |
| `rgb`          | 8-8-8       | — |
| `lab`          | 8-64-64     | — |

Descriptors are cached in `descriptors/` (or precomputed with `src/compute_descriptors.py`). Use `-h` on any script for bin/range options.

## Experiments

```
python src/run_experiments.py --grid {default,lab,ycbcr} --output config_results/<name>.csv
python src/run_experiments.py --fusion --fusion-descriptor {hsv,rgb,lab,ycbcr} --output <file>.csv
python src/sweep_hsv.py --output config_results/hsv_sweep.csv
```

Sweep results are in `config_results/`, other results in `results/`, and `notebooks/max_config.ipynb` picks the best configurations.

## Plots

Scripts in `src/visualizations/` write to `plots/`:

- `plot_best_comparison.py`: all measures for the best config of each descriptor
- `plot_hsv_histograms.py <id>` / `plot_hue_mask.py <id>`: per-image HSV descriptor and hue mask
- `plot_results.py`, `plot_example_descriptors.py`: report figures
