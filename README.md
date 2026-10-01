# Team 8 — Museum Painting Retrieval (C1)

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the course zips and extract them into `data/`, one folder per zip:

- `BBDD.zip` → `data/BBDD/` (museum database)
- `qsd1_w1.zip` → `data/qsd1_w1/` (development queries + `gt_corresps.pkl`)
- `qst1_w1.zip` → `data/qst1_w1/` (test queries, no ground truth)

```
data/
├── BBDD/        bbdd_00000.jpg, ...
├── qsd1_w1/     00000.jpg, ..., gt_corresps.pkl
└── qst1_w1/     00000.jpg, ...
```

`data/` and the zips are git-ignored.

## Run

Run the retrieval and evaluate mAP@1 and mAP@5 on the development set:

```
python src/run_retrieval.py --descriptor hsv --measure hellinger
python src/run_retrieval.py --descriptor ycbcr --measure chi2
python src/run_retrieval.py --descriptor rgb --rgb-bins 16 16 16 --measure l1
python src/run_retrieval.py --descriptor lab --lab-bins 16 16 16 --measure hellinger
python src/run_retrieval.py --descriptor lab --measure wasserstein
```

Save the test set results to a `.pkl` file for submission:

```
python src/run_retrieval.py --descriptor hsv --measure hellinger --query-set qst1 --no-gt --output results/QST1/method1/result.pkl
```

- `--descriptor`: `hsv`, `ycbcr`, `rgb` or `lab`
- `--measure`: `euclidean`, `l1`, `chi2`, `intersection`, `hellinger` or `wasserstein`
- `--k`: number of results returned per query (default 10)
- HSV options: `--hue-valid-weight` and `--hue-smoothing` (both disabled by default).

Descriptors are computed the first time a script needs them and cached in `descriptors/`. To compute them ahead of time, run `python src/compute_descriptors.py --data data/BBDD`. Run any script with `-h` to see the bin and range options.

## Experiments

```
python src/run_experiments.py --output results/experiments.csv
```

Edit `EXPERIMENTS` in `src/run_experiments.py` to configure parameter grids.
For HSV, set `hue_valid_weight` and/or `hue_smoothing` to `[False, True]` to sweep the options.
The default evaluates all four descriptors with the same six measures on QSD1
and saves mAP@1/mAP@5 in a CSV. Add `--overwrite` to replace an existing CSV.

## Tests

```
python -m pytest -q
```
