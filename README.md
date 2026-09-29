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
```

Save the test set results to a `.pkl` file for submission:

```
python src/run_retrieval.py --descriptor hsv --measure hellinger --query-set qst1 --no-gt --output results/qst1_method1.pkl
```

- `--descriptor`: `hsv` or `ycbcr`
- `--measure`: `euclidean`, `l1`, `chi2`, `intersection` or `hellinger`
- `--k`: number of results returned per query (default 10)

Descriptors are computed the first time a script needs them and cached in `descriptors/`. To compute them ahead of time, run `python src/compute_descriptors.py --data data/BBDD`. Run any script with `-h` to see the bin and range options.
