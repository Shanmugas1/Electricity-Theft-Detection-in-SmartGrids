# Energy Theft Detection in Smart Grids Using Machine Learning

Working code project accompanying the "Zeroth Review" presentation.
Detects electricity theft from smart-meter consumption time series and
produces a prioritized inspection list, following the pipeline in the
slides: **collect/preprocess → feature engineer → train models → score &
rank → generate alerts.**

## Project structure

```
energy_theft_detection/
├── data/
│   ├── generate_data.py       # synthetic smart-meter data generator
│   └── smart_meter_data.csv   # generated dataset (created by the script above)
├── src/
│   ├── data_preprocessing.py  # load + clean consumption series
│   ├── feature_engineering.py # weekly/statistical/autocorrelation/change-point features
│   ├── models.py              # Random Forest, XGBoost, LSTM
│   ├── evaluate.py            # accuracy/precision/recall/F1/ROC-AUC + precision@k
│   ├── risk_scoring.py        # combine model scores into a ranked alert list
│   └── visualize_results.py   # comparison charts
├── outputs/                   # generated after running the pipeline
├── run_pipeline.py            # end-to-end script
└── requirements.txt
```

## About the dataset

The proposal (slide 5) references public **SGCC-style** smart meter
datasets with labeled normal/theft cases. Those live behind external
download portals this environment can't reach, so `data/generate_data.py`
creates a **synthetic** dataset with the same shape (`CONS_NO` rows, daily
kWh columns, `FLAG` theft label) and injects four realistic theft
signatures: partial bypass, periodic zero-reporting, sudden-drop tampering,
and consumption capping.

**To use the real SGCC dataset instead:** download it, save it with columns
`CONS_NO, <date columns...>, FLAG`, and point `run_pipeline.py`'s
`DATA_PATH` at it — no other code changes needed.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate the synthetic dataset (skip if you're supplying real data)
python data/generate_data.py --n_consumers 1200 --n_days 730 --out data/smart_meter_data.csv

# 2. Run the full pipeline: preprocessing -> features -> training -> evaluation -> alerts
python run_pipeline.py

# 3. (optional) Generate comparison charts
python src/visualize_results.py
```

## What gets produced (`outputs/`)

| File | Contents |
|---|---|
| `model_comparison.csv` / `.png` | Accuracy, precision, recall, F1, ROC-AUC per model |
| `precision_recall_at_k.csv` / `.png` | Precision/recall among the top 5%/10%/20% highest-risk consumers — the operational metric for inspection budgets |
| `inspection_alert_list.csv` | Every test-set consumer ranked by combined risk score |
| `metrics.json` | Full metrics incl. confusion matrices, all models |

## Models (slide 5 / slide 6)

- **Random Forest** and **XGBoost** — trained on 16 engineered features per
  consumer (load variance, weekday/weekend ratio, autocorrelation at 1/7/30
  day lags, change-point score, capping score, zero-day runs, trend slope).
- **LSTM** — trained directly on the (down-sampled) daily consumption
  sequence, to capture patterns the hand-crafted features might miss.
- **Risk scoring** — weighted combination of all three models' probabilities
  produces the final ranked alert list.

On the synthetic dataset, Random Forest and XGBoost substantially
outperform the LSTM baseline — expected at this data scale (a few hundred
consumers); LSTMs typically need thousands–tens of thousands of sequences to
outperform tree models on tabular-style features. This gap itself is a
legitimate finding to report in later reviews, and `src/models.py` is
structured so hyperparameters (LSTM units, epochs, tree depth) are easy to
tune once the real, larger SGCC dataset is substituted in.

## Scope note

This matches the **Zeroth Review scope** from the slides: offline training
and evaluation only. Real-time deployment, SCADA/billing integration, and a
field-staff dashboard are explicitly out of scope for this stage.
