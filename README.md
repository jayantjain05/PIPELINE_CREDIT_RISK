# Credit Risk ETL + Scoring Pipeline (PySpark)

An end-to-end PySpark pipeline that ingests a private-credit style loan
portfolio (borrower → facility → transaction), engineers risk features,
and scores each facility's default risk using Spark ML.

This is a from-scratch, pipeline-engineered version of a portfolio risk
scoring workflow — data ingestion, validation, feature engineering, and
model training/scoring are all implemented as distributed Spark
transformations rather than pandas/sklearn on a single dataframe.

## Problem

Private credit lenders manage portfolios across multiple borrowers,
each with one or more credit facilities, tracked through monthly
payment/exposure snapshots. To monitor portfolio health, you need to:

1. Reliably ingest and validate borrower/facility/transaction data
2. Derive standard credit risk features (delinquency buckets, vintage,
   utilization, payment behavior trends)
3. Score each facility's risk of near-term deterioration
4. Summarize risk at the portfolio level for reporting

## Data

Data is **synthetically generated** (`src/generate_synthetic_data.py`)
to mimic a private-credit portfolio at borrower → facility →
transaction grain, with realistic-ish payment behavior driven by a
risk-tier-dependent default probability. This was a deliberate choice
over a public dataset (e.g. Lending Club) so the schema matches the
borrower/facility/collateral/transaction structure used in real
private-credit reporting, rather than a single flat loan table.

Generated volumes: ~2,000 borrowers, ~2,950 facilities, ~57,500 monthly
transaction records (24 months of history per facility).

## Pipeline stages

| Stage | File | What it does |
|---|---|---|
| 1. Ingest | `src/ingest.py` | Reads raw CSVs with an explicit enforced schema; logs row counts, null rates, and duplicate counts per table |
| 2. Clean & validate | `src/clean.py` | Dedupes on natural keys; enforces referential integrity (facilities must reference a valid borrower, transactions a valid facility); **quarantines** failing records to a separate output rather than silently dropping them |
| 3. Feature engineering | `src/features.py` | Builds a facility-level snapshot with DPD buckets, vintage (months on book), utilization ratio, and trailing-window features (roll-to-worse-bucket flag, 3-month avg payment ratio, 6-month max DPD) computed with Spark window functions |
| 4. Scoring | `src/score.py` | Trains Logistic Regression and Random Forest classifiers via `pyspark.ml.Pipeline` (StringIndexer → OneHotEncoder → VectorAssembler → classifier); evaluates both with AUC and KS statistic |
| 5. Output | `src/main.py` | Writes the feature table and scored portfolio to Parquet (partitioned by risk tier), plus a portfolio-level summary CSV (facility count, avg risk score, avg utilization by tier × DPD bucket) |

## Key design decisions

- **Quarantine, don't drop.** Records failing validation are written to
  `data/processed/quarantine/` rather than discarded, so nothing
  disappears without a trace — mirrors how you'd want reconciliation
  to work in a real reporting pipeline.
- **Window functions for trailing behavior**, not row-by-row Python
  loops — `rolled_worse_3mo`, `payment_ratio_trailing3_avg`, and
  `dpd_max_trailing6` are all computed with Spark `Window` partitioned
  by facility and ordered by period, which is the idiomatic
  distributed way to do this instead of collecting to a driver.
- **Random Forest vs Logistic Regression** are both trained and
  compared (not just one), since in prior modeling work Random Forest
  consistently outperformed Logistic Regression — this pipeline
  confirms the same pattern (see Results).
- **Label is a proxy** (60+ DPD or rolled to worse bucket in trailing
  3 months) built directly from the same window features used to
  predict it — so the reported AUC/KS are **optimistic by
  construction** and should not be read as production-grade
  discrimination power. See Limitations.

## Results (on synthetic data)

| Model | AUC | KS |
|---|---|---|
| Logistic Regression | 0.976 | 0.875 |
| Random Forest | 0.979 | 0.898 |

AUC and KS are the standard scorecard evaluation metrics used in
credit risk modeling. Random Forest modestly outperforms Logistic
Regression here, consistent with prior benchmarking.

## Limitations

- **Synthetic data**: payment behavior is simulated from a simple
  risk-tier-based probability, not real borrower behavior — it won't
  capture real-world correlations (macro shocks, industry
  concentration effects, seasonality).
- **Label leakage by construction**: the target label is derived from
  the same trailing-window DPD features used as predictors, so the
  AUC/KS numbers above demonstrate the pipeline mechanics, not a
  validated forward-looking default model. A production version would
  use a strictly forward-looking label (e.g. "did this facility hit
  90+ DPD in the *next* 3 months") with a proper train/test time split
  to avoid look-ahead bias.
- **Single-machine Spark** (`local[*]`): this runs on one machine for
  development; running on a real cluster (EMR/Databricks) would need
  partition tuning and possibly a different file format strategy for
  the raw layer.
- **No orchestration tool**: `main.py` runs stages sequentially with
  logging. A production version would use Airflow/Dagster for
  scheduling, retries, and lineage rather than a single script.

## Running it

```bash
pip install -r requirements.txt

# 1. Generate synthetic data (~2,000 borrowers, ~2,950 facilities, ~57,500 transactions)
python src/generate_synthetic_data.py

# 2. Run the full pipeline: ingest -> clean -> features -> score -> output
python src/main.py

# 3. Run tests
pytest tests/
```

Output lands in `data/processed/`:
- `feature_table/` — full engineered feature set (Parquet)
- `scored_portfolio/` — scored facilities, partitioned by risk tier (Parquet)
- `portfolio_summary/` — aggregated risk summary by tier × DPD bucket (CSV)
- `quarantine/` — any records that failed validation (CSV, if present)

## Project structure

```
credit-risk-pyspark/
├── data/
│   ├── raw/            # generated synthetic input CSVs
│   └── processed/      # pipeline outputs
├── src/
│   ├── generate_synthetic_data.py
│   ├── ingest.py
│   ├── clean.py
│   ├── features.py
│   ├── score.py
│   └── main.py
├── tests/
│   └── test_features.py
├── requirements.txt
└── README.md
```
