---
title: Credit Risk ETL & Scoring Pipeline
subtitle: An end-to-end PySpark pipeline for private-credit portfolio risk
author: jayantjain05/PIPELINE_CREDIT_RISK
date: September 2026
---

## Problem

Private credit lenders manage portfolios across many borrowers, each with one
or more credit facilities, tracked through monthly payment/exposure snapshots.

To monitor portfolio health, you need to:

- Reliably **ingest and validate** borrower / facility / transaction data
- Derive standard **credit risk features** (DPD buckets, vintage, utilization, payment trends)
- **Score** each facility's risk of near-term deterioration
- **Summarize** risk at the portfolio level for reporting

This pipeline implements the full workflow as distributed Spark
transformations, not pandas/sklearn on a single dataframe.

## Data Model

Synthetically generated at **borrower → facility → transaction** grain, to
mirror how real private-credit portfolios are structured (not a single flat
loan table like public datasets such as Lending Club).

| Entity | Key fields |
|---|---|
| Borrower | borrower_id, industry, risk_tier, annual_revenue, years_in_business |
| Facility | facility_id, borrower_id, facility_type, credit_limit, interest_rate, collateral_value |
| Transaction | facility_id, period_end_date, scheduled/actual payment, outstanding_exposure, days_past_due |

**Generated volumes:** 2,000 borrowers · 2,954 facilities · 57,580 monthly
transaction records (up to 24 months of history per facility)

## Pipeline Architecture

**Ingest → Clean & Validate → Feature Engineering → Scoring → Output**

| Stage | File | What it does |
|---|---|---|
| 1. Ingest | `ingest.py` | Enforced schema; row count, null-rate, dupe logging |
| 2. Clean | `clean.py` | Dedup on natural keys; referential integrity; quarantine bad rows |
| 3. Features | `features.py` | DPD buckets, vintage, utilization, trailing-window features |
| 4. Scoring | `score.py` | LR + RF via `pyspark.ml.Pipeline`; AUC / KS evaluation |
| 5. Output | `main.py` | Parquet feature table + scored portfolio + summary CSV |

## Key Design Decisions

- **Quarantine, don't drop** — failing records go to `data/processed/quarantine/` rather than being silently discarded
- **Window functions for trailing behavior** — `rolled_worse_3mo`, `payment_ratio_trailing3_avg`, `dpd_max_trailing6` computed with Spark `Window`, partitioned by facility, ordered by period — not row-by-row Python loops
- **Two models compared** — Random Forest benchmarked against Logistic Regression rather than shipping a single model
- **Label is a proxy** — built from the same trailing-window features used to predict it, so reported AUC/KS are optimistic by construction (see Limitations)

## Feature Engineering

Computed per facility over full transaction history, reduced to the latest
snapshot per facility:

- **dpd_bucket** — Current / 1–30 / 31–60 / 61–90 / 90+
- **vintage_months** — months between snapshot date and disbursement
- **utilization_ratio** — outstanding_exposure / credit_limit
- **rolled_worse_3mo** — DPD worse than 3 months ago (`F.lag` over a window)
- **payment_ratio_trailing3_avg** — trailing 3-month avg of actual/scheduled payment
- **dpd_max_trailing6** — max DPD over trailing 6 months

Categorical fields (`risk_tier`, `industry`, `facility_type`) are indexed and
one-hot encoded, then assembled with numeric features into a single Spark ML
feature vector.

## Pipeline Run Results

Actual execution of `python src/main.py` on the generated dataset:

| Metric | Value |
|---|---|
| Facilities scored | 2,954 |
| Records quarantined | 0 |
| Test set size | 730 facilities |
| High-risk rate (test set) | 9.3% |
| Avg. utilization (test set) | 19.3% |

Ingest: 2,000 borrowers · 2,954 facilities · 57,580 transactions — all valid,
zero quarantined.

## Model Comparison

| Model | AUC | KS statistic |
|---|---|---|
| Logistic Regression | 0.9755 | 0.8746 |
| **Random Forest** (100 trees, depth 6) | **0.9786** | **0.8977** |

AUC and KS are the standard scorecard evaluation metrics used in credit risk
modeling. Random Forest modestly outperforms Logistic Regression, consistent
with expectations for this kind of tabular feature set.

## Risk Tier Breakdown

Random Forest scores on the held-out test set:

| Risk tier | Facilities | Avg risk score | Avg utilization | High-risk count |
|---|---|---|---|---|
| Prime | 338 | 0.028 | 17.7% | 9 |
| Near-Prime | 279 | 0.103 | 18.1% | 29 |
| Subprime | 113 | 0.282 | 26.7% | 30 |

Risk score and delinquency both increase monotonically from Prime to
Subprime — the model recovers the intended risk ordering.

## Limitations

- **Label leakage by construction** — the target label is derived from the
  same trailing-window DPD features used as predictors, so AUC/KS here
  demonstrate pipeline mechanics, not a validated forward-looking model. A
  production version would use a strictly forward-looking label with a
  proper time-based train/test split.
- **Synthetic data** — payment behavior is a simple risk-tier probability,
  not real borrower behavior; no macro shocks, industry concentration, or
  seasonality effects.
- **Single-machine Spark** (`local[*]`) — a real cluster deployment
  (EMR/Databricks) would need partition tuning and a different raw-layer
  file strategy.
- **No orchestration tool** — a production version would use Airflow/Dagster
  for scheduling, retries, and lineage instead of a single script.

## Running It

```bash
pip install -r requirements.txt

# 1. Generate synthetic data
python src/generate_synthetic_data.py

# 2. Run the full pipeline
python src/main.py

# 3. Run tests
pytest tests/
```

Output lands in `data/processed/`: `feature_table/`, `scored_portfolio/`
(partitioned by risk tier), `portfolio_summary/`, and `quarantine/` (if any).

**Repository:** github.com/jayantjain05/PIPELINE_CREDIT_RISK
