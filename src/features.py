"""
features.py

Feature engineering layer — builds a facility-level, latest-snapshot
feature table combining borrower, facility, and transaction history.

Features:
  - DPD bucket (current delinquency bucket)
  - Vintage (months on book)
  - Utilization ratio (outstanding exposure / credit limit)
  - Roll rate: whether the facility rolled to a worse DPD bucket vs.
    3 months prior (uses a Spark window function over facility history)
  - Payment shortfall ratio: trailing-3-month avg (actual / scheduled)
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def dpd_bucket_expr(col="days_past_due"):
    return (
        F.when(F.col(col) == 0, "Current")
        .when(F.col(col) <= 30, "1-30")
        .when(F.col(col) <= 60, "31-60")
        .when(F.col(col) <= 90, "61-90")
        .otherwise("90+")
    )


def add_dpd_bucket(df: DataFrame) -> DataFrame:
    return df.withColumn("dpd_bucket", dpd_bucket_expr())


def add_rolling_features(df: DataFrame) -> DataFrame:
    """Adds trailing window features per facility, ordered by period_end_date."""
    w = Window.partitionBy("facility_id").orderBy("period_end_date")
    w_trailing3 = w.rowsBetween(-2, 0)

    df = df.withColumn("dpd_3mo_ago", F.lag("days_past_due", 3).over(w))
    df = df.withColumn(
        "rolled_worse_3mo",
        F.when(
            F.col("dpd_3mo_ago").isNotNull() & (F.col("days_past_due") > F.col("dpd_3mo_ago")),
            1,
        ).otherwise(0),
    )
    df = df.withColumn(
        "payment_ratio",
        F.when(F.col("scheduled_payment") > 0, F.col("actual_payment") / F.col("scheduled_payment")).otherwise(1.0),
    )
    df = df.withColumn("payment_ratio_trailing3_avg", F.avg("payment_ratio").over(w_trailing3))
    df = df.withColumn("dpd_max_trailing6", F.max("days_past_due").over(w.rowsBetween(-5, 0)))
    return df


def build_facility_snapshot(transactions: DataFrame) -> DataFrame:
    """Reduce transaction history down to the latest snapshot per facility,
    carrying forward the rolling features computed over full history."""
    enriched = add_dpd_bucket(transactions)
    enriched = add_rolling_features(enriched)

    latest_window = Window.partitionBy("facility_id").orderBy(F.col("period_end_date").desc())
    latest = (
        enriched.withColumn("rn", F.row_number().over(latest_window))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )
    return latest


def add_vintage(df: DataFrame, snapshot_date_col="period_end_date") -> DataFrame:
    return df.withColumn(
        "vintage_months",
        F.months_between(F.col(snapshot_date_col), F.col("disbursement_date")).cast("int"),
    )


def add_utilization(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "utilization_ratio",
        F.when(F.col("credit_limit") > 0, F.col("outstanding_exposure") / F.col("credit_limit")).otherwise(None),
    )


def build_feature_table(borrowers: DataFrame, facilities: DataFrame, transactions: DataFrame) -> DataFrame:
    facility_snapshot = build_facility_snapshot(transactions)

    joined = (
        facility_snapshot
        .join(facilities, on="facility_id", how="inner")
        .join(borrowers, on="borrower_id", how="inner")
    )

    joined = add_vintage(joined)
    joined = add_utilization(joined)

    feature_cols = [
        "borrower_id", "facility_id", "period_end_date",
        "risk_tier", "industry", "years_in_business", "annual_revenue",
        "facility_type", "credit_limit", "interest_rate", "collateral_value",
        "outstanding_exposure", "days_past_due", "dpd_bucket",
        "vintage_months", "utilization_ratio",
        "rolled_worse_3mo", "payment_ratio_trailing3_avg", "dpd_max_trailing6",
    ]
    return joined.select(*feature_cols)
