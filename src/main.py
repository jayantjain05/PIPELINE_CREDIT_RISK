"""
main.py

Orchestrates the full credit risk ETL + scoring pipeline:

  1. Ingest raw borrower/facility/transaction data
  2. Clean & validate (quarantine bad records)
  3. Engineer features (DPD buckets, vintage, utilization, roll rates)
  4. Score the portfolio (Random Forest, benchmarked against Logistic Regression)
  5. Write scored output + portfolio-level summary, partitioned by risk tier

Run from the project root:
    python src/main.py
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ingest import load_raw_tables
from clean import clean_borrowers, clean_facilities, clean_transactions
from features import build_feature_table
from score import score_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("credit-risk-pyspark-pipeline")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def main():
    t0 = time.time()
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    logger.info("=== Stage 1: Ingest ===")
    raw = load_raw_tables(spark, RAW_DIR)

    logger.info("=== Stage 2: Clean & validate ===")
    borrowers, borrowers_bad = clean_borrowers(raw["borrowers"])
    facilities, facilities_bad = clean_facilities(raw["facilities"], borrowers)
    transactions, transactions_bad = clean_transactions(raw["transactions"], facilities)

    quarantine_dir = os.path.join(PROCESSED_DIR, "quarantine")
    for name, df in [("borrowers", borrowers_bad), ("facilities", facilities_bad), ("transactions", transactions_bad)]:
        if df.count() > 0:
            df.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(quarantine_dir, name))

    logger.info("=== Stage 3: Feature engineering ===")
    feature_table = build_feature_table(borrowers, facilities, transactions)
    feature_table.cache()
    logger.info(f"Feature table rows: {feature_table.count():,}")

    feature_table.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "feature_table"))

    logger.info("=== Stage 4: Scoring ===")
    scored_output, metrics = score_portfolio(feature_table)

    scored_output.write.mode("overwrite").partitionBy("risk_tier").parquet(
        os.path.join(PROCESSED_DIR, "scored_portfolio")
    )

    logger.info("=== Stage 5: Portfolio summary ===")
    summary = (
        scored_output.groupBy("risk_tier", "dpd_bucket")
        .agg(
            F.count("*").alias("facility_count"),
            F.avg("risk_score").alias("avg_risk_score"),
            F.avg("utilization_ratio").alias("avg_utilization"),
        )
        .orderBy("risk_tier", "dpd_bucket")
    )
    summary.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(PROCESSED_DIR, "portfolio_summary")
    )
    summary.show(20, truncate=False)

    logger.info("=== Model comparison (AUC / KS) ===")
    for m in metrics:
        logger.info(f"  {m['model']}: AUC={m['auc']:.4f}  KS={m['ks']:.4f}")

    logger.info(f"Pipeline completed in {time.time() - t0:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
