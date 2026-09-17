"""
ingest.py

Ingestion layer: reads raw borrower / facility / transaction CSVs into
Spark DataFrames with an explicitly enforced schema, and logs basic
row-count / null-rate stats so downstream stages can trust the input.
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, DateType
)
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BORROWER_SCHEMA = StructType([
    StructField("borrower_id", StringType(), False),
    StructField("industry", StringType(), True),
    StructField("risk_tier", StringType(), True),
    StructField("annual_revenue", DoubleType(), True),
    StructField("years_in_business", IntegerType(), True),
    StructField("onboarding_date", DateType(), True),
])

FACILITY_SCHEMA = StructType([
    StructField("facility_id", StringType(), False),
    StructField("borrower_id", StringType(), False),
    StructField("facility_type", StringType(), True),
    StructField("credit_limit", DoubleType(), True),
    StructField("interest_rate", DoubleType(), True),
    StructField("disbursement_date", DateType(), True),
    StructField("maturity_date", DateType(), True),
    StructField("collateral_value", DoubleType(), True),
])

TRANSACTION_SCHEMA = StructType([
    StructField("facility_id", StringType(), False),
    StructField("period_end_date", DateType(), False),
    StructField("scheduled_payment", DoubleType(), True),
    StructField("actual_payment", DoubleType(), True),
    StructField("outstanding_exposure", DoubleType(), True),
    StructField("days_past_due", IntegerType(), True),
])


def _read_csv(spark: SparkSession, path: str, schema: StructType) -> DataFrame:
    df = (
        spark.read
        .option("header", True)
        .option("dateFormat", "yyyy-MM-dd")
        .schema(schema)
        .csv(path)
    )
    return df


def _log_stats(df: DataFrame, name: str, key_cols: list[str]):
    total = df.count()
    logger.info(f"[{name}] row count = {total:,}")
    for c in key_cols:
        nulls = df.filter(F.col(c).isNull()).count()
        if nulls:
            logger.warning(f"[{name}] {c}: {nulls:,} null values ({nulls/total:.2%})")
    dupes = total - df.dropDuplicates(key_cols).count()
    if dupes:
        logger.warning(f"[{name}] {dupes:,} duplicate rows on key {key_cols}")


def load_raw_tables(spark: SparkSession, raw_dir: str) -> dict[str, DataFrame]:
    borrowers = _read_csv(spark, f"{raw_dir}/borrowers.csv", BORROWER_SCHEMA)
    facilities = _read_csv(spark, f"{raw_dir}/facilities.csv", FACILITY_SCHEMA)
    transactions = _read_csv(spark, f"{raw_dir}/transactions.csv", TRANSACTION_SCHEMA)

    _log_stats(borrowers, "borrowers", ["borrower_id"])
    _log_stats(facilities, "facilities", ["facility_id"])
    _log_stats(transactions, "transactions", ["facility_id", "period_end_date"])

    return {"borrowers": borrowers, "facilities": facilities, "transactions": transactions}
