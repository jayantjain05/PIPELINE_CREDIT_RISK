"""
clean.py

Cleaning & validation layer.

- Deduplicates on natural keys
- Flags/quarantines records that fail basic validation rules instead of
  silently dropping them (so nothing disappears without a trace)
- Enforces referential integrity: transactions must reference a known
  facility, facilities must reference a known borrower
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def clean_borrowers(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    df = df.dropDuplicates(["borrower_id"])
    valid = df.filter(
        F.col("borrower_id").isNotNull()
        & F.col("annual_revenue").isNotNull()
        & (F.col("annual_revenue") > 0)
    )
    quarantined = df.subtract(valid)
    logger.info(f"[borrowers] valid={valid.count():,} quarantined={quarantined.count():,}")
    return valid, quarantined


def clean_facilities(df: DataFrame, valid_borrowers: DataFrame) -> tuple[DataFrame, DataFrame]:
    df = df.dropDuplicates(["facility_id"])
    valid = (
        df.join(valid_borrowers.select("borrower_id"), on="borrower_id", how="inner")
        .filter(
            F.col("credit_limit").isNotNull()
            & (F.col("credit_limit") > 0)
            & F.col("disbursement_date").isNotNull()
        )
    )
    quarantined = df.join(valid.select("facility_id"), on="facility_id", how="left_anti")
    logger.info(f"[facilities] valid={valid.count():,} quarantined={quarantined.count():,}")
    return valid, quarantined


def clean_transactions(df: DataFrame, valid_facilities: DataFrame) -> tuple[DataFrame, DataFrame]:
    df = df.dropDuplicates(["facility_id", "period_end_date"])
    valid = (
        df.join(valid_facilities.select("facility_id"), on="facility_id", how="inner")
        .filter(
            F.col("outstanding_exposure").isNotNull()
            & (F.col("outstanding_exposure") >= 0)
            & F.col("days_past_due").isNotNull()
        )
    )
    quarantined = df.join(valid.select("facility_id", "period_end_date"),
                           on=["facility_id", "period_end_date"], how="left_anti")
    logger.info(f"[transactions] valid={valid.count():,} quarantined={quarantined.count():,}")
    return valid, quarantined
