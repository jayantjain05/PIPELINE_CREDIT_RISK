"""
Basic unit tests for the feature engineering logic.
Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pyspark.sql import SparkSession
from features import add_dpd_bucket, add_utilization, add_vintage


@pytest.fixture(scope="module")
def spark():
    s = SparkSession.builder.appName("test").master("local[1]").getOrCreate()
    yield s
    s.stop()


def test_dpd_bucket(spark):
    df = spark.createDataFrame(
        [(0,), (15,), (45,), (75,), (120,)], ["days_past_due"]
    )
    result = add_dpd_bucket(df).select("dpd_bucket").rdd.flatMap(lambda x: x).collect()
    assert result == ["Current", "1-30", "31-60", "61-90", "90+"]


def test_utilization_ratio(spark):
    df = spark.createDataFrame(
        [(1000.0, 500.0), (1000.0, 0.0), (0.0, 100.0)],
        ["credit_limit", "outstanding_exposure"],
    )
    result = add_utilization(df).select("utilization_ratio").rdd.flatMap(lambda x: x).collect()
    assert result[0] == pytest.approx(0.5)
    assert result[1] == pytest.approx(0.0)
    assert result[2] is None


def test_vintage_months(spark):
    df = spark.createDataFrame(
        [("2024-01-01", "2023-01-01")], ["period_end_date", "disbursement_date"]
    ).selectExpr("to_date(period_end_date) as period_end_date", "to_date(disbursement_date) as disbursement_date")
    result = add_vintage(df).select("vintage_months").rdd.flatMap(lambda x: x).collect()
    assert result[0] == 12
