"""
score.py

Scoring layer — trains and applies a Spark ML classification model on
the engineered feature table to produce a risk score per facility.

Label definition: a facility is flagged "high_risk" (1) if it is
currently 60+ DPD or rolled to a worse bucket in the trailing 3 months.
This is a proxy label for demonstration; in production this would be
a proper forward-looking default flag (e.g. 90+ DPD within N months).

Trains both Logistic Regression and Random Forest (matching prior
benchmark work where Random Forest outperformed LR) and evaluates both
with AUC and a KS statistic, which are the standard credit-risk model
evaluation metrics.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def add_label(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "label",
        F.when((F.col("days_past_due") >= 60) | (F.col("rolled_worse_3mo") == 1), 1).otherwise(0),
    )


CATEGORICAL_COLS = ["risk_tier", "industry", "facility_type"]
NUMERIC_COLS = [
    "years_in_business", "annual_revenue", "credit_limit", "interest_rate",
    "collateral_value", "outstanding_exposure", "vintage_months",
    "utilization_ratio", "payment_ratio_trailing3_avg", "dpd_max_trailing6",
]


def build_pipeline_stages(classifier):
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL_COLS]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in CATEGORICAL_COLS]
    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in CATEGORICAL_COLS] + NUMERIC_COLS,
        outputCol="features",
        handleInvalid="keep",
    )
    return indexers + encoders + [assembler, classifier]


def ks_statistic(predictions: DataFrame, prob_col="probability", label_col="label") -> float:
    """Computes the KS statistic between predicted-positive and predicted-negative
    class score distributions — standard credit scorecard evaluation metric."""
    get_prob1 = F.udf(lambda v: float(v[1]), "double")
    scored = predictions.withColumn("score", get_prob1(F.col(prob_col)))

    from pyspark.sql.window import Window
    total_pos = scored.filter(F.col(label_col) == 1).count()
    total_neg = scored.filter(F.col(label_col) == 0).count()
    if total_pos == 0 or total_neg == 0:
        return float("nan")

    w = Window.orderBy(F.col("score").desc()).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    cum = (
        scored
        .withColumn("is_pos", (F.col(label_col) == 1).cast("int"))
        .withColumn("is_neg", (F.col(label_col) == 0).cast("int"))
        .withColumn("cum_pos", F.sum("is_pos").over(w) / total_pos)
        .withColumn("cum_neg", F.sum("is_neg").over(w) / total_neg)
        .withColumn("ks", F.abs(F.col("cum_pos") - F.col("cum_neg")))
    )
    return cum.agg(F.max("ks")).first()[0]


def train_and_evaluate(feature_df: DataFrame, model_name: str, classifier, seed=42):
    train_df, test_df = feature_df.randomSplit([0.75, 0.25], seed=seed)

    pipeline = Pipeline(stages=build_pipeline_stages(classifier))
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(predictions)
    ks = ks_statistic(predictions)

    print(f"[{model_name}] AUC = {auc:.4f} | KS = {ks:.4f} | test rows = {test_df.count():,}")
    return model, predictions, {"model": model_name, "auc": auc, "ks": ks}


def score_portfolio(feature_df: DataFrame):
    labeled = add_label(feature_df)

    lr_model, lr_preds, lr_metrics = train_and_evaluate(
        labeled, "LogisticRegression", LogisticRegression(featuresCol="features", labelCol="label")
    )
    rf_model, rf_preds, rf_metrics = train_and_evaluate(
        labeled, "RandomForest",
        RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=6, seed=42),
    )

    get_prob1 = F.udf(lambda v: float(v[1]), "double")
    scored_output = (
        rf_preds
        .withColumn("risk_score", get_prob1(F.col("probability")))
        .select(
            "borrower_id", "facility_id", "period_end_date", "risk_tier",
            "dpd_bucket", "utilization_ratio", "label", "risk_score",
        )
    )

    return scored_output, [lr_metrics, rf_metrics]
