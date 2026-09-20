"""Calculate transparent performance metrics from validated synthetic creator data.

The thresholds in this module are portfolio-project rules. They are not
official TikTok benchmarks and do not create a final Creator Growth Score.
"""

from pathlib import Path
import sys

import pandas as pd

try:
    # This import works when another module imports src.metrics from the project root.
    from src.validate_data import validate_creator_data
except ModuleNotFoundError:
    # This import works when this file is run directly: python src/metrics.py.
    from validate_data import validate_creator_data


RAW_COLUMN_COUNT = 13
DERIVED_COLUMNS = [
    "views_per_follower",
    "views_per_follower_pct",
    "engagement_rate_pct",
    "engagement_band",
    "growth_rate_pct",
    "growth_band",
    "posting_frequency_score",
    "recency_score",
    "activity_score",
    "campaign_participation_score",
    "response_rate_pct",
    "response_band",
    "conversion_rate_pct",
    "is_recently_inactive",
    "is_high_engagement",
    "is_high_growth",
    "is_consistently_active",
]


def calculate_creator_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return original creator data plus transparent derived performance metrics.

    The input is validated first. ``copy()`` ensures that the caller's original
    DataFrame is not changed while calculations are added to the new one.
    """
    validate_creator_data(df)
    metrics = df.copy()

    # A creator's views relative to their audience size. This is a diagnostic
    # indicator, not a conversion rate. Replacing zero keeps the formula safe.
    safe_followers = metrics["followers"].replace(0, pd.NA)
    metrics["views_per_follower"] = (metrics["avg_views"] / safe_followers).fillna(0)
    metrics["views_per_follower_pct"] = metrics["views_per_follower"] * 100

    metrics["engagement_rate_pct"] = metrics["engagement_rate"] * 100
    # Portfolio rules: Low < 3%, Medium < 6%, High < 9%, otherwise Very High.
    metrics["engagement_band"] = pd.cut(
        metrics["engagement_rate"],
        bins=[float("-inf"), 0.03, 0.06, 0.09, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
        right=False,
    ).astype(str)

    metrics["growth_rate_pct"] = metrics["follower_growth_30d"] * 100
    # Portfolio rules: negative is Declining; then Stable, Growing, High Growth.
    metrics["growth_band"] = pd.cut(
        metrics["follower_growth_30d"],
        bins=[float("-inf"), 0, 0.03, 0.10, float("inf")],
        labels=["Declining", "Stable", "Growing", "High Growth"],
        right=False,
    ).astype(str)

    # One video per day is the theoretical 30-day maximum used for this score.
    metrics["posting_frequency_score"] = (metrics["videos_30d"] / 30 * 100).clip(0, 100)
    metrics["recency_score"] = (1 - metrics["last_active_days"] / 45) * 100
    metrics["recency_score"] = metrics["recency_score"].clip(0, 100)
    # Activity is weighted 60% posting and 40% recent activity.
    metrics["activity_score"] = (
        0.6 * metrics["posting_frequency_score"] + 0.4 * metrics["recency_score"]
    ).clip(0, 100)

    metrics["campaign_participation_score"] = (
        metrics["campaign_participation_90d"] / 12 * 100
    ).clip(0, 100)

    metrics["response_rate_pct"] = metrics["response_rate"] * 100
    metrics["response_band"] = pd.cut(
        metrics["response_rate"],
        bins=[float("-inf"), 0.50, 0.70, 0.85, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
        right=False,
    ).astype(str)
    metrics["conversion_rate_pct"] = metrics["conversion_rate"] * 100

    # These are simple operational flags for a later segmentation module.
    metrics["is_recently_inactive"] = metrics["last_active_days"] >= 14
    metrics["is_high_engagement"] = metrics["engagement_rate"] >= 0.06
    metrics["is_high_growth"] = metrics["follower_growth_30d"] >= 0.10
    metrics["is_consistently_active"] = (
        (metrics["videos_30d"] >= 8) & (metrics["last_active_days"] <= 7)
    )

    _validate_metric_output(df, metrics)
    return metrics


def _validate_metric_output(raw_data: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Check that calculations preserve input data and create valid outputs."""
    if len(metrics) != len(raw_data):
        raise ValueError("Metric output must contain the same number of creators as the input.")
    if not metrics["creator_id"].equals(raw_data["creator_id"]):
        raise ValueError("Metric output must preserve creator_id values and their order.")
    if not all(column in metrics.columns for column in raw_data.columns):
        raise ValueError("Metric output is missing one or more original raw-data columns.")
    if not all(column in metrics.columns for column in DERIVED_COLUMNS):
        raise ValueError("Metric output is missing one or more derived metric columns.")
    if metrics.isna().any().any():
        raise ValueError("Metric calculations introduced missing values.")

    score_columns = [
        "posting_frequency_score", "recency_score", "activity_score",
        "campaign_participation_score",
    ]
    for column in score_columns:
        if not metrics[column].between(0, 100).all():
            raise ValueError(f"Metric '{column}' must stay between 0 and 100.")

    flag_columns = [
        "is_recently_inactive", "is_high_engagement", "is_high_growth",
        "is_consistently_active",
    ]
    for column in flag_columns:
        if not pd.api.types.is_bool_dtype(metrics[column]):
            raise ValueError(f"Operational flag '{column}' must contain True or False values.")


def summarize_metrics(df: pd.DataFrame) -> dict:
    """Return a simple, reusable summary of calculated creator metrics."""
    return {
        "total_creators": len(df),
        "average_engagement_rate": round(df["engagement_rate"].mean(), 4),
        "average_follower_growth": round(df["follower_growth_30d"].mean(), 4),
        "average_activity_score": round(df["activity_score"].mean(), 2),
        "average_views_per_follower": round(df["views_per_follower"].mean(), 4),
        "average_response_rate": round(df["response_rate"].mean(), 4),
        "average_conversion_rate": round(df["conversion_rate"].mean(), 4),
        "recently_inactive_creator_count": int(df["is_recently_inactive"].sum()),
        "high_engagement_creator_count": int(df["is_high_engagement"].sum()),
        "high_growth_creator_count": int(df["is_high_growth"].sum()),
        "consistently_active_creator_count": int(df["is_consistently_active"].sum()),
    }


def main() -> None:
    """Load validated raw data, calculate metrics, and save the processed CSV."""
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "creators.csv"
    output_path = project_root / "data" / "processed" / "creator_metrics.csv"

    try:
        creators = pd.read_csv(input_path)
        metrics = calculate_creator_metrics(creators)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(output_path, index=False)

        print("Creator metrics calculated successfully.")
        print(f"Rows: {len(metrics)}")
        print(f"Original columns: {RAW_COLUMN_COUNT}")
        print(f"Derived metric columns: {len(DERIVED_COLUMNS)}")
        print(f"Output file: {output_path}")
        print("Summary metrics:")
        for name, value in summarize_metrics(metrics).items():
            print(f"- {name}: {value}")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"METRICS FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
