"""Calculate a transparent Creator Growth Score for this portfolio project.

This is a project-defined framework, not an official TikTok scoring system or
platform benchmark. It does not use machine learning.
"""

from pathlib import Path
import sys

import pandas as pd


REQUIRED_METRIC_COLUMNS = [
    "creator_id",
    "followers",
    "engagement_rate",
    "follower_growth_30d",
    "last_active_days",
    "activity_score",
    "campaign_participation_score",
    "response_rate",
    "conversion_rate",
    "is_recently_inactive",
    "is_consistently_active",
]

SCORE_BANDS = ["Low Priority", "Developing", "Growth Potential", "High Potential"]
SCORE_COLUMNS = [
    "engagement_component",
    "growth_component",
    "activity_component",
    "campaign_component",
    "response_conversion_component",
    "creator_growth_score",
    "growth_score_band",
    "priority_reason",
    "is_high_growth_priority",
    "is_growth_opportunity",
    "needs_reactivation",
]


def _normalise_to_score(values: pd.Series, minimum: float, maximum: float) -> pd.Series:
    """Map a portfolio data range to 0–100 and clip unusual values safely."""
    return ((values - minimum) / (maximum - minimum) * 100).clip(0, 100)


def _priority_reason(row: pd.Series) -> str:
    """Return a reproducible explanation using only the creator's metrics."""
    if row["last_active_days"] >= 14 and row["engagement_rate"] >= 0.06:
        return "Currently inactive despite strong engagement"
    if row["engagement_rate"] >= 0.06 and row["follower_growth_30d"] >= 0.10:
        return "High engagement and strong recent growth"
    if row["engagement_rate"] >= 0.06 and row["follower_growth_30d"] < 0:
        return "Strong engagement but declining growth"
    if row["follower_growth_30d"] >= 0.10 and row["campaign_participation_score"] < 35:
        return "High growth potential but limited campaign participation"
    # This is an explanation only, not a scoring component: follower count does
    # not directly raise the Creator Growth Score.
    if row["followers"] >= 100_000 and row["engagement_rate"] < 0.03:
        return "Large audience but low engagement"
    if row["is_consistently_active"] and 0 <= row["follower_growth_30d"] < 0.10:
        return "Consistent activity with moderate growth"
    if row["follower_growth_30d"] < 0:
        return "Declining growth requires closer monitoring"
    return "Moderate performance with room for growth"


def calculate_creator_growth_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return metric data plus a 0–100 Creator Growth Score and explanations."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Creator metrics must be provided as a pandas DataFrame.")
    missing_columns = [column for column in REQUIRED_METRIC_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Creator metrics are missing required columns: {', '.join(missing_columns)}.")
    if df[REQUIRED_METRIC_COLUMNS].isna().any().any():
        raise ValueError("Creator metrics contain missing values required for scoring.")

    scores = df.copy()

    # Engagement reflects how strongly an audience responds to content.
    scores["engagement_component"] = _normalise_to_score(scores["engagement_rate"], 0.01, 0.15)
    # Growth reflects recent momentum; negative 30-day growth lowers this score.
    scores["growth_component"] = _normalise_to_score(scores["follower_growth_30d"], -0.10, 0.25)
    # Activity reflects whether the creator is currently posting and recently active.
    scores["activity_component"] = scores["activity_score"].clip(0, 100)
    # Participation is a simple indication of operational campaign readiness.
    scores["campaign_component"] = scores["campaign_participation_score"].clip(0, 100)

    # Response and conversion can indicate collaboration and commercial potential.
    response_component = _normalise_to_score(scores["response_rate"], 0.20, 0.98)
    conversion_component = _normalise_to_score(scores["conversion_rate"], 0.002, 0.08)
    scores["response_conversion_component"] = (response_component + conversion_component) / 2

    # Follower count is intentionally NOT a positive scoring factor. Audience
    # size alone should not determine a creator's potential in this project.
    scores["creator_growth_score"] = (
        0.30 * scores["engagement_component"]
        + 0.25 * scores["growth_component"]
        + 0.20 * scores["activity_component"]
        + 0.15 * scores["campaign_component"]
        + 0.10 * scores["response_conversion_component"]
    ).clip(0, 100).round(2)

    # Project-defined priority categories, not official platform categories.
    scores["growth_score_band"] = pd.cut(
        scores["creator_growth_score"],
        bins=[float("-inf"), 40, 60, 80, float("inf")],
        labels=SCORE_BANDS,
        right=False,
    ).astype(str)
    scores["priority_reason"] = scores.apply(_priority_reason, axis=1)

    scores["is_high_growth_priority"] = scores["creator_growth_score"] >= 80
    scores["is_growth_opportunity"] = scores["creator_growth_score"].between(60, 80, inclusive="left")
    scores["needs_reactivation"] = (
        (scores["last_active_days"] >= 14) & (scores["creator_growth_score"] >= 50)
    )

    _validate_score_output(df, scores)
    return scores


def _validate_score_output(metric_data: pd.DataFrame, scores: pd.DataFrame) -> None:
    """Check that scoring preserves metrics and introduces only valid fields."""
    if len(scores) != len(metric_data):
        raise ValueError("Scoring output must keep the same number of creators.")
    if not scores["creator_id"].equals(metric_data["creator_id"]):
        raise ValueError("Scoring output must preserve creator_id values and order.")
    if not all(column in scores.columns for column in metric_data.columns):
        raise ValueError("Scoring output is missing one or more metric columns.")
    if not all(column in scores.columns for column in SCORE_COLUMNS):
        raise ValueError("Scoring output is missing one or more score columns.")
    if scores.isna().any().any():
        raise ValueError("Scoring introduced missing values.")
    if not scores["creator_growth_score"].between(0, 100).all():
        raise ValueError("creator_growth_score must stay between 0 and 100.")
    if not scores["growth_score_band"].isin(SCORE_BANDS).all():
        raise ValueError("growth_score_band contains an invalid project-defined value.")

    flag_columns = ["is_high_growth_priority", "is_growth_opportunity", "needs_reactivation"]
    for column in flag_columns:
        if not pd.api.types.is_bool_dtype(scores[column]):
            raise ValueError(f"Flag '{column}' must contain True or False values.")


def summarize_scores(df: pd.DataFrame) -> dict:
    """Return the concise score summary displayed by the command-line script."""
    band_counts = df["growth_score_band"].value_counts()
    return {
        "total_creators": len(df),
        "minimum_creator_growth_score": round(df["creator_growth_score"].min(), 2),
        "maximum_creator_growth_score": round(df["creator_growth_score"].max(), 2),
        "average_creator_growth_score": round(df["creator_growth_score"].mean(), 2),
        "high_potential_creators": int(band_counts.get("High Potential", 0)),
        "growth_potential_creators": int(band_counts.get("Growth Potential", 0)),
        "developing_creators": int(band_counts.get("Developing", 0)),
        "low_priority_creators": int(band_counts.get("Low Priority", 0)),
        "creators_flagged_for_reactivation": int(df["needs_reactivation"].sum()),
    }


def main() -> None:
    """Load calculated metrics, score creators, and save a new CSV file."""
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "processed" / "creator_metrics.csv"
    output_path = project_root / "data" / "processed" / "creator_scores.csv"

    try:
        metrics = pd.read_csv(input_path)
        scores = calculate_creator_growth_score(metrics)
        scores.to_csv(output_path, index=False)
        print("Creator Growth Scores calculated successfully.")
        print(f"Output file: {output_path}")
        for name, value in summarize_scores(scores).items():
            print(f"- {name}: {value}")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"SCORING FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
