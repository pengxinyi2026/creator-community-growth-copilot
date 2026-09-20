"""Create deterministic, project-defined creator operations segments.

These are independent portfolio-project groups, not official TikTok categories.
They use existing synthetic metrics and scores only; no AI or ML is used.
"""

from pathlib import Path
import sys

import pandas as pd


PRIMARY_SEGMENTS = {
    "High Potential",
    "Re-engagement",
    "Growth Opportunity",
    "Consistent Active",
    "Other / Developing",
}
PRIMARY_SEGMENT_ORDER = [
    "High Potential",
    "Re-engagement",
    "Growth Opportunity",
    "Consistent Active",
    "Other / Developing",
]
SECONDARY_OPPORTUNITIES = {
    "Growth Support",
    "Campaign Activation",
    "Re-engagement",
    "Community Nurturing",
    "Monitor",
}
SECONDARY_OPPORTUNITY_ORDER = [
    "Growth Support",
    "Campaign Activation",
    "Re-engagement",
    "Community Nurturing",
    "Monitor",
]

SCORE_COLUMNS = [
    "creator_id",
    "creator_growth_score",
    "growth_score_band",
    "is_high_growth_priority",
    "is_growth_opportunity",
    "needs_reactivation",
]
METRIC_COLUMNS = [
    "creator_id",
    "market",
    "language",
    "category",
    "is_high_engagement",
    "is_high_growth",
    "is_consistently_active",
    "activity_score",
    "campaign_participation_score",
    "response_rate_pct",
    "conversion_rate_pct",
    "last_active_days",
    "growth_rate_pct",
]
SEGMENT_COLUMNS = [
    "creator_id",
    "market",
    "language",
    "category",
    "creator_growth_score",
    "score_band",
    "primary_segment",
    "segment_reason",
    "secondary_opportunity",
    "is_high_growth_priority",
    "is_growth_opportunity",
    "needs_reactivation",
    "is_high_engagement",
    "is_high_growth",
    "is_consistently_active",
    "activity_score",
    "campaign_participation_score",
    "response_rate_pct",
    "conversion_rate_pct",
    "last_active_days",
]


def _check_creator_ids(df: pd.DataFrame, table_name: str) -> None:
    """Raise a clear error for missing, empty, or duplicate creator IDs."""
    if "creator_id" not in df.columns:
        raise ValueError(f"{table_name} is missing the creator_id column.")
    if df["creator_id"].isna().any():
        raise ValueError(f"{table_name} contains missing creator_id values.")
    if df["creator_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{table_name} contains empty creator_id values.")
    duplicate_ids = df.loc[df["creator_id"].duplicated(keep=False), "creator_id"].unique()
    if len(duplicate_ids):
        raise ValueError(f"{table_name} contains duplicate creator_id values: {', '.join(duplicate_ids)}.")


def combine_creator_data(scores: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Check and combine the score and metric tables using their creator IDs."""
    missing_score_columns = [column for column in SCORE_COLUMNS if column not in scores.columns]
    missing_metric_columns = [column for column in METRIC_COLUMNS if column not in metrics.columns]
    if missing_score_columns:
        raise ValueError(f"creator_scores data is missing: {', '.join(missing_score_columns)}.")
    if missing_metric_columns:
        raise ValueError(f"creator_metrics data is missing: {', '.join(missing_metric_columns)}.")

    _check_creator_ids(scores, "creator_scores data")
    _check_creator_ids(metrics, "creator_metrics data")

    score_ids = set(scores["creator_id"])
    metric_ids = set(metrics["creator_id"])
    if score_ids != metric_ids:
        only_in_scores = sorted(score_ids - metric_ids)
        only_in_metrics = sorted(metric_ids - score_ids)
        raise ValueError(
            "creator_id values do not match between files. "
            f"Only in scores: {only_in_scores}. Only in metrics: {only_in_metrics}."
        )

    combined = scores[SCORE_COLUMNS].merge(
        metrics[METRIC_COLUMNS], on="creator_id", how="inner", validate="one_to_one"
    )
    if combined.isna().any().any():
        raise ValueError("Combined score and metric data contains missing values.")
    return combined


def _primary_segment(row: pd.Series) -> str:
    """Assign one mutually exclusive segment using the documented priority order."""
    if row["creator_growth_score"] >= 80:
        return "High Potential"
    if row["needs_reactivation"]:
        return "Re-engagement"
    if 60 <= row["creator_growth_score"] < 80:
        return "Growth Opportunity"
    if row["is_consistently_active"]:
        return "Consistent Active"
    return "Other / Developing"


def _segment_reason(row: pd.Series) -> str:
    """Create one short, data-based operations explanation for a creator."""
    segment = row["primary_segment"]
    if segment == "High Potential":
        if row["is_high_engagement"] and row["is_high_growth"]:
            return "High-priority creator with strong engagement and growth momentum."
        if row["activity_score"] >= 75:
            return "High-priority creator with strong overall signals and activity."
        return "High-priority creator with strong overall growth signals."
    if segment == "Re-engagement":
        if row["is_high_engagement"]:
            return "Recent inactivity despite strong engagement suggests re-engagement potential."
        return "Recent inactivity with sufficient score suggests re-engagement potential."
    if segment == "Growth Opportunity":
        if row["campaign_participation_score"] < 40:
            return "Growth signals are strong; campaign participation can be developed."
        if row["is_high_growth"]:
            return "High recent growth indicates a clear opportunity for support."
        return "Good overall potential with room to strengthen operational signals."
    if segment == "Consistent Active":
        if row["is_high_engagement"]:
            return "Consistent activity and strong engagement support ongoing community work."
        return "Consistent recent activity supports ongoing community engagement."
    if row["last_active_days"] >= 14:
        return "Limited current signals and recent inactivity call for monitoring."
    return "Limited current signals; monitor engagement, growth, and activity changes."


def _secondary_opportunity(row: pd.Series) -> str:
    """Choose one next action using fixed, documented priority rules."""
    segment = row["primary_segment"]
    if segment == "High Potential":
        if row["campaign_participation_score"] < 50:
            return "Campaign Activation"
        if row["activity_score"] < 60:
            return "Growth Support"
        return "Community Nurturing"
    if segment == "Re-engagement":
        return "Re-engagement"
    if segment == "Growth Opportunity":
        # High growth takes priority over limited campaign participation here.
        if row["growth_rate_pct"] >= 10 or row["is_high_growth"]:
            return "Growth Support"
        if row["campaign_participation_score"] < 40:
            return "Campaign Activation"
        return "Community Nurturing"
    if segment == "Consistent Active":
        return "Campaign Activation"
    return "Monitor"


def segment_creators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a concise creator operations table with segments and next actions."""
    required_columns = set(SCORE_COLUMNS + METRIC_COLUMNS)
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Data for segmentation is missing: {', '.join(missing_columns)}.")
    _check_creator_ids(df, "Segmentation data")
    if df[list(required_columns)].isna().any().any():
        raise ValueError("Segmentation data contains missing values.")

    segments = df.copy()
    segments["primary_segment"] = segments.apply(_primary_segment, axis=1)
    segments["segment_reason"] = segments.apply(_segment_reason, axis=1)
    segments["secondary_opportunity"] = segments.apply(_secondary_opportunity, axis=1)
    segments = segments.rename(columns={"growth_score_band": "score_band"})
    segments = segments[SEGMENT_COLUMNS].copy()
    _validate_segment_output(segments)
    return segments


def _validate_segment_output(segments: pd.DataFrame, expected_rows: int | None = None) -> None:
    """Validate segment coverage, allowed values, uniqueness, and empty fields."""
    if expected_rows is not None and len(segments) != expected_rows:
        raise ValueError(f"Expected {expected_rows} segmented creators, found {len(segments)}.")
    if not segments["creator_id"].is_unique:
        raise ValueError("Segment output contains duplicate creator_id values.")
    if segments["primary_segment"].isna().any() or segments["secondary_opportunity"].isna().any():
        raise ValueError("Segment output has missing primary or secondary segment values.")
    if not segments["primary_segment"].isin(PRIMARY_SEGMENTS).all():
        raise ValueError("Segment output contains an invalid primary_segment value.")
    if not segments["secondary_opportunity"].isin(SECONDARY_OPPORTUNITIES).all():
        raise ValueError("Segment output contains an invalid secondary_opportunity value.")


def print_segment_summary(segments: pd.DataFrame, output_path: Path) -> None:
    """Print the requested concise counts for the command-line workflow."""
    print("Creator segmentation completed successfully.")
    print("Primary segment counts:")
    primary_counts = segments["primary_segment"].value_counts().reindex(PRIMARY_SEGMENT_ORDER, fill_value=0)
    print(primary_counts.to_string())
    print("Secondary opportunity counts:")
    secondary_counts = segments["secondary_opportunity"].value_counts().reindex(SECONDARY_OPPORTUNITY_ORDER, fill_value=0)
    print(secondary_counts.to_string())
    print("Overlap or missing primary segments: none (one rule order per creator).")
    print(f"Output file: {output_path}")


def main() -> None:
    """Load score and metric CSVs, segment creators, and save the new output."""
    project_root = Path(__file__).resolve().parents[1]
    scores_path = project_root / "data" / "processed" / "creator_scores.csv"
    metrics_path = project_root / "data" / "processed" / "creator_metrics.csv"
    output_path = project_root / "data" / "processed" / "creator_segments.csv"

    try:
        scores = pd.read_csv(scores_path)
        metrics = pd.read_csv(metrics_path)
        combined = combine_creator_data(scores, metrics)
        segments = segment_creators(combined)
        _validate_segment_output(segments)
        segments.to_csv(output_path, index=False)
        print_segment_summary(segments, output_path)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"SEGMENTATION FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
