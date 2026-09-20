"""Identify synthetic creators with transparent reactivation potential.

This independent portfolio-project module uses deterministic local rules only.
It is not an official TikTok system and does not use AI, APIs, or ML.
"""

from pathlib import Path
import sys

import pandas as pd


METRIC_COLUMNS = [
    "creator_id", "market", "language", "category", "followers", "avg_views",
    "engagement_rate", "follower_growth_30d", "videos_30d",
    "campaign_participation_90d", "response_rate", "conversion_rate",
    "last_active_days", "activity_score", "campaign_participation_score",
    "response_rate_pct", "is_high_engagement", "is_high_growth",
]
SCORE_COLUMNS = [
    "creator_id", "creator_growth_score", "growth_score_band",
    "engagement_component", "growth_component",
]
SEGMENT_COLUMNS = ["creator_id", "primary_segment", "secondary_opportunity"]

PRIORITIES = {"High", "Medium"}
ACTIONS = {"Personalised Outreach", "Campaign Re-invitation", "Community Nudge", "Monitor"}
OUTPUT_COLUMNS = [
    "creator_id", "market", "language", "category", "followers", "avg_views",
    "engagement_rate", "follower_growth_30d", "videos_30d",
    "campaign_participation_90d", "response_rate", "conversion_rate", "last_active_days",
    "creator_growth_score", "score_band", "primary_segment", "secondary_opportunity",
    "activity_score", "campaign_participation_score", "response_rate_pct",
    "reactivation_score", "reactivation_priority", "reactivation_action",
    "reactivation_reason", "inactivity_signal", "growth_signal", "engagement_signal",
    "activity_signal", "response_signal", "campaign_signal",
]


def _check_ids(df: pd.DataFrame, table_name: str) -> None:
    """Check that a source table has one non-empty ID per creator."""
    if df["creator_id"].isna().any() or df["creator_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{table_name} contains missing or empty creator_id values.")
    if not df["creator_id"].is_unique:
        raise ValueError(f"{table_name} contains duplicate creator_id values.")


def combine_reactivation_data(
    metrics: pd.DataFrame, scores: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    """Validate and join the three inputs without losing any creator rows."""
    sources = [
        (metrics, METRIC_COLUMNS, "creator_metrics data"),
        (scores, SCORE_COLUMNS, "creator_scores data"),
        (segments, SEGMENT_COLUMNS, "creator_segments data"),
    ]
    for df, required_columns, table_name in sources:
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"{table_name} is missing: {', '.join(missing_columns)}.")
        _check_ids(df, table_name)

    metric_ids = set(metrics["creator_id"])
    if metric_ids != set(scores["creator_id"]) or metric_ids != set(segments["creator_id"]):
        raise ValueError("creator_id values do not match across metrics, scores, and segments files.")

    combined = metrics[METRIC_COLUMNS].merge(
        scores[SCORE_COLUMNS], on="creator_id", how="inner", validate="one_to_one"
    ).merge(segments[SEGMENT_COLUMNS], on="creator_id", how="inner", validate="one_to_one")
    if len(combined) != len(metrics):
        raise ValueError("Combining reactivation inputs caused unexpected row loss.")
    if combined.isna().any().any():
        raise ValueError("Reactivation input data contains missing required values.")
    return combined


def calculate_reactivation_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent 0–100 reactivation components and their weighted score."""
    scored = df.copy()
    # Growth and engagement components were already normalised by scoring.py.
    # Activity is used as evidence of recent activity history, not as a penalty
    # for current inactivity; candidates have already passed an inactivity rule.
    scored["reactivation_score"] = (
        0.30 * scored["growth_component"].clip(0, 100)
        + 0.25 * scored["engagement_component"].clip(0, 100)
        + 0.20 * scored["response_rate_pct"].clip(0, 100)
        + 0.15 * scored["campaign_participation_score"].clip(0, 100)
        + 0.10 * scored["activity_score"].clip(0, 100)
    ).clip(0, 100).round(2)
    return scored


def assign_reactivation_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Add signals and High/Medium priority using the documented fixed rules."""
    prioritized = df.copy()
    prioritized["inactivity_signal"] = prioritized["last_active_days"] >= 14
    prioritized["growth_signal"] = prioritized["is_high_growth"]
    prioritized["engagement_signal"] = prioritized["is_high_engagement"]
    prioritized["activity_signal"] = prioritized["activity_score"] >= 60
    prioritized["response_signal"] = prioritized["response_rate_pct"] >= 70
    prioritized["campaign_signal"] = prioritized["campaign_participation_score"] >= 40

    high_priority = (
        (prioritized["last_active_days"] >= 21)
        & (prioritized["creator_growth_score"] >= 60)
        & (
            prioritized["is_high_engagement"]
            | prioritized["is_high_growth"]
            | (prioritized["response_rate"] >= 0.70)
        )
    )
    base_candidate = (
        (prioritized["last_active_days"] >= 14)
        & (prioritized["creator_growth_score"] >= 50)
    )
    prioritized["reactivation_priority"] = pd.NA
    prioritized.loc[base_candidate, "reactivation_priority"] = "Medium"
    prioritized.loc[high_priority, "reactivation_priority"] = "High"
    return prioritized


def generate_reactivation_reason(row: pd.Series) -> str:
    """Return one concise, deterministic explanation built from actual signals."""
    signals = []
    if row["engagement_signal"]:
        signals.append("strong engagement")
    if row["growth_signal"]:
        signals.append("high growth")
    if row["response_signal"]:
        signals.append("strong response readiness")
    if row["campaign_signal"]:
        signals.append("campaign history")
    signal_text = " and ".join(signals[:2]) if signals else "meaningful growth potential"
    return f"Inactive for {int(row['last_active_days'])} days but retains {signal_text}."


def assign_reactivation_action(row: pd.Series) -> str:
    """Choose one action using the specified conflict-resolution priority order."""
    if row["reactivation_priority"] == "High" and row["campaign_participation_score"] >= 40:
        return "Campaign Re-invitation"
    if row["reactivation_priority"] == "High" and row["response_rate_pct"] >= 70:
        return "Personalised Outreach"
    if row["reactivation_priority"] == "Medium" and row["is_high_engagement"]:
        return "Community Nudge"
    return "Monitor"


def identify_reactivation_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return only High or Medium reactivation candidates, sorted by priority."""
    required_columns = set(METRIC_COLUMNS + SCORE_COLUMNS + SEGMENT_COLUMNS)
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Data for reactivation is missing: {', '.join(missing_columns)}.")
    _check_ids(df, "Reactivation data")
    if df[list(required_columns)].isna().any().any():
        raise ValueError("Reactivation data contains missing required values.")

    candidates = calculate_reactivation_score(df)
    candidates = assign_reactivation_priority(candidates)
    candidates = candidates.loc[candidates["reactivation_priority"].notna()].copy()
    if candidates.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    candidates["reactivation_action"] = candidates.apply(assign_reactivation_action, axis=1)
    candidates["reactivation_reason"] = candidates.apply(generate_reactivation_reason, axis=1)
    candidates = candidates.rename(columns={"growth_score_band": "score_band"})
    priority_order = {"High": 0, "Medium": 1}
    candidates["_priority_order"] = candidates["reactivation_priority"].map(priority_order)
    candidates = candidates.sort_values(
        ["_priority_order", "reactivation_score"], ascending=[True, False]
    ).drop(columns="_priority_order")
    candidates = candidates[OUTPUT_COLUMNS].reset_index(drop=True)
    _validate_candidate_output(candidates)
    return candidates


def _validate_candidate_output(candidates: pd.DataFrame) -> None:
    """Validate a candidate list, including a valid empty result."""
    if list(candidates.columns) != OUTPUT_COLUMNS:
        raise ValueError("Reactivation output does not contain the expected columns.")
    if candidates.empty:
        return
    if not candidates["creator_id"].is_unique or candidates["creator_id"].isna().any():
        raise ValueError("Reactivation candidates have missing or duplicate creator_id values.")
    required_non_missing = [
        "reactivation_score", "reactivation_priority", "reactivation_action", "reactivation_reason",
    ]
    if candidates[required_non_missing].isna().any().any():
        raise ValueError("Reactivation candidates have missing key output values.")
    if not candidates["reactivation_score"].between(0, 100).all():
        raise ValueError("reactivation_score must stay between 0 and 100.")
    if not candidates["reactivation_priority"].isin(PRIORITIES).all():
        raise ValueError("Reactivation output contains an invalid priority.")
    if not candidates["reactivation_action"].isin(ACTIONS).all():
        raise ValueError("Reactivation output contains an invalid action.")
    base_candidate = (
        (candidates["last_active_days"] >= 14)
        & (candidates["creator_growth_score"] >= 50)
    )
    if not base_candidate.all():
        raise ValueError("Every output row must meet the base reactivation candidate rule.")


def print_reactivation_summary(total_creators: int, candidates: pd.DataFrame, output_path: Path) -> None:
    """Print the requested concise operational summary."""
    candidate_count = len(candidates)
    print(f"Total creators: {total_creators}")
    print(f"Reactivation candidates: {candidate_count}")
    print(f"High priority: {int((candidates['reactivation_priority'] == 'High').sum()) if candidate_count else 0}")
    print(f"Medium priority: {int((candidates['reactivation_priority'] == 'Medium').sum()) if candidate_count else 0}")
    print(f"Candidate percentage: {candidate_count / total_creators * 100:.2f}%")
    if candidates.empty:
        print("No reactivation candidates found in the current synthetic dataset.")
    else:
        print(f"Average reactivation score: {candidates['reactivation_score'].mean():.2f}")
        print("Top 5 candidates:")
        print(candidates[["creator_id", "reactivation_priority", "reactivation_score", "reactivation_action"]].head(5).to_string(index=False))
    print(f"Output file: {output_path}")


def run_reactivation_pipeline() -> pd.DataFrame:
    """Load all inputs, identify candidates, save the output, and return it."""
    project_root = Path(__file__).resolve().parents[1]
    metrics = pd.read_csv(project_root / "data" / "processed" / "creator_metrics.csv")
    scores = pd.read_csv(project_root / "data" / "processed" / "creator_scores.csv")
    segments = pd.read_csv(project_root / "data" / "processed" / "creator_segments.csv")
    candidates = identify_reactivation_candidates(combine_reactivation_data(metrics, scores, segments))
    output_path = project_root / "data" / "processed" / "reactivation_candidates.csv"
    candidates.to_csv(output_path, index=False)
    print_reactivation_summary(len(metrics), candidates, output_path)
    return candidates


def main() -> None:
    """Run the local, deterministic reactivation workflow from the command line."""
    try:
        run_reactivation_pipeline()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"REACTIVATION FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
