"""Match synthetic creators to a synthetic campaign using transparent rules.

This independent portfolio-project tool is not an official TikTok system. It
uses only local synthetic data and deterministic rules—no APIs, AI, or ML.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


ALLOWED_MARKETS = {"UK", "Germany", "France", "Italy", "Spain", "Poland", "Netherlands", "Belgium"}
ALLOWED_CATEGORIES = {"Beauty", "Fashion", "Lifestyle", "Fitness", "Food", "Technology"}
ALLOWED_OBJECTIVES = {"Awareness", "Engagement", "Conversion", "Creator Activation"}
ALLOWED_SEGMENTS = {
    "High Potential", "Growth Opportunity", "Consistent Active", "Re-engagement", "Other / Developing"
}
RECOMMENDATION_TIERS = {"Strong Match", "Good Match", "Potential Match", "Low Match"}
RECOMMENDED_ACTIONS = {"Priority Outreach", "Campaign Invitation", "Nurture", "Monitor"}

SEGMENT_FIT_SCORES = {
    "High Potential": 100,
    "Growth Opportunity": 90,
    "Consistent Active": 75,
    "Re-engagement": 50,
    "Other / Developing": 30,
}

SEGMENT_INPUT_COLUMNS = [
    "creator_id", "market", "language", "category", "creator_growth_score", "score_band",
    "primary_segment", "secondary_opportunity", "is_high_growth_priority",
    "is_growth_opportunity", "needs_reactivation", "is_high_engagement",
    "is_high_growth", "is_consistently_active", "activity_score",
    "campaign_participation_score", "response_rate_pct", "conversion_rate_pct",
    "last_active_days",
]
SCORE_INPUT_COLUMNS = [
    "creator_id", "followers", "avg_views", "engagement_rate", "follower_growth_30d",
    "response_rate", "conversion_rate",
]
MATCH_OUTPUT_COLUMNS = [
    "campaign_id", "campaign_name", "creator_id", "market", "language", "category",
    "followers", "avg_views", "engagement_rate", "follower_growth_30d", "response_rate",
    "conversion_rate", "creator_growth_score", "score_band", "primary_segment",
    "secondary_opportunity", "market_match_score", "category_match_score", "segment_fit_score",
    "engagement_fit_score", "growth_fit_score", "response_fit_score", "match_score",
    "recommendation_tier", "recommended_action", "match_reason", "is_eligible",
]


@dataclass(frozen=True)
class CampaignBrief:
    """Small, readable data structure describing a synthetic campaign need."""

    campaign_id: str
    campaign_name: str
    market: str
    category: str
    objective: str
    min_followers: int
    max_followers: int
    preferred_min_engagement: float
    preferred_min_growth: float
    min_response_rate: float
    preferred_segments: list[str]


def create_default_campaign() -> CampaignBrief:
    """Return the project's fictional example campaign, not a platform standard."""
    return CampaignBrief(
        campaign_id="CAMP-001",
        campaign_name="EU Beauty Growth Campaign",
        market="UK",
        category="Beauty",
        objective="Engagement",
        min_followers=5_000,
        max_followers=300_000,
        preferred_min_engagement=0.06,
        preferred_min_growth=0.05,
        min_response_rate=0.70,
        preferred_segments=["High Potential", "Growth Opportunity", "Consistent Active"],
    )


def _validate_campaign(campaign: CampaignBrief) -> None:
    """Check a campaign brief before using it in matching calculations."""
    if campaign.market not in ALLOWED_MARKETS:
        raise ValueError(f"Campaign market '{campaign.market}' is not allowed.")
    if campaign.category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Campaign category '{campaign.category}' is not allowed.")
    if campaign.objective not in ALLOWED_OBJECTIVES:
        raise ValueError(f"Campaign objective '{campaign.objective}' is not allowed.")
    if campaign.min_followers < 0 or campaign.max_followers < campaign.min_followers:
        raise ValueError("Campaign follower limits are invalid.")
    if any(segment not in ALLOWED_SEGMENTS for segment in campaign.preferred_segments):
        raise ValueError("Campaign contains an invalid preferred segment.")


def _check_unique_ids(df: pd.DataFrame, table_name: str) -> None:
    """Check creator IDs before joining two input tables."""
    if df["creator_id"].isna().any() or df["creator_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{table_name} contains missing or empty creator_id values.")
    if not df["creator_id"].is_unique:
        raise ValueError(f"{table_name} contains duplicate creator_id values.")


def load_creator_data(segments: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Combine the segment table with source fields required by campaign output."""
    missing_segments = [column for column in SEGMENT_INPUT_COLUMNS if column not in segments.columns]
    missing_scores = [column for column in SCORE_INPUT_COLUMNS if column not in scores.columns]
    if missing_segments:
        raise ValueError(f"creator_segments data is missing: {', '.join(missing_segments)}.")
    if missing_scores:
        raise ValueError(f"creator_scores data is missing: {', '.join(missing_scores)}.")
    _check_unique_ids(segments, "creator_segments data")
    _check_unique_ids(scores, "creator_scores data")

    segment_ids = set(segments["creator_id"])
    score_ids = set(scores["creator_id"])
    if segment_ids != score_ids:
        raise ValueError("creator_id values do not match between creator_segments and creator_scores.")

    creators = segments[SEGMENT_INPUT_COLUMNS].merge(
        scores[SCORE_INPUT_COLUMNS], on="creator_id", how="inner", validate="one_to_one"
    )
    if creators.isna().any().any():
        raise ValueError("Combined creator data contains missing values.")
    return creators


def _fit_against_minimum(values: pd.Series, minimum: float) -> pd.Series:
    """Return 100 when a goal is met, otherwise a proportional 0–100 score."""
    if minimum <= 0:
        return pd.Series(100.0, index=values.index)
    return (values / minimum * 100).clip(0, 100)


def calculate_match_scores(creators: pd.DataFrame, campaign: CampaignBrief) -> pd.DataFrame:
    """Calculate transparent matching components and eligibility for every creator."""
    _validate_campaign(campaign)
    required = set(SEGMENT_INPUT_COLUMNS + SCORE_INPUT_COLUMNS)
    missing_columns = sorted(required - set(creators.columns))
    if missing_columns:
        raise ValueError(f"Creator data is missing fields for matching: {', '.join(missing_columns)}.")

    matches = creators.copy()
    matches["market_match_score"] = (matches["market"] == campaign.market).astype(int) * 100
    matches["category_match_score"] = (matches["category"] == campaign.category).astype(int) * 100
    matches["segment_fit_score"] = matches["primary_segment"].map(SEGMENT_FIT_SCORES).fillna(0)
    matches["engagement_fit_score"] = _fit_against_minimum(
        matches["engagement_rate"], campaign.preferred_min_engagement
    )
    matches["growth_fit_score"] = _fit_against_minimum(
        matches["follower_growth_30d"], campaign.preferred_min_growth
    )
    matches["response_fit_score"] = _fit_against_minimum(
        matches["response_rate"], campaign.min_response_rate
    )

    # Market and category each carry 25%; the remaining signals rank eligible creators.
    matches["match_score"] = (
        0.25 * matches["market_match_score"]
        + 0.25 * matches["category_match_score"]
        + 0.15 * matches["segment_fit_score"]
        + 0.15 * matches["engagement_fit_score"]
        + 0.10 * matches["growth_fit_score"]
        + 0.10 * matches["response_fit_score"]
    ).clip(0, 100).round(2)

    matches["is_eligible"] = (
        (matches["market"] == campaign.market)
        & (matches["category"] == campaign.category)
        & (matches["followers"] >= campaign.min_followers)
        & (matches["followers"] <= campaign.max_followers)
    )
    return matches


def generate_match_reason(row: pd.Series) -> str:
    """Create a one-sentence, deterministic explanation for an eligible match."""
    signals = []
    if row["primary_segment"] == "High Potential":
        signals.append("high-priority growth signals")
    elif row["primary_segment"] == "Growth Opportunity":
        signals.append("growth potential")
    elif row["primary_segment"] == "Consistent Active":
        signals.append("consistent activity")
    if row["is_high_engagement"]:
        signals.append("high engagement")
    if row["is_high_growth"]:
        signals.append("strong growth momentum")
    if row["response_rate"] >= 0.70:
        signals.append("strong response rate")
    if row["campaign_participation_score"] >= 50:
        signals.append("campaign readiness")
    if not signals:
        signals.append("moderate performance signals")
    return f"{row['market']} {row['category']} creator with {', '.join(signals[:3])}."


def _recommendation_tier(score: float) -> str:
    """Return the project-defined operational tier for a numeric match score."""
    if score >= 90:
        return "Strong Match"
    if score >= 75:
        return "Good Match"
    if score >= 60:
        return "Potential Match"
    return "Low Match"


def _recommended_action(row: pd.Series) -> str:
    """Choose one simple next action from the tier and creator segment."""
    if row["primary_segment"] == "High Potential" and row["match_score"] >= 90:
        return "Priority Outreach"
    if row["match_score"] >= 75:
        return "Campaign Invitation"
    if row["match_score"] >= 60:
        return "Nurture"
    return "Monitor"


def match_creators(creators: pd.DataFrame, campaign: CampaignBrief) -> pd.DataFrame:
    """Return eligible creators ranked from highest to lowest campaign match score."""
    matches = calculate_match_scores(creators, campaign)
    eligible_matches = matches.loc[matches["is_eligible"]].copy()
    if eligible_matches.empty:
        return pd.DataFrame(columns=MATCH_OUTPUT_COLUMNS)

    eligible_matches["campaign_id"] = campaign.campaign_id
    eligible_matches["campaign_name"] = campaign.campaign_name
    eligible_matches["recommendation_tier"] = eligible_matches["match_score"].apply(_recommendation_tier)
    eligible_matches["recommended_action"] = eligible_matches.apply(_recommended_action, axis=1)
    eligible_matches["match_reason"] = eligible_matches.apply(generate_match_reason, axis=1)
    eligible_matches = eligible_matches.sort_values(
        ["match_score", "creator_growth_score"], ascending=[False, False]
    )
    eligible_matches = eligible_matches[MATCH_OUTPUT_COLUMNS].reset_index(drop=True)
    _validate_match_output(eligible_matches)
    return eligible_matches


def _validate_match_output(matches: pd.DataFrame) -> None:
    """Validate a saved recommendation table, including an empty valid result."""
    if list(matches.columns) != MATCH_OUTPUT_COLUMNS:
        raise ValueError("Campaign match output does not contain the expected columns.")
    if matches.empty:
        return
    if not matches["creator_id"].is_unique or matches["creator_id"].isna().any():
        raise ValueError("Campaign match output has missing or duplicate creator_id values.")
    required_non_missing = ["match_score", "recommendation_tier", "recommended_action", "match_reason"]
    if matches[required_non_missing].isna().any().any():
        raise ValueError("Campaign match output has missing recommendation values.")
    if not matches["match_score"].between(0, 100).all():
        raise ValueError("Campaign match scores must stay between 0 and 100.")
    if not matches["recommendation_tier"].isin(RECOMMENDATION_TIERS).all():
        raise ValueError("Campaign match output contains an invalid recommendation tier.")
    if not matches["recommended_action"].isin(RECOMMENDED_ACTIONS).all():
        raise ValueError("Campaign match output contains an invalid recommended action.")
    if not matches["is_eligible"].all():
        raise ValueError("Only eligible creators may appear in campaign match output.")


def print_match_summary(campaign: CampaignBrief, total_creators: int, matches: pd.DataFrame) -> None:
    """Print a concise operations summary and up to ten ranked recommendations."""
    print(f"Campaign: {campaign.campaign_name}")
    print(f"Total creators: {total_creators}")
    print(f"Eligible creators: {len(matches)}")
    for tier in ["Strong Match", "Good Match", "Potential Match", "Low Match"]:
        count = int((matches["recommendation_tier"] == tier).sum()) if not matches.empty else 0
        print(f"{tier}: {count}")
    print("Top 10 recommended creators:")
    if matches.empty:
        print("No eligible creators found for this synthetic campaign.")
    else:
        print(matches[["creator_id", "match_score", "recommendation_tier", "recommended_action"]].head(10).to_string(index=False))


def main() -> None:
    """Load local synthetic data, match it to the default campaign, and save CSV."""
    project_root = Path(__file__).resolve().parents[1]
    segments_path = project_root / "data" / "processed" / "creator_segments.csv"
    scores_path = project_root / "data" / "processed" / "creator_scores.csv"
    output_path = project_root / "data" / "processed" / "campaign_matches.csv"

    try:
        segments = pd.read_csv(segments_path)
        scores = pd.read_csv(scores_path)
        creators = load_creator_data(segments, scores)
        campaign = create_default_campaign()
        matches = match_creators(creators, campaign)
        matches.to_csv(output_path, index=False)
        print_match_summary(campaign, len(creators), matches)
        print(f"Output file: {output_path}")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"MATCHING FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
