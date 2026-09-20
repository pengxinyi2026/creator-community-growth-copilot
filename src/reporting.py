"""Build a local weekly report from existing synthetic creator operations data.

This module only summarises prior calculations. It does not redefine project
rules, call AI, or make claims about real creators or official TikTok KPIs.
"""

from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd


SCORE_BAND_ORDER = ["Low Priority", "Developing", "Growth Potential", "High Potential"]
SEGMENT_ORDER = ["High Potential", "Re-engagement", "Growth Opportunity", "Consistent Active", "Other / Developing"]


def _require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    """Raise a clear error when a report input is missing an expected field."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{table_name} is missing: {', '.join(missing_columns)}.")


def _check_unique_creator_ids(df: pd.DataFrame, table_name: str) -> None:
    """Check a creator table uses one non-empty ID per row."""
    _require_columns(df, ["creator_id"], table_name)
    if df["creator_id"].isna().any() or df["creator_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{table_name} contains missing or empty creator_id values.")
    if not df["creator_id"].is_unique:
        raise ValueError(f"{table_name} contains duplicate creator_id values.")


def validate_report_inputs(data: dict[str, pd.DataFrame]) -> None:
    """Validate IDs and minimal schemas without changing any source DataFrame."""
    creators = data["creators"]
    metrics = data["metrics"]
    scores = data["scores"]
    segments = data["segments"]
    for df, name in [(creators, "creators"), (metrics, "metrics"), (scores, "scores"), (segments, "segments")]:
        _check_unique_creator_ids(df, name)

    population_ids = set(creators["creator_id"])
    for df, name in [(metrics, "metrics"), (scores, "scores"), (segments, "segments")]:
        if set(df["creator_id"]) != population_ids:
            raise ValueError(f"creator_id values in {name} do not match the creator population.")

    _require_columns(scores, ["creator_growth_score", "growth_score_band", "engagement_rate_pct", "growth_rate_pct", "response_rate_pct"], "scores")
    _require_columns(segments, ["primary_segment", "secondary_opportunity", "activity_score"], "segments")
    _require_columns(metrics, ["engagement_rate", "follower_growth_30d", "activity_score", "is_recently_inactive"], "metrics")
    _require_columns(data["matches"], ["campaign_id", "campaign_name", "match_score", "recommendation_tier"], "campaign matches")
    _require_columns(data["reactivation"], ["reactivation_priority", "reactivation_score"], "reactivation candidates")
    _require_columns(data["ai_insights"], ["generation_status", "review_status", "ai_generated"], "AI insights")

    # These are legitimate subsets, but any present IDs must be from the known population.
    for key, name in [("matches", "campaign matches"), ("reactivation", "reactivation candidates"), ("ai_insights", "AI insights")]:
        df = data[key]
        _check_unique_creator_ids(df, name)
        if not set(df["creator_id"]).issubset(population_ids):
            raise ValueError(f"{name} contains a creator_id outside the creator population.")


def build_kpi_summary(
    creators: pd.DataFrame, scores: pd.DataFrame, segments: pd.DataFrame,
    matches: pd.DataFrame, reactivation: pd.DataFrame, ai_insights: pd.DataFrame,
) -> dict:
    """Return the core factual weekly metrics as JSON-safe Python values."""
    return {
        "total_creators": int(len(creators)),
        "high_potential_creators": int((segments["primary_segment"] == "High Potential").sum()),
        "growth_opportunity_creators": int((segments["primary_segment"] == "Growth Opportunity").sum()),
        "consistent_active_creators": int((segments["primary_segment"] == "Consistent Active").sum()),
        "reengagement_candidates": int(len(reactivation)),
        "high_reactivation_candidates": int((reactivation["reactivation_priority"] == "High").sum()),
        "average_creator_growth_score": round(float(scores["creator_growth_score"].mean()), 2),
        "median_creator_growth_score": round(float(scores["creator_growth_score"].median()), 2),
        "campaign_eligible_creators": int(len(matches)),
        "strong_campaign_matches": int((matches["recommendation_tier"] == "Strong Match").sum()),
        "good_campaign_matches": int((matches["recommendation_tier"] == "Good Match").sum()),
        "potential_campaign_matches": int((matches["recommendation_tier"] == "Potential Match").sum()),
        "ai_insight_creators": int(len(ai_insights)),
        "ai_pending_review": int((ai_insights["review_status"] == "PENDING_REVIEW").sum()),
        "ai_failed_generation": int(ai_insights["generation_status"].isin(["API_ERROR", "FAILED_VALIDATION"]).sum()),
    }


def build_community_snapshot(scores: pd.DataFrame, segments: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    """Return factual distributions and averages for a Streamlit-friendly snapshot."""
    score_distribution = scores["growth_score_band"].value_counts().reindex(SCORE_BAND_ORDER, fill_value=0)
    segment_distribution = segments["primary_segment"].value_counts().reindex(SEGMENT_ORDER, fill_value=0)
    return {
        "total_creators": int(len(metrics)),
        "score_band_distribution": {name: int(value) for name, value in score_distribution.items()},
        "primary_segment_distribution": {name: int(value) for name, value in segment_distribution.items()},
        "average_engagement_rate": round(float(metrics["engagement_rate"].mean()), 4),
        "average_follower_growth": round(float(metrics["follower_growth_30d"].mean()), 4),
        "average_activity_score": round(float(metrics["activity_score"].mean()), 2),
        "recently_inactive_count": int(metrics["is_recently_inactive"].sum()),
    }


def get_top_creator_opportunities(scores: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Return the top ten existing scores for display, without creating a new rank."""
    score_columns = [
        "creator_id", "market", "category", "creator_growth_score", "growth_score_band",
        "engagement_rate_pct", "growth_rate_pct", "response_rate_pct",
    ]
    segment_columns = ["creator_id", "primary_segment", "secondary_opportunity", "activity_score"]
    _require_columns(scores, score_columns, "scores")
    _require_columns(segments, segment_columns, "segments")
    opportunities = scores[score_columns].merge(
        segments[segment_columns], on="creator_id", how="inner", validate="one_to_one"
    )
    if len(opportunities) != len(scores):
        raise ValueError("Top creator opportunities lost rows while joining existing data.")
    opportunities = opportunities.rename(columns={"growth_score_band": "score_band"})
    columns = [
        "creator_id", "market", "category", "creator_growth_score", "score_band",
        "primary_segment", "secondary_opportunity", "activity_score", "engagement_rate_pct",
        "growth_rate_pct", "response_rate_pct",
    ]
    return opportunities[columns].sort_values("creator_growth_score", ascending=False).head(10).reset_index(drop=True)


def build_campaign_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarise one or more campaigns, including a valid empty summary."""
    columns = [
        "campaign_id", "campaign_name", "total_eligible_creators", "strong_matches",
        "good_matches", "potential_matches", "top_match_score", "average_match_score", "top_creator_ids",
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (campaign_id, campaign_name), group in matches.groupby(["campaign_id", "campaign_name"], sort=False):
        top_ids = group.sort_values("match_score", ascending=False)["creator_id"].head(5).tolist()
        rows.append({
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "total_eligible_creators": int(len(group)),
            "strong_matches": int((group["recommendation_tier"] == "Strong Match").sum()),
            "good_matches": int((group["recommendation_tier"] == "Good Match").sum()),
            "potential_matches": int((group["recommendation_tier"] == "Potential Match").sum()),
            "top_match_score": round(float(group["match_score"].max()), 2),
            "average_match_score": round(float(group["match_score"].mean()), 2),
            "top_creator_ids": top_ids,
        })
    return pd.DataFrame(rows, columns=columns)


def build_reactivation_summary(reactivation: pd.DataFrame) -> dict:
    """Summarise reactivation candidates without requiring that any exist."""
    if reactivation.empty:
        return {
            "total_candidates": 0, "high_priority": 0, "medium_priority": 0,
            "average_reactivation_score": 0, "top_candidates": [],
            "message": "No reactivation candidates found in the current synthetic dataset.",
        }
    top_columns = ["creator_id", "reactivation_priority", "reactivation_score", "reactivation_action"]
    top_candidates = reactivation.sort_values("reactivation_score", ascending=False)[top_columns].head(5)
    return {
        "total_candidates": int(len(reactivation)),
        "high_priority": int((reactivation["reactivation_priority"] == "High").sum()),
        "medium_priority": int((reactivation["reactivation_priority"] == "Medium").sum()),
        "average_reactivation_score": round(float(reactivation["reactivation_score"].mean()), 2),
        "top_candidates": top_candidates.to_dict(orient="records"),
        "message": "Reactivation candidates are listed by existing project rules.",
    }


def build_ai_review_summary(ai_insights: pd.DataFrame) -> dict:
    """Count generation and human-review statuses without assuming API use."""
    return {
        "total_ai_records": int(len(ai_insights)),
        "ai_generated_count": int(ai_insights["ai_generated"].sum()),
        "prompt_only_count": int((ai_insights["generation_status"] == "PROMPT_ONLY").sum()),
        "api_error_count": int((ai_insights["generation_status"] == "API_ERROR").sum()),
        "failed_validation_count": int((ai_insights["generation_status"] == "FAILED_VALIDATION").sum()),
        "pending_review_count": int((ai_insights["review_status"] == "PENDING_REVIEW").sum()),
        "approved_count": int((ai_insights["review_status"] == "APPROVED").sum()),
        "rejected_count": int((ai_insights["review_status"] == "REJECTED").sum()),
    }


def generate_operational_priorities(kpi_summary: dict, reactivation_summary: dict) -> list[str]:
    """Return five factual observations and clearly labelled follow-up suggestions."""
    priorities = [
        f"{kpi_summary['campaign_eligible_creators']} creators are currently eligible for campaign matching.",
        f"{kpi_summary['high_potential_creators']} creators are currently in the High Potential segment.",
        f"{kpi_summary['growth_opportunity_creators']} creators are currently in the Growth Opportunity segment.",
    ]
    if reactivation_summary["total_candidates"] == 0:
        priorities.append("No reactivation candidates meet the current reactivation criteria.")
    else:
        priorities.append(
            f"Suggested operational follow-up: review {reactivation_summary['total_candidates']} reactivation candidates."
        )
    priorities.append(
        f"Suggested operational follow-up: review {kpi_summary['ai_pending_review']} creator outreach drafts pending human review."
    )
    return priorities


def build_report_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return DataFrames that a future Streamlit dashboard can use directly."""
    kpi = build_kpi_summary(
        data["creators"], data["scores"], data["segments"], data["matches"],
        data["reactivation"], data["ai_insights"],
    )
    ai_review = build_ai_review_summary(data["ai_insights"])
    return {
        "kpi_summary_df": pd.DataFrame(kpi.items(), columns=["metric", "value"]),
        "top_creator_opportunities_df": get_top_creator_opportunities(data["scores"], data["segments"]),
        "campaign_summary_df": build_campaign_summary(data["matches"]),
        "reactivation_candidates_df": data["reactivation"].copy(),
        "ai_review_summary_df": pd.DataFrame(ai_review.items(), columns=["metric", "count"]),
    }


def build_weekly_report(data: dict[str, pd.DataFrame], report_date: str | None = None) -> dict:
    """Build one JSON-safe weekly report object from already calculated data."""
    validate_report_inputs(data)
    kpi = build_kpi_summary(
        data["creators"], data["scores"], data["segments"], data["matches"],
        data["reactivation"], data["ai_insights"],
    )
    snapshot = build_community_snapshot(data["scores"], data["segments"], data["metrics"])
    opportunities = get_top_creator_opportunities(data["scores"], data["segments"])
    campaign_summary = build_campaign_summary(data["matches"])
    reactivation_summary = build_reactivation_summary(data["reactivation"])
    ai_review_summary = build_ai_review_summary(data["ai_insights"])
    return {
        "report_title": "Weekly Creator Community Operations Report",
        "report_date": report_date or date.today().isoformat(),
        "kpi_summary": kpi,
        "community_snapshot": snapshot,
        "top_creator_opportunities": opportunities.to_dict(orient="records"),
        "campaign_summary": campaign_summary.to_dict(orient="records"),
        "reactivation_summary": reactivation_summary,
        "ai_review_summary": ai_review_summary,
        "operational_priorities": generate_operational_priorities(kpi, reactivation_summary),
    }


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Build a simple Markdown table without adding another dependency."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_markdown_report(report: dict) -> str:
    """Turn the structured report into a concise, portfolio-friendly Markdown file."""
    kpi_rows = [[key.replace("_", " ").title(), value] for key, value in report["kpi_summary"].items()]
    snapshot = report["community_snapshot"]
    top_rows = [
        [row["creator_id"], row["market"], row["category"], row["creator_growth_score"], row["primary_segment"], row["secondary_opportunity"]]
        for row in report["top_creator_opportunities"]
    ]
    campaign_rows = [
        [row["campaign_name"], row["total_eligible_creators"], row["strong_matches"], row["good_matches"], row["potential_matches"], row["average_match_score"]]
        for row in report["campaign_summary"]
    ]
    ai_rows = [[key.replace("_", " ").title(), value] for key, value in report["ai_review_summary"].items()]
    campaign_section = _markdown_table(
        ["Campaign", "Eligible", "Strong Match", "Good Match", "Potential Match", "Avg Match Score"], campaign_rows
    ) if campaign_rows else "No campaign matches found in the current synthetic dataset."

    lines = [
        "# Weekly Creator Community Operations Report",
        "",
        f"Report date: {report['report_date']}",
        "",
        "## 1. KPI Summary",
        "",
        _markdown_table(["Metric", "Value"], kpi_rows),
        "",
        "## 2. Community Snapshot",
        "",
        f"- Creator population: {snapshot['total_creators']}",
        f"- Score band distribution: {snapshot['score_band_distribution']}",
        f"- Primary segment distribution: {snapshot['primary_segment_distribution']}",
        f"- Average engagement rate: {snapshot['average_engagement_rate']}",
        f"- Average follower growth: {snapshot['average_follower_growth']}",
        f"- Average activity score: {snapshot['average_activity_score']}",
        f"- Recently inactive creators: {snapshot['recently_inactive_count']}",
        "",
        "## 3. Top Creator Opportunities",
        "",
        _markdown_table(["Creator", "Market", "Category", "Growth Score", "Segment", "Suggested Opportunity"], top_rows),
        "",
        "## 4. Campaign Summary",
        "",
        campaign_section,
        "",
        "## 5. Reactivation",
        "",
        report["reactivation_summary"]["message"],
        "",
        "## 6. AI Review Status",
        "",
        _markdown_table(["Metric", "Count"], ai_rows),
        "",
        "## 7. Suggested Operational Follow-up",
        "",
    ]
    lines.extend(f"- {priority}" for priority in report["operational_priorities"])
    return "\n".join(lines) + "\n"


def load_project_data(project_root: Path) -> dict[str, pd.DataFrame]:
    """Load all local inputs needed for a weekly report."""
    processed = project_root / "data" / "processed"
    return {
        "creators": pd.read_csv(project_root / "data" / "raw" / "creators.csv"),
        "metrics": pd.read_csv(processed / "creator_metrics.csv"),
        "scores": pd.read_csv(processed / "creator_scores.csv"),
        "segments": pd.read_csv(processed / "creator_segments.csv"),
        "matches": pd.read_csv(processed / "campaign_matches.csv"),
        "reactivation": pd.read_csv(processed / "reactivation_candidates.csv"),
        "ai_insights": pd.read_csv(processed / "ai_creator_insights.csv"),
    }


def run_reporting_pipeline() -> tuple[dict, dict[str, pd.DataFrame]]:
    """Create JSON and Markdown report files, then return report and display tables."""
    project_root = Path(__file__).resolve().parents[1]
    data = load_project_data(project_root)
    report = build_weekly_report(data)
    tables = build_report_tables(data)
    processed = project_root / "data" / "processed"
    with (processed / "weekly_report.json").open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, ensure_ascii=False, indent=2)
    (processed / "weekly_report.md").write_text(build_markdown_report(report), encoding="utf-8")
    return report, tables


def main() -> None:
    """Run the local weekly-report workflow and print the requested console summary."""
    try:
        report, _ = run_reporting_pipeline()
        kpi = report["kpi_summary"]
        print(report["report_title"])
        print(f"Total creators: {kpi['total_creators']}")
        print(f"High Potential: {kpi['high_potential_creators']}")
        print(f"Growth Opportunity: {kpi['growth_opportunity_creators']}")
        print(f"Reactivation candidates: {kpi['reengagement_candidates']}")
        print(f"Campaign eligible creators: {kpi['campaign_eligible_creators']}")
        print(f"AI records: {kpi['ai_insight_creators']}")
        print(f"Pending review: {kpi['ai_pending_review']}")
        print("Top operational priorities:")
        for number, priority in enumerate(report["operational_priorities"], start=1):
            print(f"{number}. {priority}")
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        print(f"REPORTING FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
