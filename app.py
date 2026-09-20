"""Interactive Streamlit dashboard for the synthetic creator operations project.

The dashboard reads already-calculated project results only. It does not rerun
business rules, call external APIs, send creator messages, or modify CSV files.
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_csv(path_string: str) -> pd.DataFrame:
    # Load one CSV once per Streamlit session; return an empty table if absent.
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data
def load_weekly_report(path_string: str) -> dict:
    # Load the JSON weekly report, or return an empty dictionary if unavailable.
    path = Path(path_string)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data
def load_ai_prompts(path_string: str) -> dict[str, dict]:
    # Load prompt records by creator ID without loading any environment variables.
    path = Path(path_string)
    if not path.exists():
        return {}
    prompts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            prompts[record["creator_id"]] = record
        except (json.JSONDecodeError, KeyError):
            continue
    return prompts


def load_data() -> dict:
    # Load all dashboard sources. Missing files are handled by page-level warnings.
    return {
        "raw": load_csv(str(PROJECT_ROOT / "data" / "raw" / "creators.csv")),
        "metrics": load_csv(str(PROCESSED_DIR / "creator_metrics.csv")),
        "scores": load_csv(str(PROCESSED_DIR / "creator_scores.csv")),
        "segments": load_csv(str(PROCESSED_DIR / "creator_segments.csv")),
        "matches": load_csv(str(PROCESSED_DIR / "campaign_matches.csv")),
        "reactivation": load_csv(str(PROCESSED_DIR / "reactivation_candidates.csv")),
        "ai_insights": load_csv(str(PROCESSED_DIR / "ai_creator_insights.csv")),
        "report": load_weekly_report(str(PROCESSED_DIR / "weekly_report.json")),
        "prompts": load_ai_prompts(str(PROCESSED_DIR / "ai_prompts.jsonl")),
    }


def require_data(df: pd.DataFrame, name: str) -> bool:
    # Display a friendly warning instead of a Python traceback for missing data.
    if df.empty:
        st.warning(f"{name} is missing or empty. Please run the local pipeline first.")
        return False
    return True


def build_creator_view(data: dict) -> pd.DataFrame:
    # Join existing display fields without recalculating any project logic.
    metrics = data["metrics"]
    scores = data["scores"]
    segments = data["segments"]
    if metrics.empty or scores.empty or segments.empty:
        return pd.DataFrame()

    metric_columns = [
        "creator_id", "market", "language", "category", "followers", "avg_views",
        "engagement_rate_pct", "growth_rate_pct", "videos_30d",
        "campaign_participation_90d", "response_rate_pct", "conversion_rate_pct",
        "last_active_days", "activity_score",
    ]
    score_columns = ["creator_id", "creator_growth_score", "growth_score_band", "priority_reason"]
    segment_columns = ["creator_id", "primary_segment", "secondary_opportunity", "segment_reason"]
    if not all(column in metrics.columns for column in metric_columns):
        return pd.DataFrame()
    if not all(column in scores.columns for column in score_columns):
        return pd.DataFrame()
    if not all(column in segments.columns for column in segment_columns):
        return pd.DataFrame()

    creator_view = metrics[metric_columns].merge(scores[score_columns], on="creator_id", how="inner")
    creator_view = creator_view.merge(segments[segment_columns], on="creator_id", how="inner")
    return creator_view.rename(columns={"growth_score_band": "score_band"})


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    # Apply only filters whose fields exist in the supplied display table.
    filtered = df.copy()
    for column, selected_value in [
        ("market", filters["market"]),
        ("category", filters["category"]),
        ("primary_segment", filters["segment"]),
    ]:
        if selected_value != "All" and column in filtered.columns:
            filtered = filtered.loc[filtered[column] == selected_value]
    return filtered


def render_sidebar(data: dict) -> tuple[str, dict]:
    # Render navigation and filters derived from current project data.
    st.sidebar.header("Dashboard")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Creator Segmentation", "Campaign Matching", "Reactivation", "AI Review"],
    )
    creator_view = build_creator_view(data)
    markets = ["All"] + sorted(creator_view["market"].dropna().unique().tolist()) if "market" in creator_view else ["All"]
    categories = ["All"] + sorted(creator_view["category"].dropna().unique().tolist()) if "category" in creator_view else ["All"]
    segments = ["All"] + sorted(creator_view["primary_segment"].dropna().unique().tolist()) if "primary_segment" in creator_view else ["All"]
    filters = {
        "market": st.sidebar.selectbox("Market", markets),
        "category": st.sidebar.selectbox("Category", categories),
        "segment": st.sidebar.selectbox("Primary segment", segments),
    }

    matches = data["matches"]
    if not matches.empty and {"campaign_id", "campaign_name"}.issubset(matches.columns):
        campaign_options = matches[["campaign_id", "campaign_name"]].drop_duplicates()
        labels = {
            f"{row.campaign_id} — {row.campaign_name}": row.campaign_id
            for _, row in campaign_options.iterrows()
        }
        selected_label = st.sidebar.selectbox("Campaign", list(labels))
        filters["campaign_id"] = labels[selected_label]
    else:
        filters["campaign_id"] = None
        st.sidebar.caption("No campaign file is currently available.")
    return page, filters


def distribution_chart(distribution: dict, title: str, label: str) -> None:
    # Render a compact Altair bar chart from a JSON report distribution.
    chart_data = pd.DataFrame({label: list(distribution.keys()), "Creators": list(distribution.values())})
    if chart_data.empty:
        st.info("No data available for this chart.")
        return
    chart = alt.Chart(chart_data).mark_bar(color="#4C78A8").encode(
        x=alt.X(f"{label}:N", sort=None, title=None),
        y=alt.Y("Creators:Q", title="Creators"),
        tooltip=[label, "Creators"],
    ).properties(title=title, height=260)
    st.altair_chart(chart, width="stretch")


def render_overview(data: dict, filters: dict) -> None:
    # Render the high-level recruiter-friendly operations overview.
    st.header("Overview")
    report = data["report"]
    if not report:
        st.warning("weekly_report.json is missing. Please run src/reporting.py first.")
        return
    kpi = report.get("kpi_summary", {})
    metric_items = [
        ("Total Creators", kpi.get("total_creators", 0)),
        ("High Potential", kpi.get("high_potential_creators", 0)),
        ("Growth Opportunity", kpi.get("growth_opportunity_creators", 0)),
        ("Campaign Eligible", kpi.get("campaign_eligible_creators", 0)),
        ("Reactivation Candidates", kpi.get("reengagement_candidates", 0)),
        ("AI Drafts Pending Review", kpi.get("ai_pending_review", 0)),
    ]
    columns = st.columns(6)
    for column, (label, value) in zip(columns, metric_items):
        column.metric(label, value)

    st.subheader("Community Snapshot")
    snapshot = report.get("community_snapshot", {})
    left, middle, right = st.columns(3)
    with left:
        distribution_chart(snapshot.get("score_band_distribution", {}), "Creator Growth Score Bands", "Score Band")
    with middle:
        distribution_chart(snapshot.get("primary_segment_distribution", {}), "Primary Segments", "Segment")
    with right:
        creators = build_creator_view(data)
        if not creators.empty:
            market_counts = creators["market"].value_counts().sort_index().to_dict()
            distribution_chart(market_counts, "Market Distribution", "Market")

    st.subheader("Top Creator Opportunities")
    creators = apply_filters(build_creator_view(data), filters)
    if require_data(creators, "Creator display data"):
        top_columns = [
            "creator_id", "market", "category", "creator_growth_score", "score_band",
            "primary_segment", "secondary_opportunity",
        ]
        st.dataframe(
            creators.sort_values("creator_growth_score", ascending=False)[top_columns].head(10),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Campaign Snapshot")
    matches = data["matches"]
    if filters["campaign_id"] is not None and "campaign_id" in matches:
        matches = matches.loc[matches["campaign_id"] == filters["campaign_id"]]
    if matches.empty:
        st.info("No eligible campaign matches are available.")
        return
    summary = {
        "Eligible Creators": len(matches),
        "Strong Match": int((matches["recommendation_tier"] == "Strong Match").sum()),
        "Good Match": int((matches["recommendation_tier"] == "Good Match").sum()),
        "Potential Match": int((matches["recommendation_tier"] == "Potential Match").sum()),
        "Average Match Score": f"{matches['match_score'].mean():.2f}",
    }
    st.caption(f"Campaign: {matches.iloc[0]['campaign_name']}")
    summary_columns = st.columns(5)
    for column, (label, value) in zip(summary_columns, summary.items()):
        column.metric(label, value)
    match_columns = ["creator_id", "market", "category", "match_score", "recommendation_tier", "recommended_action"]
    st.dataframe(matches.sort_values("match_score", ascending=False)[match_columns].head(5), hide_index=True, width="stretch")


def render_creator_detail(creator: pd.Series, data: dict) -> None:
   # Render one creator's existing metrics, scores, segment, and related records.
    st.subheader("Creator Overview")
    labels = [
        ("Followers", creator["followers"]), ("Average Views", creator["avg_views"]),
        ("Engagement Rate", f"{creator['engagement_rate_pct']:.2f}%"),
        ("Growth Rate", f"{creator['growth_rate_pct']:.2f}%"),
        ("Videos (30d)", creator["videos_30d"]), ("Campaigns (90d)", creator["campaign_participation_90d"]),
        ("Response Rate", f"{creator['response_rate_pct']:.2f}%"), ("Conversion Rate", f"{creator['conversion_rate_pct']:.2f}%"),
        ("Last Active Days", creator["last_active_days"]),
    ]
    metric_columns = st.columns(3)
    for index, (label, value) in enumerate(labels):
        metric_columns[index % 3].metric(label, value)
    st.markdown(
        f"**Creator Growth Score:** {creator['creator_growth_score']:.2f}  \n**Score Band:** {creator['score_band']}  \n**Primary Segment:** {creator['primary_segment']}  \n**Secondary Opportunity:** {creator['secondary_opportunity']}"
    )
    st.caption(f"Priority Reason: {creator['priority_reason']}")
    st.caption(f"Segment Reason: {creator['segment_reason']}")

    matches = data["matches"]
    creator_matches = matches.loc[matches["creator_id"] == creator["creator_id"]] if "creator_id" in matches else pd.DataFrame()
    if not creator_matches.empty:
        match = creator_matches.iloc[0]
        st.subheader("Campaign Match")
        st.write(f"**Match Score:** {match['match_score']:.2f} | **Tier:** {match['recommendation_tier']} | **Action:** {match['recommended_action']}")
        st.caption(f"Match Reason: {match['match_reason']}")

    insights = data["ai_insights"]
    creator_insights = insights.loc[insights["creator_id"] == creator["creator_id"]] if "creator_id" in insights else pd.DataFrame()
    if not creator_insights.empty:
        insight = creator_insights.iloc[0]
        st.subheader("AI Insight")
        st.warning("AI-assisted draft — human review required")
        st.write(insight["growth_insight"])
        st.write("**Key Strengths:**", _read_json_list(insight["key_strengths"]))
        st.write("**Growth Opportunities:**", _read_json_list(insight["growth_opportunities"]))
        st.write("**Recommended Next Step:**", insight["recommended_next_step"])


def render_segmentation(data: dict, filters: dict) -> None:
    # Render segment distribution, a filterable creator table, and a detail selector.
    st.header("Creator Segmentation")
    creators = build_creator_view(data)
    if not require_data(creators, "Creator segmentation data"):
        return
    st.metric("Creator Population", len(creators))
    filtered = apply_filters(creators, filters).sort_values("creator_growth_score", ascending=False)
    st.metric("Filtered Creators", len(filtered))
    if not filtered.empty:
        distribution_chart(
            filtered["primary_segment"].value_counts().to_dict(),
            "Filtered Creator Segments",
            "Segment",
        )
    display_columns = [
        "creator_id", "market", "language", "category", "followers", "avg_views",
        "engagement_rate_pct", "growth_rate_pct", "activity_score", "creator_growth_score",
        "score_band", "primary_segment", "secondary_opportunity",
    ]
    st.dataframe(filtered[display_columns], hide_index=True, width="stretch")
    if filtered.empty:
        st.info("No creators match the current filters.")
        return
    creator_id = st.selectbox("Creator ID", filtered["creator_id"].tolist())
    render_creator_detail(filtered.loc[filtered["creator_id"] == creator_id].iloc[0], data)


def render_campaign_matching(data: dict, filters: dict) -> None:
   # Render the selected campaign's existing match results without recalculation.
    st.header("Campaign Matching")
    matches = data["matches"]
    if not require_data(matches, "Campaign matching data"):
        return
    if filters["campaign_id"] is not None:
        matches = matches.loc[matches["campaign_id"] == filters["campaign_id"]]
    matches = apply_filters(matches, filters)
    if matches.empty:
        st.info("No eligible creators match the selected campaign and filters.")
        return
    st.subheader("Selected Campaign")
    st.write(f"**Campaign ID:** {matches.iloc[0]['campaign_id']}  \n**Campaign Name:** {matches.iloc[0]['campaign_name']}")
    summary_columns = st.columns(5)
    summary_values = [
        ("Eligible Creators", len(matches)),
        ("Strong Match", int((matches["recommendation_tier"] == "Strong Match").sum())),
        ("Good Match", int((matches["recommendation_tier"] == "Good Match").sum())),
        ("Potential Match", int((matches["recommendation_tier"] == "Potential Match").sum())),
        ("Average Match Score", f"{matches['match_score'].mean():.2f}"),
    ]
    for column, (label, value) in zip(summary_columns, summary_values):
        column.metric(label, value)
    minimum_score = st.slider("Minimum match score", 0, 100, 0)
    tier = st.selectbox("Recommendation tier", ["All"] + sorted(matches["recommendation_tier"].unique().tolist()))
    shown = matches.loc[matches["match_score"] >= minimum_score]
    if tier != "All":
        shown = shown.loc[shown["recommendation_tier"] == tier]
    shown = shown.sort_values("match_score", ascending=False)
    columns = [
        "creator_id", "market", "category", "followers", "engagement_rate", "follower_growth_30d",
        "creator_growth_score", "primary_segment", "match_score", "recommendation_tier", "recommended_action",
    ]
    st.dataframe(shown[columns], hide_index=True, width="stretch")
    if shown.empty:
        st.info("No campaign matches meet this score threshold and tier filter.")
        return
    creator_id = st.selectbox("View campaign match for creator", shown["creator_id"].tolist())
    match = shown.loc[shown["creator_id"] == creator_id].iloc[0]
    st.subheader("Match Detail")
    st.metric("Match Score", f"{match['match_score']:.2f}")
    component_columns = [
        "market_match_score", "category_match_score", "segment_fit_score",
        "engagement_fit_score", "growth_fit_score", "response_fit_score",
    ]
    if all(column in match.index for column in component_columns):
        st.dataframe(pd.DataFrame({"Component": component_columns, "Score": [match[column] for column in component_columns]}), hide_index=True, width="stretch")
    st.write(f"**Match Reason:** {match['match_reason']}")
    st.write(f"**Recommended Action:** {match['recommended_action']}")


def render_reactivation(data: dict, filters: dict) -> None:
    # Render a useful empty state or the existing reactivation candidate list.
    st.header("Creator Reactivation")
    candidates = apply_filters(data["reactivation"], filters)
    if candidates.empty:
        st.info("No reactivation candidates found in the current synthetic dataset.")
        st.caption("The current synthetic dataset does not contain creators meeting the defined reactivation criteria.")
        return
    metrics = st.columns(4)
    metrics[0].metric("Total Candidates", len(candidates))
    metrics[1].metric("High Priority", int((candidates["reactivation_priority"] == "High").sum()))
    metrics[2].metric("Medium Priority", int((candidates["reactivation_priority"] == "Medium").sum()))
    metrics[3].metric("Average Reactivation Score", f"{candidates['reactivation_score'].mean():.2f}")
    columns = [
        "creator_id", "market", "category", "last_active_days", "creator_growth_score",
        "reactivation_score", "reactivation_priority", "reactivation_action", "reactivation_reason",
    ]
    st.dataframe(candidates.sort_values("reactivation_score", ascending=False)[columns], hide_index=True, width="stretch")


def _read_json_list(value: object) -> list[str]:
    # Read JSON list fields from CSV safely for display.
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def render_ai_review(data: dict, filters: dict) -> None:
    # Show draft content and a session-only human review demonstration control.
    st.header("AI-Assisted Creator Insights & Outreach Review")
    insights = apply_filters(data["ai_insights"], filters)
    if not require_data(insights, "AI insight data"):
        return
    summary = st.columns(7)
    values = [
        ("AI Records", len(insights)), ("AI-generated", int(insights["ai_generated"].sum())),
        ("Prompt-only", int((insights["generation_status"] == "PROMPT_ONLY").sum())),
        ("Pending Review", int((insights["review_status"] == "PENDING_REVIEW").sum())),
        ("Approved", int((insights["review_status"] == "APPROVED").sum())),
        ("Rejected", int((insights["review_status"] == "REJECTED").sum())),
        ("API Errors", int((insights["generation_status"] == "API_ERROR").sum())),
    ]
    for column, (label, value) in zip(summary, values):
        column.metric(label, value)
    pending = insights.loc[insights["review_status"] == "PENDING_REVIEW"]
    st.caption(f"Pending review drafts: {len(pending)}")
    creator_id = st.selectbox("Creator draft", insights["creator_id"].tolist())
    insight = insights.loc[insights["creator_id"] == creator_id].iloc[0]
    st.warning("AI-assisted draft — human review required")
    st.subheader("Creator Context")
    st.write({
        "market": insight["market"], "language": insight["language"], "category": insight["category"],
        "creator_growth_score": insight["creator_growth_score"], "match_score": insight["match_score"],
        "primary_segment": insight["primary_segment"],
    })
    st.subheader("Growth Insight")
    st.write(insight["growth_insight"])
    st.write("**Key Strengths:**", _read_json_list(insight["key_strengths"]))
    st.write("**Growth Opportunities:**", _read_json_list(insight["growth_opportunities"]))
    st.write("**Recommended Next Step:**", insight["recommended_next_step"])
    st.subheader("Creator Outreach Draft")
    st.write(f"**Subject:** {insight['outreach_subject']}")
    st.text_area("Message", insight["outreach_message"], height=220, disabled=True)
    st.write("**Personalisation Points:**", _read_json_list(insight["personalisation_points"]))
    st.caption(f"Generation Status: {insight['generation_status']} | Review Status: {insight['review_status']}")

    review_key = f"demo_review_{creator_id}"
    if review_key not in st.session_state:
        st.session_state[review_key] = insight["review_status"]
    st.selectbox("Demo-only review decision", ["PENDING_REVIEW", "APPROVED", "REJECTED"], key=review_key)
    st.info("This demo control is stored only for this browser session. It does not send or approve real messages, and it does not modify any CSV file.")
    prompt_record = data["prompts"].get(creator_id)
    if prompt_record:
        with st.expander("View Prompt Used"):
            st.text_area("Insight Prompt", prompt_record.get("insight_prompt", ""), height=240, disabled=True)
            st.text_area("Outreach Prompt", prompt_record.get("outreach_prompt", ""), height=240, disabled=True)


def render_project_notes() -> None:
    # Render a compact safety and methodology note at the bottom of every page.
    st.divider()
    st.subheader("Project Notes")
    st.caption(
        "Creator data is synthetic. Scores and segments are deterministic project-defined rules. "
        "Campaign matching uses transparent weighted rules. AI is used only for interpretation and draft communication. "
        "Creator-facing AI outputs require human review. No automated messaging is enabled. "
        "This independent portfolio project is not affiliated with TikTok."
    )


def main() -> None:
    # Configure and render the interactive dashboard from existing local results.
    st.set_page_config(page_title="Creator Community Growth Copilot", page_icon="", layout="wide")
    st.title("AI-Powered Creator Community Growth Copilot")
    st.caption("Creator community growth, campaign matching and AI-assisted operations dashboard")
    st.info("Synthetic portfolio project using deterministic creator analytics and AI-assisted workflow. AI-assisted creator communication drafts require human review. ")
    data = load_data()
    page, filters = render_sidebar(data)
    if page == "Overview":
        render_overview(data, filters)
    elif page == "Creator Segmentation":
        render_segmentation(data, filters)
    elif page == "Campaign Matching":
        render_campaign_matching(data, filters)
    elif page == "Reactivation":
        render_reactivation(data, filters)
    else:
        render_ai_review(data, filters)
    render_project_notes()


if __name__ == "__main__":
    main()
