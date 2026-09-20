"""Create optional AI insights and human-reviewed outreach drafts.

AI interprets existing synthetic signals only. Deterministic project rules still
decide selection, scoring, segments, eligibility, and reactivation candidates.
No message is sent automatically by this module.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd


try:
    from dotenv import load_dotenv
except ImportError:
    # The project works in prompt-only mode even before python-dotenv is installed.
    def load_dotenv() -> bool:
        return False


MATCH_COLUMNS = [
    "creator_id", "campaign_id", "campaign_name", "market", "language", "category",
    "followers", "avg_views", "engagement_rate", "follower_growth_30d", "response_rate",
    "conversion_rate", "creator_growth_score", "score_band", "primary_segment",
    "secondary_opportunity", "match_score", "recommendation_tier", "recommended_action",
    "match_reason",
]
METRIC_COLUMNS = [
    "creator_id", "videos_30d", "campaign_participation_90d", "last_active_days",
    "activity_score", "campaign_participation_score", "response_rate_pct",
    "is_high_engagement", "is_high_growth", "is_consistently_active",
]
CONTEXT_COLUMNS = MATCH_COLUMNS + [column for column in METRIC_COLUMNS if column != "creator_id"]
OUTPUT_COLUMNS = [
    "creator_id", "campaign_id", "campaign_name", "market", "language", "category",
    "creator_growth_score", "score_band", "primary_segment", "secondary_opportunity",
    "match_score", "recommendation_tier", "recommended_action", "growth_insight",
    "key_strengths", "growth_opportunities", "recommended_next_step", "outreach_subject",
    "outreach_message", "personalisation_points", "ai_generated", "human_review_required",
    "review_status", "generation_status", "raw_ai_response",
]


def load_environment() -> dict:
    """Load optional local settings without printing or storing the API key."""
    load_dotenv()
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("OPENAI_MODEL"),
    }


def _plain_value(value):
    """Convert pandas or NumPy scalar values into JSON-friendly Python values."""
    return value.item() if hasattr(value, "item") else value


def build_creator_context(row: pd.Series) -> dict:
    """Return only the existing fields needed to explain one eligible creator."""
    missing_columns = [column for column in CONTEXT_COLUMNS if column not in row.index]
    if missing_columns:
        raise ValueError(f"Creator row is missing context fields: {', '.join(missing_columns)}.")
    return {column: _plain_value(row[column]) for column in CONTEXT_COLUMNS}


def build_insight_prompt(context: dict) -> str:
    """Build a constrained prompt for an operational growth insight."""
    data = json.dumps(context, ensure_ascii=False, indent=2)
    return f"""You are assisting a creator community operations team in an independent synthetic-data portfolio project. Use only the structured data below.

Do not calculate, alter, or question scores. Do not invent facts, identities, age, gender, income, profession, personal history, or commercial outcomes. Do not describe any project score or segment as an official TikTok rating or benchmark. Use concise, actionable operational language.

Return valid JSON only, with exactly this shape:
{{
  "growth_insight": "one or two sentences",
  "key_strengths": ["maximum three short items"],
  "growth_opportunities": ["maximum three short items"],
  "recommended_next_step": "one short action"
}}

Creator data:
{data}"""


def build_outreach_prompt(context: dict) -> str:
    """Build a constrained prompt for a draft that a human must review."""
    data = json.dumps(context, ensure_ascii=False, indent=2)
    return f"""Create a professional, friendly, concise creator outreach draft for an independent synthetic-data portfolio project. Use only the campaign and creator fields below.

Do not invent benefits, payment, collaboration terms, deadlines, previous campaign participation, or personal information. Do not imply information came from another source. Do not use phrases such as "I noticed you are" or "I know that you". Prefer wording grounded in the supplied data, such as "Based on your recent content performance". This is a draft only: a human must review it before any use.

Return valid JSON only, with exactly this shape:
{{
  "subject": "short subject",
  "message": "80 to 150 word draft beginning with AI-generated draft — human review required.",
  "personalisation_points": ["maximum three factual points from supplied data"]
}}

Campaign and creator data:
{data}"""


def generate_fallback_insight(row: pd.Series) -> dict:
    """Create useful local insight text when AI is not requested or unavailable."""
    strengths = []
    opportunities = []
    if row["is_high_growth"]:
        strengths.append("Strong recent follower growth momentum")
    if row["is_high_engagement"]:
        strengths.append("Strong engagement relative to the project rules")
    if row["activity_score"] >= 60:
        strengths.append("Recent activity supports consistent content production")
    if row["campaign_participation_score"] < 50:
        opportunities.append("Campaign participation is an area for development")
    if row["activity_score"] < 60:
        opportunities.append("Content activity could be strengthened")
    if row["response_rate_pct"] < 70:
        opportunities.append("Response readiness could be improved")
    if not strengths:
        strengths.append("Eligible for this campaign based on deterministic matching rules")
    if not opportunities:
        opportunities.append("Maintain current performance signals through ongoing support")

    if row["is_high_growth"] and row["is_high_engagement"]:
        insight = "Creator shows strong recent growth and engagement signals."
    elif row["is_high_engagement"]:
        insight = "Creator has strong engagement signals, while growth can be monitored."
    elif row["is_high_growth"]:
        insight = "Creator shows strong recent follower growth momentum."
    else:
        insight = "Creator matches the campaign through the existing deterministic operational rules."

    return {
        "growth_insight": insight,
        "key_strengths": strengths[:3],
        "growth_opportunities": opportunities[:3],
        "recommended_next_step": f"Human review of the {row['recommended_action']} recommendation.",
    }


def _fallback_outreach(row: pd.Series) -> dict:
    """Create a safe local draft; it remains pending mandatory human review."""
    subject = f"Potential collaboration: {row['campaign_name']}"
    message = (
        "AI-generated draft — human review required.\n\n"
        f"Hello,\n\nBased on your recent content performance in the {row['category']} category "
        f"and the {row['market']} market, we are exploring a potential fit for the "
        f"{row['campaign_name']}. Your current campaign match signals suggest this may be "
        "worth discussing. We would be happy to share further details and understand whether "
        "there could be interest in a future collaboration. Please let us know if you would be "
        "open to reviewing a brief.\n\nBest regards"
    )
    points = [f"{row['market']} {row['category']} content", row["match_reason"]]
    if row["is_high_engagement"]:
        points.append("High engagement signal")
    return {"subject": subject, "message": message, "personalisation_points": points[:3]}


def call_ai(prompt: str, api_key: str, model: str) -> str:
    """Call the current OpenAI Python SDK only after explicit user opt-in."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=model, input=prompt)
    return response.output_text


def validate_ai_output(output: object, output_type: str) -> dict:
    """Parse and validate the small JSON structures requested from a model."""
    parsed = json.loads(output) if isinstance(output, str) else output
    if not isinstance(parsed, dict):
        raise ValueError("AI response must be a JSON object.")
    if output_type == "insight":
        required = {"growth_insight", "key_strengths", "growth_opportunities", "recommended_next_step"}
        list_fields = ["key_strengths", "growth_opportunities"]
    elif output_type == "outreach":
        required = {"subject", "message", "personalisation_points"}
        list_fields = ["personalisation_points"]
    else:
        raise ValueError("Unknown AI output type.")
    if set(parsed) != required:
        raise ValueError("AI response does not have the expected JSON keys.")
    if any(not isinstance(parsed[field], str) or not parsed[field].strip() for field in required - set(list_fields)):
        raise ValueError("AI response has an empty text field.")
    if any(not isinstance(parsed[field], list) or len(parsed[field]) > 3 for field in list_fields):
        raise ValueError("AI response has an invalid list field.")
    if any(not all(isinstance(item, str) and item.strip() for item in parsed[field]) for field in list_fields):
        raise ValueError("AI response lists must contain non-empty text items.")
    return parsed


def _make_output_row(row: pd.Series, insight: dict, outreach: dict, status: str, raw_response: str = "") -> dict:
    """Create one CSV-ready record with JSON-formatted list columns."""
    return {
        "creator_id": row["creator_id"], "campaign_id": row["campaign_id"],
        "campaign_name": row["campaign_name"], "market": row["market"],
        "language": row["language"], "category": row["category"],
        "creator_growth_score": row["creator_growth_score"], "score_band": row["score_band"],
        "primary_segment": row["primary_segment"], "secondary_opportunity": row["secondary_opportunity"],
        "match_score": row["match_score"], "recommendation_tier": row["recommendation_tier"],
        "recommended_action": row["recommended_action"], "growth_insight": insight["growth_insight"],
        "key_strengths": json.dumps(insight["key_strengths"], ensure_ascii=False),
        "growth_opportunities": json.dumps(insight["growth_opportunities"], ensure_ascii=False),
        "recommended_next_step": insight["recommended_next_step"], "outreach_subject": outreach["subject"],
        "outreach_message": outreach["message"],
        "personalisation_points": json.dumps(outreach["personalisation_points"], ensure_ascii=False),
        "ai_generated": True, "human_review_required": True, "review_status": "PENDING_REVIEW",
        "generation_status": status, "raw_ai_response": raw_response,
    }


def generate_creator_insights(
    df: pd.DataFrame, use_ai: bool = False, ai_caller=call_ai
) -> tuple[pd.DataFrame, list[dict]]:
    """Generate prompts and either API or deterministic content for eligible creators."""
    missing_columns = [column for column in CONTEXT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"AI input is missing required fields: {', '.join(missing_columns)}.")
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), []
    if df[CONTEXT_COLUMNS].isna().any().any():
        raise ValueError("AI input contains missing required values.")

    settings = load_environment()
    can_call_api = use_ai and bool(settings["api_key"]) and bool(settings["model"])
    output_rows = []
    prompt_records = []

    for _, row in df.iterrows():
        context = build_creator_context(row)
        insight_prompt = build_insight_prompt(context)
        outreach_prompt = build_outreach_prompt(context)
        prompt_records.append({
            "creator_id": row["creator_id"], "campaign_id": row["campaign_id"],
            "insight_prompt": insight_prompt, "outreach_prompt": outreach_prompt,
        })
        fallback_insight = generate_fallback_insight(row)
        fallback_outreach = _fallback_outreach(row)

        if not can_call_api:
            output_rows.append(_make_output_row(row, fallback_insight, fallback_outreach, "PROMPT_ONLY"))
            continue

        try:
            insight_raw = ai_caller(insight_prompt, settings["api_key"], settings["model"])
            outreach_raw = ai_caller(outreach_prompt, settings["api_key"], settings["model"])
            insight = validate_ai_output(insight_raw, "insight")
            outreach = validate_ai_output(outreach_raw, "outreach")
            # Enforce the visible safety label even if a model forgot to include it.
            if not outreach["message"].startswith("AI-generated draft — human review required"):
                outreach["message"] = "AI-generated draft — human review required.\n\n" + outreach["message"]
            output_rows.append(_make_output_row(row, insight, outreach, "AI_GENERATED"))
        except ValueError:
            raw_response = json.dumps({"insight": locals().get("insight_raw", ""), "outreach": locals().get("outreach_raw", "")})
            output_rows.append(_make_output_row(row, fallback_insight, fallback_outreach, "FAILED_VALIDATION", raw_response))
        except Exception:
            output_rows.append(_make_output_row(row, fallback_insight, fallback_outreach, "API_ERROR"))

    results = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    return results, prompt_records


def save_prompt_records(prompt_records: list[dict], output_path: Path) -> None:
    """Save one structured prompt record per line without ever including API keys."""
    with output_path.open("w", encoding="utf-8") as prompt_file:
        for record in prompt_records:
            prompt_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_ai_input(matches: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Enrich eligible matches with the existing metric fields needed for prompts."""
    missing_match_columns = [column for column in MATCH_COLUMNS if column not in matches.columns]
    missing_metric_columns = [column for column in METRIC_COLUMNS if column not in metrics.columns]
    if missing_match_columns:
        raise ValueError(f"campaign_matches data is missing: {', '.join(missing_match_columns)}.")
    if missing_metric_columns:
        raise ValueError(f"creator_metrics data is missing: {', '.join(missing_metric_columns)}.")
    if not matches["creator_id"].is_unique or matches["creator_id"].isna().any():
        raise ValueError("campaign_matches data has missing or duplicate creator_id values.")
    if not metrics["creator_id"].is_unique:
        raise ValueError("creator_metrics data has duplicate creator_id values.")
    if not set(matches["creator_id"]).issubset(set(metrics["creator_id"])):
        raise ValueError("campaign_matches contains creator_id values absent from creator_metrics.")

    enriched = matches[MATCH_COLUMNS].merge(
        metrics[METRIC_COLUMNS], on="creator_id", how="left", validate="one_to_one"
    )
    if len(enriched) != len(matches) or enriched.isna().any().any():
        raise ValueError("Unable to enrich campaign matches without missing values or row loss.")
    return enriched


def run_pipeline(use_ai: bool = False, limit: int | None = None) -> pd.DataFrame:
    """Create local prompt records and review-required insight drafts for matches."""
    project_root = Path(__file__).resolve().parents[1]
    matches = pd.read_csv(project_root / "data" / "processed" / "campaign_matches.csv")
    metrics = pd.read_csv(project_root / "data" / "processed" / "creator_metrics.csv")
    ai_input = load_ai_input(matches, metrics)
    if limit is not None:
        ai_input = ai_input.head(limit)
    results, prompt_records = generate_creator_insights(ai_input, use_ai=use_ai)
    results.to_csv(project_root / "data" / "processed" / "ai_creator_insights.csv", index=False)
    save_prompt_records(prompt_records, project_root / "data" / "processed" / "ai_prompts.jsonl")
    return results


def main() -> None:
    """Run prompt-only mode by default; API use requires the --use-ai option."""
    parser = argparse.ArgumentParser(description="Generate review-required creator insight drafts.")
    parser.add_argument("--use-ai", action="store_true", help="Call OpenAI only when key and model are configured.")
    parser.add_argument("--limit", type=int, help="Process only the first N eligible campaign matches.")
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 0:
        parser.error("--limit must be zero or a positive integer.")

    try:
        results = run_pipeline(use_ai=arguments.use_ai, limit=arguments.limit)
        print(f"Creator insight records: {len(results)}")
        print(f"Generation statuses: {results['generation_status'].value_counts().to_dict() if len(results) else {}}")
        if arguments.use_ai and not (load_environment()["api_key"] and load_environment()["model"]):
            print("OpenAI key or model is not configured; prompt-only fallback mode was used.")
        print("All outreach drafts are AI-generated drafts — human review required (PENDING_REVIEW).")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"AI INSIGHTS FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
