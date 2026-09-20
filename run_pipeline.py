"""One-command runner for the creator community growth workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_stage(
    stage_number: int,
    total_stages: int,
    label: str,
    script: str,
    args: list[str] | None = None,
) -> None:
    """Run one project module and stop immediately if it fails."""
    command = [sys.executable, str(PROJECT_ROOT / script)]
    if args:
        command.extend(args)

    print()
    print("=" * 72)
    print(f"Stage {stage_number}/{total_stages}: {label}")
    print("=" * 72)
    print("Command:", " ".join(command))

    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print()
        print(f"PIPELINE FAILED at Stage {stage_number}/{total_stages}: {label}")
        raise SystemExit(result.returncode)

    print(f"Stage {stage_number}/{total_stages} completed successfully.")


def verify_outputs(paths: list[str]) -> None:
    """Verify that expected pipeline outputs exist."""
    missing = [path for path in paths if not (PROJECT_ROOT / path).exists()]

    if missing:
        print()
        print("PIPELINE FAILED: expected output files are missing:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the creator community growth workflow end to end."
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Regenerate the 300-row synthetic creator dataset before processing.",
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Attempt OpenAI API generation for AI insights.",
    )
    parser.add_argument(
        "--ai-limit",
        type=int,
        default=None,
        help="Process only the first N eligible campaign matches for AI insights.",
    )

    args = parser.parse_args()

    if args.ai_limit is not None and args.ai_limit < 0:
        parser.error("--ai-limit must be zero or a positive integer.")

    print("=" * 72)
    print("Creator Community Growth Copilot")
    print("One-command pipeline")
    print("=" * 72)

    if args.generate_synthetic:
        print()
        print("Synthetic mode: ON")
        print("The raw creators.csv file will be regenerated.")
        run_stage(
            1,
            9,
            "Generate synthetic creator data",
            "src/generate_data.py",
        )
        verify_outputs(["data/raw/creators.csv"])
        next_stage = 2
        total_stages = 9
    else:
        print()
        print("Synthetic mode: OFF")
        print("Using the existing data/raw/creators.csv.")
        next_stage = 1
        total_stages = 8

    stages = [
        (
            "Validate raw creator data",
            "src/validate_data.py",
            None,
            ["data/raw/creators.csv"],
        ),
        (
            "Calculate creator metrics",
            "src/metrics.py",
            None,
            ["data/processed/creator_metrics.csv"],
        ),
        (
            "Calculate Creator Growth Scores",
            "src/scoring.py",
            None,
            ["data/processed/creator_scores.csv"],
        ),
        (
            "Build creator segmentation",
            "src/segmentation.py",
            None,
            ["data/processed/creator_segments.csv"],
        ),
        (
            "Match creators to the campaign",
            "src/matching.py",
            None,
            ["data/processed/campaign_matches.csv"],
        ),
        (
            "Identify reactivation candidates",
            "src/reactivation.py",
            None,
            ["data/processed/reactivation_candidates.csv"],
        ),
        (
            "Generate AI-assisted creator insights",
            "src/ai_insights.py",
            (
                (["--use-ai"] if args.use_ai else [])
                + (
                    ["--limit", str(args.ai_limit)]
                    if args.ai_limit is not None
                    else []
                )
            ),
            [
                "data/processed/ai_creator_insights.csv",
                "data/processed/ai_prompts.jsonl",
            ],
        ),
        (
            "Build the weekly operations report",
            "src/reporting.py",
            None,
            [
                "data/processed/weekly_report.json",
                "data/processed/weekly_report.md",
            ],
        ),
    ]

    for offset, (label, script, stage_args, outputs) in enumerate(stages):
        stage_number = next_stage + offset
        run_stage(
            stage_number,
            total_stages,
            label,
            script,
            stage_args,
        )
        verify_outputs(outputs)

    print()
    print("=" * 72)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print("Processed creator records: data/raw/creators.csv")
    print("Dashboard data: data/processed/")
    print()
    print("To open the dashboard:")
    print("./.venv/bin/python -m streamlit run app.py")


if __name__ == "__main__":
    main()
