"""Reusable validation checks for synthetic creator CSV data.

This module only checks data. It never changes the DataFrame or the CSV file.
"""

from pathlib import Path
import sys

import pandas as pd


REQUIRED_COLUMNS = [
    "creator_id", "market", "language", "category", "followers", "avg_views",
    "engagement_rate", "follower_growth_30d", "videos_30d",
    "campaign_participation_90d", "response_rate", "conversion_rate",
    "last_active_days",
]

INTEGER_COLUMNS = [
    "followers", "avg_views", "videos_30d", "campaign_participation_90d",
    "last_active_days",
]
RATE_COLUMNS = [
    "engagement_rate", "follower_growth_30d", "response_rate", "conversion_rate",
]

ALLOWED_MARKETS = {"UK", "Germany", "France", "Italy", "Spain", "Poland", "Netherlands", "Belgium"}
ALLOWED_LANGUAGES = {"English", "German", "French", "Italian", "Spanish", "Polish", "Dutch"}
ALLOWED_CATEGORIES = {"Beauty", "Fashion", "Lifestyle", "Fitness", "Food", "Technology"}

VALUE_RANGES = {
    "followers": (500, 500_000),
    "avg_views": (1_000, 2_000_000),
    "engagement_rate": (0.01, 0.15),
    "follower_growth_30d": (-0.10, 0.25),
    "videos_30d": (0, 30),
    "campaign_participation_90d": (0, 12),
    "response_rate": (0.20, 0.98),
    "conversion_rate": (0.002, 0.08),
    "last_active_days": (0, 45),
}

NON_NEGATIVE_COLUMNS = [
    "followers", "avg_views", "videos_30d", "campaign_participation_90d",
    "last_active_days",
]


def _empty_report(df: pd.DataFrame) -> dict:
    """Create a report with counts that future pipeline code can reuse."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_value_count": 0,
        "duplicate_creator_count": 0,
        "invalid_market_count": 0,
        "invalid_language_count": 0,
        "invalid_category_count": 0,
        "range_violations": {},
    }


def validate_creator_data(df: pd.DataFrame) -> dict:
    """Validate a creator DataFrame and return a report when it is valid.

    A ValueError is raised when one or more checks fail. All detected issues are
    included in one message so a beginner can fix them together.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Creator data must be provided as a pandas DataFrame.")

    report = _empty_report(df)
    errors = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    unexpected_columns = [column for column in df.columns if column not in REQUIRED_COLUMNS]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}.")
    if unexpected_columns:
        errors.append(f"Unexpected columns: {', '.join(unexpected_columns)}.")

    # Stop here: later checks need every required column to exist.
    if missing_columns:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))

    missing_by_column = df[REQUIRED_COLUMNS].isna().sum()
    affected_columns = missing_by_column[missing_by_column > 0]
    report["missing_value_count"] = int(affected_columns.sum())
    if not affected_columns.empty:
        details = ", ".join(f"{column} ({count})" for column, count in affected_columns.items())
        errors.append(f"Missing values found in: {details}.")

    # Empty strings are different from missing values, but are also invalid IDs.
    empty_id_count = int(df["creator_id"].astype(str).str.strip().eq("").sum())
    duplicate_ids = df.loc[df["creator_id"].duplicated(keep=False), "creator_id"].astype(str).unique().tolist()
    report["duplicate_creator_count"] = len(duplicate_ids)
    if empty_id_count:
        errors.append(f"creator_id contains {empty_id_count} empty value(s).")
    if duplicate_ids:
        errors.append(f"Duplicate creator_id values: {', '.join(duplicate_ids)}.")

    # Do not convert text to numbers. A wrong type should be fixed at the source.
    numeric_columns = INTEGER_COLUMNS + RATE_COLUMNS
    valid_numeric_columns = []
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            errors.append(f"Column '{column}' must be numeric; conversion was not attempted.")
        else:
            valid_numeric_columns.append(column)

    for column in INTEGER_COLUMNS:
        if column in valid_numeric_columns:
            values_are_whole = (df[column].dropna() % 1 == 0).all()
            if not values_are_whole:
                errors.append(f"Column '{column}' must contain whole-number values.")

    invalid_market_count = int((~df["market"].isin(ALLOWED_MARKETS)).sum())
    invalid_language_count = int((~df["language"].isin(ALLOWED_LANGUAGES)).sum())
    invalid_category_count = int((~df["category"].isin(ALLOWED_CATEGORIES)).sum())
    report["invalid_market_count"] = invalid_market_count
    report["invalid_language_count"] = invalid_language_count
    report["invalid_category_count"] = invalid_category_count
    if invalid_market_count:
        errors.append(f"Invalid market values: {invalid_market_count} row(s).")
    if invalid_language_count:
        errors.append(f"Invalid language values: {invalid_language_count} row(s).")
    if invalid_category_count:
        errors.append(f"Invalid category values: {invalid_category_count} row(s).")

    for column, (minimum, maximum) in VALUE_RANGES.items():
        if column in valid_numeric_columns:
            invalid_count = int((~df[column].between(minimum, maximum)).sum())
            report["range_violations"][column] = invalid_count
            if invalid_count:
                errors.append(
                    f"Column '{column}' has {invalid_count} row(s) outside {minimum} to {maximum}."
                )

    # This explicit check makes the non-negative rule visible in the code.
    for column in NON_NEGATIVE_COLUMNS:
        if column in valid_numeric_columns:
            negative_count = int((df[column] < 0).sum())
            if negative_count:
                errors.append(f"Column '{column}' has {negative_count} negative value(s).")

    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))

    return report


def print_validation_report(report: dict) -> None:
    """Print a short, beginner-friendly version of a validation report."""
    print("Validation report")
    print(f"Rows: {report['row_count']}")
    print(f"Columns: {report['column_count']}")
    print(f"Missing values: {report['missing_value_count']}")
    print(f"Duplicate creator IDs: {report['duplicate_creator_count']}")
    print(f"Invalid markets: {report['invalid_market_count']}")
    print(f"Invalid languages: {report['invalid_language_count']}")
    print(f"Invalid categories: {report['invalid_category_count']}")
    print(f"Range violations: {report['range_violations']}")


def main() -> None:
    """Load the raw CSV, validate it, and exit with a useful status code."""
    input_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "creators.csv"
    try:
        creators = pd.read_csv(input_path)
        report = validate_creator_data(creators)
        print_validation_report(report)
        print("VALIDATION PASSED")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
