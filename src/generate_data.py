"""Generate reproducible, fully synthetic creator data for this portfolio project.

The generated records are fictional. They do not represent real creators,
TikTok accounts, or private data.
"""

from pathlib import Path
import random

import pandas as pd


# Each tuple contains a market, its primary language, and a relative weight.
# A larger weight means that market is slightly more likely to be selected.
MARKETS = [
    ("UK", "English", 24),
    ("Germany", "German", 18),
    ("France", "French", 17),
    ("Italy", "Italian", 12),
    ("Spain", "Spanish", 12),
    ("Poland", "Polish", 8),
    ("Netherlands", "Dutch", 6),
    ("Belgium", "French", 3),
]

CATEGORIES = ["Beauty", "Fashion", "Lifestyle", "Fitness", "Food", "Technology"]

# These profiles are used only while generating data. They are not saved in
# the CSV: a later module will create transparent, data-driven segments.
PROFILE_WEIGHTS = {
    "high_potential": 18,
    "large_low_engagement": 15,
    "micro_strong_engagement": 20,
    "consistent": 27,
    "declining": 12,
    "dormant": 8,
}

REQUIRED_COLUMNS = [
    "creator_id", "market", "language", "category", "followers", "avg_views",
    "engagement_rate", "follower_growth_30d", "videos_30d",
    "campaign_participation_90d", "response_rate", "conversion_rate",
    "last_active_days",
]


def _rounded_rate(value: float) -> float:
    """Round a rate to four decimal places while keeping decimal notation."""
    return round(value, 4)


def _profile_values(profile: str, rng: random.Random) -> dict:
    """Return correlated fictional performance values for one behaviour profile."""
    # Related ranges make a realistic mix instead of independent random numbers.
    if profile == "high_potential":
        followers, engagement_rate = rng.randint(5_000, 120_000), rng.uniform(0.07, 0.15)
        follower_growth_30d, videos_30d = rng.uniform(0.04, 0.25), rng.randint(10, 30)
        campaign_participation_90d, response_rate = rng.randint(3, 12), rng.uniform(0.65, 0.98)
        conversion_rate, last_active_days, view_multiplier = rng.uniform(0.015, 0.08), rng.randint(0, 7), rng.uniform(0.7, 4.0)
    elif profile == "large_low_engagement":
        followers, engagement_rate = rng.randint(100_000, 500_000), rng.uniform(0.01, 0.05)
        follower_growth_30d, videos_30d = rng.uniform(-0.02, 0.08), rng.randint(4, 18)
        campaign_participation_90d, response_rate = rng.randint(2, 10), rng.uniform(0.30, 0.80)
        conversion_rate, last_active_days, view_multiplier = rng.uniform(0.002, 0.025), rng.randint(0, 20), rng.uniform(0.5, 4.5)
    elif profile == "micro_strong_engagement":
        followers, engagement_rate = rng.randint(500, 25_000), rng.uniform(0.08, 0.15)
        follower_growth_30d, videos_30d = rng.uniform(0.03, 0.25), rng.randint(8, 30)
        campaign_participation_90d, response_rate = rng.randint(0, 5), rng.uniform(0.55, 0.98)
        conversion_rate, last_active_days, view_multiplier = rng.uniform(0.01, 0.08), rng.randint(0, 6), rng.uniform(0.6, 5.0)
    elif profile == "consistent":
        followers, engagement_rate = rng.randint(8_000, 200_000), rng.uniform(0.035, 0.09)
        follower_growth_30d, videos_30d = rng.uniform(-0.01, 0.09), rng.randint(8, 22)
        campaign_participation_90d, response_rate = rng.randint(1, 7), rng.uniform(0.45, 0.90)
        conversion_rate, last_active_days, view_multiplier = rng.uniform(0.006, 0.04), rng.randint(0, 12), rng.uniform(0.4, 2.5)
    elif profile == "declining":
        followers, engagement_rate = rng.randint(5_000, 300_000), rng.uniform(0.015, 0.07)
        follower_growth_30d, videos_30d = rng.uniform(-0.10, 0.015), rng.randint(1, 12)
        campaign_participation_90d, response_rate = rng.randint(0, 4), rng.uniform(0.20, 0.70)
        conversion_rate, last_active_days, view_multiplier = rng.uniform(0.002, 0.03), rng.randint(7, 30), rng.uniform(0.15, 1.5)
    else:  # dormant
        followers, engagement_rate = rng.randint(500, 250_000), rng.uniform(0.01, 0.10)
        follower_growth_30d, videos_30d = rng.uniform(-0.10, 0.03), rng.randint(0, 2)
        campaign_participation_90d, response_rate = rng.randint(0, 2), rng.uniform(0.20, 0.75)
        conversion_rate, last_active_days, view_multiplier = rng.uniform(0.002, 0.025), rng.randint(21, 45), rng.uniform(0.05, 1.0)

    # Views are related to audience size and activity, then kept in range.
    avg_views = max(1_000, min(int(followers * view_multiplier), 2_000_000))
    return {
        "followers": followers, "avg_views": avg_views,
        "engagement_rate": _rounded_rate(engagement_rate),
        "follower_growth_30d": _rounded_rate(follower_growth_30d),
        "videos_30d": videos_30d,
        "campaign_participation_90d": campaign_participation_90d,
        "response_rate": _rounded_rate(response_rate),
        "conversion_rate": _rounded_rate(conversion_rate),
        "last_active_days": last_active_days,
    }


def _validate_creators(creators: pd.DataFrame, expected_rows: int) -> None:
    """Raise a clear error when generated data does not meet project rules."""
    if len(creators) != expected_rows:
        raise ValueError(f"Expected {expected_rows} creators, but generated {len(creators)}.")
    if list(creators.columns) != REQUIRED_COLUMNS:
        raise ValueError("Generated data does not contain the required columns in the expected order.")
    if creators.isna().any().any():
        raise ValueError("Generated data contains missing values.")
    if not creators["creator_id"].is_unique:
        raise ValueError("Generated creator_id values must be unique.")

    ranges = {
        "followers": (500, 500_000), "avg_views": (1_000, 2_000_000),
        "engagement_rate": (0.01, 0.15), "follower_growth_30d": (-0.10, 0.25),
        "videos_30d": (0, 30), "campaign_participation_90d": (0, 12),
        "response_rate": (0.20, 0.98), "conversion_rate": (0.002, 0.08),
        "last_active_days": (0, 45),
    }
    for column, (minimum, maximum) in ranges.items():
        if not pd.api.types.is_numeric_dtype(creators[column]):
            raise ValueError(f"Column '{column}' must contain numeric values.")
        if not creators[column].between(minimum, maximum).all():
            raise ValueError(f"Column '{column}' has values outside {minimum} to {maximum}.")

    integer_columns = ["followers", "avg_views", "videos_30d", "campaign_participation_90d", "last_active_days"]
    for column in integer_columns:
        if not pd.api.types.is_integer_dtype(creators[column]):
            raise ValueError(f"Column '{column}' must contain whole numbers.")


def generate_creators(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a reproducible DataFrame of fictional creators.

    Args:
        n: Number of synthetic creators to generate. Must be at least 1.
        seed: Fixed seed that makes the same inputs produce the same data.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    rng = random.Random(seed)
    market_names = [market[0] for market in MARKETS]
    market_weights = [market[2] for market in MARKETS]
    languages_by_market = {market: language for market, language, _ in MARKETS}
    profile_names = list(PROFILE_WEIGHTS)
    profile_weights = list(PROFILE_WEIGHTS.values())

    rows = []
    for number in range(1, n + 1):
        market = rng.choices(market_names, weights=market_weights, k=1)[0]
        profile = rng.choices(profile_names, weights=profile_weights, k=1)[0]
        row = {
            "creator_id": f"creator_{number:03d}", "market": market,
            "language": languages_by_market[market], "category": rng.choice(CATEGORIES),
        }
        row.update(_profile_values(profile, rng))
        rows.append(row)

    creators = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    _validate_creators(creators, n)
    return creators


def main() -> None:
    """Generate 300 creators, save the CSV, and print a simple summary."""
    creators = generate_creators()
    output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "creators.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    creators.to_csv(output_path, index=False)

    print("Synthetic creator data generated successfully.")
    print(f"Rows: {len(creators)}")
    print(f"Columns: {len(creators.columns)}")
    print(f"Markets represented: {', '.join(sorted(creators['market'].unique()))}")
    print(f"Categories represented: {', '.join(sorted(creators['category'].unique()))}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
