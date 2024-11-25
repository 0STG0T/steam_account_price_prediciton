#!/usr/bin/env python3
"""Generate sample data for testing and development."""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_data(n_samples=100, category_id=1):
    """Generate realistic-looking sample data."""
    np.random.seed(42)

    data = []
    base_date = datetime.now()

    for _ in range(n_samples):
        price = np.random.lognormal(5, 1)  # Realistic price distribution
        sold_price = price * np.random.normal(0.95, 0.1)  # Slight variation from price

        sample = {
            "category_id": category_id,
            "price": price,
            "sold_price": sold_price,
            "view_count": int(np.random.lognormal(5, 1)),
            "steam_level": int(np.random.normal(20, 10)),
            "steam_games": int(np.random.lognormal(4, 1)),
            "steam_friends": int(np.random.lognormal(4, 0.8)),
            "steam_balance": f"USD {np.random.uniform(10, 500):.2f}",
            "steam_full_games": {"total": int(np.random.lognormal(3, 1)), "games": [{"playtime": int(np.random.lognormal(4, 1))}]},
            "item_origin": np.random.choice(["market", "trade", "drop"]),
            "extended_guarantee": np.random.choice(["yes", "no"]),
            "nsb": np.random.choice(["yes", "no"]),
            "email_type": np.random.choice(["gmail", "outlook", "yahoo"]),
            "item_domain": "steam",
            "resale_item_origin": np.random.choice(["market", "trade"]),
            "steam_country": np.random.choice(["US", "EU", "RU"]),
            "steam_community_ban": np.random.choice(["yes", "no"], p=[0.05, 0.95]),
            "steam_is_limited": np.random.choice(["yes", "no"]),
            "steam_cs2_wingman_rank_id": np.random.randint(1, 19),
            "steam_cs2_rank_id": np.random.randint(1, 19),
            "steam_cs2_ban_type": np.random.choice(["none", "vac", "game"], p=[0.9, 0.05, 0.05]),
            "created_at": (base_date - timedelta(days=np.random.randint(1, 365))).timestamp()
        }
        data.append(sample)

    return pd.DataFrame(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample data")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--category-id", type=int, default=1, help="Category ID")
    parser.add_argument("--validation-only", action="store_true", help="Generate only validation data")
    args = parser.parse_args()

    if not args.validation_only:
        # Generate training data
        train_df = generate_sample_data(n_samples=args.n_samples, category_id=args.category_id)
        train_df.to_json(f"{args.output_dir}/sample_train.json", orient="records")
        print(f"Generated {args.n_samples} training samples")

    # Generate validation data
    val_df = generate_sample_data(n_samples=args.n_samples // 5, category_id=args.category_id)
    val_df.to_json(f"{args.output_dir}/sample_validation.json", orient="records")

    # Save actuals with only necessary columns for monitoring
    actuals_df = val_df[['sold_price']].copy()
    actuals_df.to_csv(f"{args.output_dir}/actuals.csv", index=False)
    print(f"Generated {args.n_samples // 5} validation samples")
    print(f"Files saved in {args.output_dir}")

    print(f"Generated {args.n_samples // 5} validation samples")
    print(f"Files saved in {args.output_dir}")
