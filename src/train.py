#!/usr/bin/env python3
"""Train a model for a specific category."""

import argparse
import json
import pandas as pd
from pathlib import Path
from src.models.single_cat_model import SingleCategoryModel

def main():
    parser = argparse.ArgumentParser(description="Train category-specific model")
    parser.add_argument("--data-path", required=True, help="Path to training data")
    parser.add_argument("--category-id", type=int, required=True, help="Category ID")
    parser.add_argument("--output-dir", default="models/trained", help="Output directory")
    args = parser.parse_args()

    # Load and process data
    with open(args.data_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Initialize and train model
    model = SingleCategoryModel(category_number=args.category_id)
    model.train(df)

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"category_{args.category_id}_model.cbm"
    model.export(str(model_path))

if __name__ == "__main__":
    main()
