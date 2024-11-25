#!/usr/bin/env python3
"""Export CatBoost model to ONNX format."""

import argparse
from pathlib import Path
from src.models.single_cat_model import SingleCategoryModel

def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--output-path", required=True, help="Path for ONNX model")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model = SingleCategoryModel.load(args.model_path)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    model.export(str(output_path))
    print("Export complete")

if __name__ == "__main__":
    main()
