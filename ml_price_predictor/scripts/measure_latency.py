#!/usr/bin/env python3
"""Measure model prediction latency."""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from src.models.single_cat_model import SingleCategoryModel

def measure_prediction_latency(model_path, n_iterations=100):  # Reduced from 1000
    """Measure model prediction latency."""
    try:
        print(f"Loading model from {model_path}")
        model = SingleCategoryModel.load(model_path)

        # Generate sample input
        sample_input = {
            "price": 100.0,
            "steam_balance": "USD 100.00",
            "view_count": 500,
            "steam_level": 20,
            "steam_games": 50,
            "steam_friends": 100,
            "item_origin": "market",
            "steam_country": "US",
            "steam_community_ban": "no",
            "steam_is_limited": "no",
            "steam_full_games": {"total": 50, "list": {}},
            "steam_currency": "USD",
            "email_type": "gmail",
            "extended_guarantee": "yes",
            "nsb": "no",
            "item_domain": "steam",
            "resale_item_origin": "market",
            "steam_cs2_wingman_rank_id": "1",
            "steam_cs2_rank_id": "1",
            "steam_cs2_ban_type": "none",
            "published_date": int(time.time()),
            "update_stat_date": int(time.time()),
            "refreshed_date": int(time.time()),
            "steam_register_date": int(time.time()),
            "steam_last_activity": int(time.time())
        }

        print(f"Running {n_iterations} iterations...")
        latencies = []
        for i in range(n_iterations):
            if i % 10 == 0:  # Progress update every 10 iterations
                print(f"Progress: {i}/{n_iterations}")
            try:
                start_time = time.perf_counter()
                model.predict_single(sample_input)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
            except Exception as e:
                print(f"Error in iteration {i}: {str(e)}")
                continue

        if not latencies:
            return {
                "error": "No successful predictions",
                "mean_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0
            }

        results = {
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies))
        }

        return results
    except Exception as e:
        print(f"Error in latency measurement: {str(e)}")
        return {
            "error": str(e),
            "mean_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
            "min_latency_ms": 0,
            "max_latency_ms": 0
        }

def main():
    parser = argparse.ArgumentParser(description="Measure model prediction latency")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--n-iterations", type=int, default=100, help="Number of iterations")  # Reduced from 1000
    parser.add_argument("--output-file", help="Path to save results")
    args = parser.parse_args()

    results = measure_prediction_latency(args.model_path, args.n_iterations)

    print("\nLatency Measurements:")
    print("====================")
    print(f"Mean latency: {results['mean_latency_ms']:.2f}ms")
    print(f"P95 latency: {results['p95_latency_ms']:.2f}ms")
    print(f"P99 latency: {results['p99_latency_ms']:.2f}ms")
    print(f"Min latency: {results['min_latency_ms']:.2f}ms")
    print(f"Max latency: {results['max_latency_ms']:.2f}ms")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()
