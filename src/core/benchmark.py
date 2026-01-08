import argparse
import asyncio
import time
import os
from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd

from src.core.dataset import load_hotpot_qa_eval
from src.core.full_context_baseline import AsyncFullContextBaseline
from src.core.rag_baseline import RagBaseline
from src.core.coa_baseline import AsyncCoABaseline
from src.core.coa_map_reduce import AsyncCoAMapReduce
from src.core.evaluation import evaluate_batch

# Ensure results directory exists
os.makedirs("results", exist_ok=True)


async def run_model(model_name, model_instance, samples):
    print(f"\n--- Benchmarking {model_name} ---")
    start_time = time.perf_counter()

    # Run Prediction
    predictions = await model_instance.run_batch(samples)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_latency = total_time / len(samples)

    # Run Evaluation
    metrics = evaluate_batch(samples, predictions)

    return {
        "Model": model_name,
        "Total Time (s)": round(total_time, 2),
        "Avg Latency (s)": round(avg_latency, 2),
        "Exact Match": metrics.em,
        "F1 Score": metrics.f1,
    }


def plot_results(results_df):
    """Generates a comparison chart."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for F1 Score
    colors = ["#bdc3c7", "#f39c12", "#2ecc71"]  # Grey, Orange, Green
    bars = ax1.bar(
        results_df["Model"],
        results_df["F1 Score"],
        color=colors,
        alpha=0.7,
        label="F1 Score",
    )
    ax1.set_ylabel("F1 Score", color="#2c3e50", fontweight="bold")
    ax1.set_ylim(0, 1.0)

    # Line chart for Latency
    ax2 = ax1.twinx()
    ax2.plot(
        results_df["Model"],
        results_df["Avg Latency (s)"],
        color="#c0392b",
        marker="o",
        linewidth=2,
        label="Latency (s)",
    )
    ax2.set_ylabel("Avg Latency (seconds)", color="#c0392b", fontweight="bold")

    plt.title("Architecture Comparison: F1 Score vs Latency", fontsize=14)
    plt.savefig("results/benchmark_chart.png")
    print("\n✅ Chart saved to results/benchmark_chart.png")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    # 1. Load Data (Same samples for all models)
    print(f"Loading {args.num_samples} samples...")
    dataset = load_hotpot_qa_eval()
    samples = list(islice(dataset, args.num_samples))

    results = []

    # 2. Define Models
    models = [
        ("Full Context (Baseline)", AsyncFullContextBaseline(max_concurrency=50)),
        ("RAG (Baseline)", RagBaseline(max_concurrency=50)),
        ("CoA (Baseline)", AsyncCoABaseline(max_concurrency=10)),
        ("CoA Map-Reduce (Ours)", AsyncCoAMapReduce(max_concurrency=10)),
    ]

    # 3. Run Benchmarks
    for name, model in models:
        stats = await run_model(name, model, samples)
        results.append(stats)

    # 4. Save & Visualize
    df = pd.DataFrame(results)
    print("\n=== FINAL RESULTS ===")
    print(df.to_markdown(index=False))

    df.to_csv("results/benchmark_metrics.csv", index=False)
    plot_results(df)


if __name__ == "__main__":
    asyncio.run(main())
