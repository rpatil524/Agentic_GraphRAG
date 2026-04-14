"""
Unified Tier 3 runner.

Usage:
  python evaluation/run_tier3.py --dataset automated
  python evaluation/run_tier3.py --dataset human
  python evaluation/run_tier3.py --dataset both
"""

import argparse
import asyncio
import os

from _bootstrap import ensure_paths, output_dir

ensure_paths()

from baseline_rag.vector_rag import NaiveVectorRAG
from tier2_trajectory import evaluate_tier_2
from tier3_ragas import RagasEvaluator
from tier3_regas_human_dataset import (
    DEFAULT_AGENT_OUTPUT as HUMAN_AGENT_OUTPUT,
    DEFAULT_BASELINE_OUTPUT as HUMAN_BASELINE_OUTPUT,
    DEFAULT_DATASET_PATH as HUMAN_DATASET_PATH,
    DEFAULT_RESULTS_OUTPUT as HUMAN_RESULTS_OUTPUT,
    HumanDatasetRagasEvaluator,
    run_agent_on_human_dataset,
    run_baseline_on_human_dataset,
)


AUTOMATED_DATASET_PATH = os.getenv(
    "AUTOMATED_DATASET_PATH",
    "evaluation/datasets/automated_dataset.json",
)
AUTOMATED_AGENT_OUTPUT = str(output_dir() / "tier2_trajectory_results.json")
AUTOMATED_BASELINE_OUTPUT = str(output_dir() / "baseline_results.json")
AUTOMATED_RESULTS_OUTPUT = str(output_dir() / "tier3_ragas_results.json")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def run_automated(limit=None, concurrency=15):
    print("\n=== Automated Benchmark: Tier 2 (Graph Agent) ===")
    evaluate_tier_2(
        dataset_path=AUTOMATED_DATASET_PATH,
        output_path=AUTOMATED_AGENT_OUTPUT,
        limit=limit,
    )

    print("\n=== Automated Benchmark: Baseline ===")
    baseline = NaiveVectorRAG(api_key=OPENAI_API_KEY, concurrency=concurrency)
    await baseline.evaluate_golden_dataset_async(
        dataset_path=AUTOMATED_DATASET_PATH,
        output_path=AUTOMATED_BASELINE_OUTPUT,
        limit=limit,
    )

    print("\n=== Automated Benchmark: Tier 3 ===")
    evaluator = RagasEvaluator(api_key=OPENAI_API_KEY, concurrency=concurrency)
    await evaluator.run_evaluation_async(
        graph_results_path=AUTOMATED_AGENT_OUTPUT,
        baseline_results_path=AUTOMATED_BASELINE_OUTPUT,
        output_path=AUTOMATED_RESULTS_OUTPUT,
    )


async def run_human(limit=None, concurrency=15, dataset_path=HUMAN_DATASET_PATH):
    print("\n=== Human Benchmark: Graph Agent ===")
    run_agent_on_human_dataset(
        dataset_path=dataset_path,
        output_path=HUMAN_AGENT_OUTPUT,
        limit=limit,
    )

    print("\n=== Human Benchmark: Baseline ===")
    await run_baseline_on_human_dataset(
        dataset_path=dataset_path,
        output_path=HUMAN_BASELINE_OUTPUT,
        concurrency=concurrency,
        limit=limit,
    )

    print("\n=== Human Benchmark: Tier 3 ===")
    evaluator = HumanDatasetRagasEvaluator(
        api_key=OPENAI_API_KEY,
        concurrency=concurrency,
    )
    await evaluator.run_evaluation_async(
        graph_results_path=HUMAN_AGENT_OUTPUT,
        baseline_results_path=HUMAN_BASELINE_OUTPUT,
        output_path=HUMAN_RESULTS_OUTPUT,
    )


async def main():
    parser = argparse.ArgumentParser(description="Unified Tier 3 runner for SHAB benchmarks.")
    parser.add_argument(
        "--dataset",
        choices=["automated", "human", "both"],
        default="both",
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional question limit for quick test runs.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrency for async baseline/judge calls.",
    )
    parser.add_argument(
        "--human-dataset-path",
        default=HUMAN_DATASET_PATH,
        help="Path to the human benchmark JSON file.",
    )
    args = parser.parse_args()

    if args.dataset in {"automated", "both"}:
        await run_automated(limit=args.limit, concurrency=args.concurrency)

    if args.dataset in {"human", "both"}:
        await run_human(
            limit=args.limit,
            concurrency=args.concurrency,
            dataset_path=args.human_dataset_path,
        )


if __name__ == "__main__":
    asyncio.run(main())
