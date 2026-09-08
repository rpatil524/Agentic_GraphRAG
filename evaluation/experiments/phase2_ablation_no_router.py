#!/usr/bin/env python3
"""Run the no-intent-router ablation on the human benchmark.

The ablated agent disables intent routing by exposing the full tool schema for
every question. The production investigator is not modified.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import tqdm
from neo4j import GraphDatabase
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.tier3_regas_human_dataset import load_human_dataset
from phase2_ablation_no_reflection import (
    ANSWER_MODEL,
    DATASET_PATH,
    RESULTS_DIR,
    JUDGE_MODEL,
    count_failed_tool_calls,
    count_tool_calls,
    estimate_judge_cost,
    fmt,
    is_empty_answer,
    judge_all,
    percentile,
    sum_usage,
)
from agenticGraphRAG.investigator import SHABInvestigator


FULL_COMPARISON_PATH = RESULTS_DIR / "results_phase2_ablation_no_reflection.csv"


class NoRouterInvestigator(SHABInvestigator):
    """Investigator variant that bypasses intent classification.

    Returning "all" forces the existing `ask` method to load the full tool
    schema for every query. This is intentionally isolated to the ablation
    runner so the production agent is not changed.
    """

    def _route_intent(self, user_question):
        return "all"


def run_no_router_answers(questions, output_path, max_iterations):
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "shabdb")
    api_key = os.getenv("OPENAI_API_KEY")

    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    investigator = NoRouterInvestigator(
        driver=driver,
        api_key=api_key,
        model=ANSWER_MODEL,
        iterations=max_iterations,
        database=database,
    )
    rows = []

    try:
        with output_path.open("w", encoding="utf-8") as out:
            for q in tqdm.tqdm(questions, desc="No-router GraphRAG"):
                trace = []

                def trace_callback(msg):
                    trace.append(msg)

                start = time.time()
                try:
                    response = investigator.ask(
                        q["question_text"],
                        trace_callback=trace_callback,
                    )
                    answer = response.get("answer", "No answer generated.")
                    retrieved_context = json.dumps(response.get("data", []), ensure_ascii=False)
                    new_messages = response.get("new_messages", [])
                except Exception as exc:
                    answer = f"Error: {exc}"
                    retrieved_context = "[]"
                    new_messages = []
                latency = time.time() - start

                assistant_llm_calls = sum(
                    1 for msg in new_messages if msg.get("role") == "assistant"
                )
                row = {
                    "system": "No-intent-router Agentic GraphRAG",
                    "ablation": "no_router",
                    "max_iterations": max_iterations,
                    "question_id": q["question_id"],
                    "question": q["question_text"],
                    "tier": q["tier"],
                    "category": q["category"],
                    "expected_answer": q["expected_answer"],
                    "agent_answer": answer,
                    "retrieved_context": retrieved_context,
                    "trace_log": trace,
                    "latency_seconds": latency,
                    "tool_calls": count_tool_calls(trace),
                    "failed_tool_calls": count_failed_tool_calls(trace),
                    "empty_answer": is_empty_answer(answer),
                    # No router LLM call is made in this ablation.
                    "agent_llm_calls_estimated": assistant_llm_calls,
                    "answer_model": ANSWER_MODEL,
                    "judge_model": JUDGE_MODEL,
                }
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    finally:
        driver.close()

    return rows


async def score_rows(rows, concurrency):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(concurrency)

    async def score_row(row):
        scores, usages = await judge_all(
            client,
            sem,
            row["question"],
            row["expected_answer"],
            row["agent_answer"],
            row["retrieved_context"],
        )
        row.update(scores)
        row.update(usages)
        return row

    scored = []
    tasks = [score_row(row) for row in rows]
    with tqdm.tqdm(total=len(tasks), desc="Judge no-router") as pbar:
        for task in asyncio.as_completed(tasks):
            scored.append(await task)
            pbar.update(1)
    return scored


def summarize_no_router(rows):
    latencies = [row["latency_seconds"] for row in rows]
    total_tool_calls = sum(row["tool_calls"] for row in rows)
    usage = sum_usage(rows)
    cost = estimate_judge_cost(rows)
    return {
        "system": "No-intent-router Agentic GraphRAG",
        "source": "live ablation run",
        "n": len(rows),
        "Faithfulness": statistics.mean(row["Faithfulness"] for row in rows),
        "Correctness": statistics.mean(row["Correctness"] for row in rows),
        "Answer_Relevance": statistics.mean(row["Answer_Relevance"] for row in rows),
        "Information_Recall": statistics.mean(row["Information_Recall"] for row in rows),
        "average_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "average_tool_calls": statistics.mean(row["tool_calls"] for row in rows),
        "average_llm_calls": statistics.mean(row["agent_llm_calls_estimated"] for row in rows),
        "failed_tool_call_rate": (
            sum(row["failed_tool_calls"] for row in rows) / total_tool_calls
            if total_tool_calls
            else 0.0
        ),
        "empty_answer_rate": sum(1 for row in rows if row["empty_answer"]) / len(rows),
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "prompt_tokens": usage["prompt_tokens"] if usage["token_usage_available"] else None,
        "completion_tokens": usage["completion_tokens"] if usage["token_usage_available"] else None,
        "total_tokens": usage["total_tokens"] if usage["token_usage_available"] else None,
        "estimated_openai_cost_usd": cost,
        "cost_note": (
            "Estimated from logged judge token usage only. Agent answer-generation "
            "token usage is not logged by the requests-based investigator wrapper."
        ),
    }


def load_full_summary():
    if not FULL_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FULL_COMPARISON_PATH}. Run the no-reflection ablation first "
            "or provide another full-system summary with faithfulness."
        )
    with FULL_COMPARISON_PATH.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["system"] == "Full Agentic GraphRAG":
                return row
    raise ValueError(f"No Full Agentic GraphRAG row found in {FULL_COMPARISON_PATH}")


def write_outputs_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as out:
        for row in sorted(rows, key=lambda x: x["question_id"]):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_csv(path, rows):
    fieldnames = [
        "system",
        "source",
        "n",
        "Faithfulness",
        "Correctness",
        "Answer_Relevance",
        "Information_Recall",
        "average_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
        "average_tool_calls",
        "average_llm_calls",
        "failed_tool_call_rate",
        "empty_answer_rate",
        "answer_model",
        "judge_model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_openai_cost_usd",
        "cost_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table*}[htbp]\n")
        fh.write("\\centering\n")
        fh.write("\\caption{No-intent-router ablation on the 60-question manually curated benchmark.}\n")
        fh.write("\\label{tab:phase2_ablation_no_router}\n")
        fh.write("\\small\n")
        fh.write("\\begin{tabular}{lcccccccc}\n")
        fh.write("\\toprule\n")
        fh.write("System & Faith. & Corr. & Rel. & Recall & Mean s & P95 s & Avg. Tools & Empty \\\\\n")
        fh.write("\\midrule\n")
        for row in rows:
            fh.write(
                f"{row['system']} & {fmt(row['Faithfulness'])} & {fmt(row['Correctness'])} & "
                f"{fmt(row['Answer_Relevance'])} & {fmt(row['Information_Recall'])} & "
                f"{fmt(row['average_latency_seconds'], 2)} & {fmt(row['p95_latency_seconds'], 2)} & "
                f"{fmt(row['average_tool_calls'], 2)} & {fmt(row['empty_answer_rate'])} \\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")
        fh.write("\\end{table*}\n")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable-router", action="store_true", default=True)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_human_dataset(str(DATASET_PATH))

    answers_path = RESULTS_DIR / "outputs_ablation_no_router_human.jsonl"
    metrics_path = RESULTS_DIR / "results_phase2_ablation_no_router.csv"
    table_path = RESULTS_DIR / "table_ablation_no_router.tex"

    no_router_rows = run_no_router_answers(
        questions,
        answers_path,
        max_iterations=args.max_iterations,
    )
    no_router_rows = await score_rows(no_router_rows, args.concurrency)
    write_outputs_jsonl(answers_path, no_router_rows)

    summaries = [
        load_full_summary(),
        summarize_no_router(no_router_rows),
    ]
    write_metrics_csv(metrics_path, summaries)
    write_latex(table_path, summaries)

    print(f"Wrote {answers_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    asyncio.run(main())
