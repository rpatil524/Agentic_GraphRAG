#!/usr/bin/env python3
"""Run the no-reflection-loop ablation on the human benchmark.

The full Agentic GraphRAG row is read from the saved Tier 3 human-benchmark
outputs. The ablated system is run live with the same investigator code but
with the bounded reflection loop restricted to one planning/tool iteration.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
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

from evaluation.tier3_regas_human_dataset import load_human_dataset
from agenticGraphRAG.investigator import SHABInvestigator


RESULTS_DIR = ROOT / "evaluation" / "results"
DATASET_PATH = ROOT / "evaluation" / "datasets" / "golden_dataset.json"
FULL_AGENT_OUTPUT = ROOT / "evaluation" / "results" / "tier2_trajectory_human_results.json"
FULL_AGENT_SCORES = ROOT / "evaluation" / "results" / "tier3_regas_human_dataset_results.json"

ANSWER_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini-2024-07-18")
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-2025-08-07")
MODEL_PRICES_PER_MILLION = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-2025-08-07": {"input": 1.25, "output": 10.00},
}


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def usage_dict(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def count_tool_calls(trace_log):
    return sum(1 for line in trace_log or [] if "Called:" in str(line))


def count_failed_tool_calls(trace_log):
    return sum(
        1
        for line in trace_log or []
        if "🛑 Error" in str(line) or "⚠️ Empty Result" in str(line)
    )


def is_empty_answer(answer):
    text = (answer or "").strip().lower()
    return (
        not text
        or text in {"search completed.", "no answer generated."}
        or text.startswith("error")
        or "i do not know" in text
        or "i don't know" in text
    )


def sum_usage(rows):
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "token_usage_available": True,
    }
    usage_keys = [
        "Faithfulness_judge_usage",
        "Correctness_judge_usage",
        "Answer_Relevance_judge_usage",
        "Information_Recall_judge_usage",
    ]
    for row in rows:
        for key in usage_keys:
            usage = row.get(key) or {}
            if usage.get("total_tokens") is None:
                totals["token_usage_available"] = False
                continue
            totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
            totals["completion_tokens"] += usage.get("completion_tokens") or 0
            totals["total_tokens"] += usage.get("total_tokens") or 0
    return totals


def cost_for_usage(model, usage):
    prices = MODEL_PRICES_PER_MILLION.get(model)
    if not prices or usage.get("total_tokens") is None:
        return None
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return (
        prompt_tokens * prices["input"] / 1_000_000
        + completion_tokens * prices["output"] / 1_000_000
    )


def estimate_judge_cost(rows):
    total = 0.0
    complete = True
    for row in rows:
        for metric in ["Faithfulness", "Correctness", "Answer_Relevance", "Information_Recall"]:
            cost = cost_for_usage(JUDGE_MODEL, row.get(f"{metric}_judge_usage") or {})
            if cost is None:
                complete = False
            else:
                total += cost
    return total if complete else None


async def judge_float(client, sem, prompt):
    async with sem:
        last_error = None
        for attempt in range(4):
            try:
                response = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                score = float(response.choices[0].message.content.strip())
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"Judge returned an out-of-range score: {score}")
                return score, usage_dict(response)
            except Exception as exc:
                last_error = exc
                if "429" in str(exc):
                    await asyncio.sleep(5 * (attempt + 1))
                elif attempt < 3:
                    await asyncio.sleep(attempt + 1)
    raise RuntimeError("LLM judging failed after four attempts.") from last_error


async def judge_all(client, sem, question, expected, answer, context):
    prompts = {
        "Faithfulness": f"""Given the context and the answer, compute a faithfulness score between 0.0 and 1.0.
1.0 means the answer is 100% derived from the context. 0.0 means the answer states completely fabricated facts.
Output ONLY a float number.

Context: {context}
Answer: {answer}
""",
        "Correctness": f"""Given the question, the expected answer, and the actual answer, compute a correctness score between 0.0 and 1.0.
1.0 means the actual answer is factually correct and captures the expected answer well.
0.0 means the actual answer is factually wrong or fails to answer the question correctly.
Be tolerant to differences in wording, formatting, ordering, or level of detail, as long as the substance is correct.
Output ONLY a float number.

Question: {question}
Expected Answer: {expected}
Actual Answer: {answer}
""",
        "Answer_Relevance": f"""Given the question and the answer, compute an answer relevance score between 0.0 and 1.0.
1.0 means the answer directly and clearly answers the question. 0.0 means it does not answer it.
Output ONLY a float number.

Question: {question}
Answer: {answer}
""",
        "Information_Recall": f"""Compare the Expected Answer to the Actual Answer.
Score from 0.0 to 1.0 how much of the Expected Answer's core information is present in the Actual Answer.
Output ONLY a float number.

Expected: {expected}
Actual: {answer}
""",
    }
    tasks = {
        metric: asyncio.create_task(judge_float(client, sem, prompt))
        for metric, prompt in prompts.items()
    }
    scores = {}
    usages = {}
    for metric, task in tasks.items():
        score, usage = await task
        scores[metric] = score
        usages[f"{metric}_judge_usage"] = usage
    return scores, usages


async def judge_faithfulness_only(client, sem, question_id, answer, context):
    prompt = f"""Given the context and the answer, compute a faithfulness score between 0.0 and 1.0.
1.0 means the answer is 100% derived from the context. 0.0 means the answer states completely fabricated facts.
Output ONLY a float number.

Context: {context}
Answer: {answer}
"""
    score, usage = await judge_float(client, sem, prompt)
    return {
        "question_id": question_id,
        "Faithfulness": score,
        "Faithfulness_judge_usage": usage,
    }


def run_no_reflection_answers(questions, output_path, max_iterations):
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "shabdb")
    api_key = os.getenv("OPENAI_API_KEY")

    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    investigator = SHABInvestigator(
        driver=driver,
        api_key=api_key,
        model=ANSWER_MODEL,
        iterations=max_iterations,
        database=database,
    )
    rows = []

    try:
        with output_path.open("w", encoding="utf-8") as out:
            for q in tqdm.tqdm(questions, desc="No-reflection GraphRAG"):
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
                    "system": "No-reflection-loop Agentic GraphRAG",
                    "ablation": "no_reflection",
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
                    # One extra LLM call is used by the intent router.
                    "agent_llm_calls_estimated": assistant_llm_calls + 1,
                    "answer_model": ANSWER_MODEL,
                    "judge_model": JUDGE_MODEL,
                }
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    finally:
        driver.close()

    return rows


async def score_no_reflection(rows, concurrency):
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
    with tqdm.tqdm(total=len(tasks), desc="Judge no-reflection") as pbar:
        for task in asyncio.as_completed(tasks):
            scored.append(await task)
            pbar.update(1)
    return scored


async def score_full_faithfulness(concurrency):
    data = json.loads(FULL_AGENT_OUTPUT.read_text(encoding="utf-8"))
    rows = data.get("trajectories", [])
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        judge_faithfulness_only(
            client,
            sem,
            row.get("question_id"),
            row.get("agent_answer", ""),
            row.get("retrieved_context", "[]"),
        )
        for row in rows
    ]
    scored = []
    with tqdm.tqdm(total=len(tasks), desc="Judge full faithfulness") as pbar:
        for task in asyncio.as_completed(tasks):
            scored.append(await task)
            pbar.update(1)
    return scored


def summarize_no_reflection(rows):
    latencies = [row["latency_seconds"] for row in rows]
    total_tool_calls = sum(row["tool_calls"] for row in rows)
    usage = sum_usage(rows)
    cost = estimate_judge_cost(rows)
    return {
        "system": "No-reflection-loop Agentic GraphRAG",
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


def summarize_full_system(full_faithfulness_rows):
    trajectory_data = json.loads(FULL_AGENT_OUTPUT.read_text(encoding="utf-8"))
    score_data = json.loads(FULL_AGENT_SCORES.read_text(encoding="utf-8"))
    trajectories = trajectory_data.get("trajectories", [])
    latencies = [row["latency_seconds"] for row in trajectories]
    tool_calls = [count_tool_calls(row.get("trace_log", [])) for row in trajectories]
    total_tool_calls = sum(tool_calls)
    failed_tool_calls = sum(count_failed_tool_calls(row.get("trace_log", [])) for row in trajectories)
    return {
        "system": "Full Agentic GraphRAG",
        "source": "saved Tier 3 human outputs; faithfulness judged from saved contexts",
        "n": len(trajectories),
        "Faithfulness": statistics.mean(row["Faithfulness"] for row in full_faithfulness_rows),
        "Correctness": score_data["Correctness"],
        "Answer_Relevance": score_data["Answer_Relevance"],
        "Information_Recall": score_data["Information_Recall"],
        "average_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "average_tool_calls": statistics.mean(tool_calls),
        "average_llm_calls": None,
        "failed_tool_call_rate": failed_tool_calls / total_tool_calls if total_tool_calls else 0.0,
        "empty_answer_rate": sum(1 for row in trajectories if is_empty_answer(row.get("agent_answer"))) / len(trajectories),
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "estimated_openai_cost_usd": None,
        "cost_note": (
            "Original full-system answer generation and Tier 3 judge token usage were not logged; "
            "complete cost is not available from the saved outputs."
        ),
    }


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


def fmt(value, digits=3):
    if value is None or value == "":
        return "--"
    return f"{float(value):.{digits}f}"


def write_latex(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table*}[htbp]\n")
        fh.write("\\centering\n")
        fh.write("\\caption{No-reflection-loop ablation on the 60-question manually curated benchmark.}\n")
        fh.write("\\label{tab:phase2_ablation_no_reflection}\n")
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
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_human_dataset(str(DATASET_PATH))

    raw_answers_path = RESULTS_DIR / "outputs_ablation_no_reflection_human.jsonl"
    metrics_path = RESULTS_DIR / "results_phase2_ablation_no_reflection.csv"
    table_path = RESULTS_DIR / "table_ablation_no_reflection.tex"

    no_reflection_rows = run_no_reflection_answers(
        questions,
        raw_answers_path,
        max_iterations=args.max_iterations,
    )
    no_reflection_rows = await score_no_reflection(no_reflection_rows, args.concurrency)
    write_outputs_jsonl(raw_answers_path, no_reflection_rows)

    full_faithfulness_rows = await score_full_faithfulness(args.concurrency)
    summaries = [
        summarize_full_system(full_faithfulness_rows),
        summarize_no_reflection(no_reflection_rows),
    ]
    write_metrics_csv(metrics_path, summaries)
    write_latex(table_path, summaries)

    print(f"Wrote {raw_answers_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    asyncio.run(main())
