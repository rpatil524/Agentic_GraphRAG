#!/usr/bin/env python3
"""Run vector-RAG top-k answer-quality sensitivity on the human benchmark.

This script uses the existing all-MiniLM Chroma database. It does not rebuild
embeddings and does not query Neo4j. OpenAI is used only for answer generation
and LLM-as-a-judge scoring.
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
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluation.baseline_rag.vector_rag import NaiveVectorRAG
from evaluation.tier3_regas_human_dataset import load_human_dataset


RESULTS_DIR = ROOT / "evaluation" / "results"
DATASET_PATH = ROOT / "evaluation" / "datasets" / "golden_dataset.json"
CHROMA_PATH = ROOT / "evaluation" / "baseline_rag" / "chroma_db"

ANSWER_MODEL = os.getenv("OPENAI_BASELINE_MODEL", "gpt-4o-mini-2024-07-18")
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-2025-08-07")
MODEL_PRICES_PER_MILLION = {
    # Standard API text-token rates used for an approximate run-cost estimate.
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
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
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def build_answer_prompt(question, context_docs):
    parsed_docs = []
    for doc in context_docs:
        if isinstance(doc, dict):
            parsed_docs.append(str(doc.get("text", doc)))
        else:
            parsed_docs.append(str(doc))

    context = "\n".join(parsed_docs)
    if not context.strip():
        context = "NO CONTEXT RETRIEVED."

    prompt = f"""You are an assistant answering questions about the Swiss Commercial Register based ONLY on the provided context.

Your job is to extract the answer as directly as possible from the context.
If the question asks for a list of companies or people, list all matching names you can support from the context.
If the question asks for a date or location, return only the relevant date or location.
Do not add extra commentary. Do not say more than the context supports.
If the information is not in the context, say briefly that you do not know.

Context:
{context}

Question: {question}
"""
    return prompt, context


async def generate_answer(client, question, context_docs):
    prompt, context = build_answer_prompt(question, context_docs)
    response = await client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content, context, usage_dict(response)


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


async def run_for_k(k, questions, concurrency):
    baseline = NaiveVectorRAG(
        api_key=os.getenv("OPENAI_API_KEY"),
        db_path=str(CHROMA_PATH),
        concurrency=concurrency,
    )
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(concurrency)
    judge_sem = asyncio.Semaphore(concurrency)
    out_path = RESULTS_DIR / f"outputs_vector_topk_answer_quality_k{k}.jsonl"

    async def process_question(q):
        async with sem:
            start = time.time()
            context_docs = await asyncio.to_thread(
                baseline.retrieve_expanded,
                q["question_text"],
                k,
                k,
            )
            answer, context_text, answer_usage = await generate_answer(
                client, q["question_text"], context_docs
            )
            scores, judge_usages = await judge_all(
                client,
                judge_sem,
                q["question_text"],
                q["expected_answer"],
                answer,
                context_text,
            )
            latency = time.time() - start
            context_chars = len(context_text)
            row = {
                "k": k,
                "question_id": q["question_id"],
                "question": q["question_text"],
                "tier": q["tier"],
                "category": q["category"],
                "expected_answer": q["expected_answer"],
                "baseline_answer": answer,
                "retrieved_context": context_docs,
                "retrieved_doc_count": len(context_docs),
                "context_char_count": context_chars,
                "latency_seconds": latency,
                "answer_model": ANSWER_MODEL,
                "judge_model": JUDGE_MODEL,
                "answer_usage": answer_usage,
                **judge_usages,
                **scores,
            }
            return row

    results = []
    with out_path.open("w", encoding="utf-8") as out:
        tasks = [process_question(q) for q in questions]
        with tqdm.tqdm(total=len(tasks), desc=f"Vector top-k answer quality k={k}") as pbar:
            for task in asyncio.as_completed(tasks):
                row = await task
                results.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                pbar.update(1)
    return results, out_path


def sum_usage(rows):
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "token_usage_available": True,
    }
    usage_keys = [
        "answer_usage",
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


def estimate_cost(rows):
    total = 0.0
    complete = True
    for row in rows:
        answer_cost = cost_for_usage(ANSWER_MODEL, row.get("answer_usage") or {})
        if answer_cost is None:
            complete = False
        else:
            total += answer_cost

        for metric in ["Faithfulness", "Correctness", "Answer_Relevance", "Information_Recall"]:
            judge_cost = cost_for_usage(JUDGE_MODEL, row.get(f"{metric}_judge_usage") or {})
            if judge_cost is None:
                complete = False
            else:
                total += judge_cost
    return total if complete else None


def summarize(k, rows):
    latencies = [row["latency_seconds"] for row in rows]
    usage = sum_usage(rows)
    estimated_cost = estimate_cost(rows)
    return {
        "k": k,
        "n": len(rows),
        "Faithfulness": statistics.mean(row["Faithfulness"] for row in rows),
        "Correctness": statistics.mean(row["Correctness"] for row in rows),
        "Answer_Relevance": statistics.mean(row["Answer_Relevance"] for row in rows),
        "Information_Recall": statistics.mean(row["Information_Recall"] for row in rows),
        "average_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "average_retrieved_documents": statistics.mean(row["retrieved_doc_count"] for row in rows),
        "average_context_char_count": statistics.mean(row["context_char_count"] for row in rows),
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "prompt_tokens": usage["prompt_tokens"] if usage["token_usage_available"] else None,
        "completion_tokens": usage["completion_tokens"] if usage["token_usage_available"] else None,
        "total_tokens": usage["total_tokens"] if usage["token_usage_available"] else None,
        "estimated_openai_cost_usd": estimated_cost,
        "cost_note": (
            "Estimated from logged token usage using standard per-million-token rates: "
            "gpt-4o-mini input $0.15/output $0.60; gpt-5 input $1.25/output $10.00."
        )
        if estimated_cost is not None
        else "Token usage was not fully logged by the API responses or pricing was unavailable.",
    }


def write_metrics(rows):
    path = RESULTS_DIR / "results_phase2_vector_topk_answer_quality.csv"
    fieldnames = [
        "k",
        "n",
        "Faithfulness",
        "Correctness",
        "Answer_Relevance",
        "Information_Recall",
        "average_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
        "average_retrieved_documents",
        "average_context_char_count",
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
    return path


def write_latex(rows):
    path = RESULTS_DIR / "table_vector_topk_answer_quality.tex"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table*}[htbp]\n")
        fh.write("\\centering\n")
        fh.write("\\caption{Vector-RAG answer-quality sensitivity by retrieval depth on the manually curated benchmark.}\n")
        fh.write("\\label{tab:phase2_vector_topk_answer_quality}\n")
        fh.write("\\small\n")
        fh.write("\\begin{tabular}{ccccccccc}\n")
        fh.write("\\toprule\n")
        fh.write("$k$ & Faith. & Corr. & Rel. & Recall & Mean s & Median s & P95 s & Avg. Docs \\\\\n")
        fh.write("\\midrule\n")
        for row in rows:
            fh.write(
                f"{row['k']} & {row['Faithfulness']:.3f} & {row['Correctness']:.3f} & "
                f"{row['Answer_Relevance']:.3f} & {row['Information_Recall']:.3f} & "
                f"{row['average_latency_seconds']:.2f} & {row['median_latency_seconds']:.2f} & "
                f"{row['p95_latency_seconds']:.2f} & {row['average_retrieved_documents']:.2f} \\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")
        fh.write("\\end{table*}\n")
    return path


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--dataset", choices=["human"], default="human")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_human_dataset(str(DATASET_PATH))
    all_metrics = []
    output_paths = []
    for k in args.k:
        rows, out_path = await run_for_k(k, questions, args.concurrency)
        output_paths.append(out_path)
        all_metrics.append(summarize(k, rows))

    metrics_path = write_metrics(all_metrics)
    table_path = write_latex(all_metrics)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {table_path}")
    for path in output_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
