#!/usr/bin/env python3
"""Run hybrid dense+lexical RAG answer-quality sensitivity.

Dense retrieval uses the existing full-corpus Chroma/all-MiniLM-L6-v2 baseline.
Lexical retrieval uses Neo4j `global_search` in `shabdb`. Candidates are fused
with reciprocal rank fusion and evaluated with the same answer-generation and
Tier 3 LLM-as-a-judge protocol as the vector and lexical baselines.
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
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluation.baseline_rag.vector_rag import NaiveVectorRAG
from evaluation.tier3_regas_human_dataset import load_human_dataset
from phase2_lexical_answer_quality import (
    ANSWER_MODEL,
    DATASET_PATH,
    RESULTS_DIR,
    JUDGE_MODEL,
    Neo4jLexicalRetriever,
    build_answer_prompt,
    estimate_cost,
    generate_answer,
    judge_all,
    percentile,
    sum_usage,
)


CHROMA_PATH = ROOT / "evaluation" / "baseline_rag" / "chroma_db"
RRF_CONSTANT = 60


def dense_context(doc):
    return f"SOURCE: dense_chroma\nTEXT: {doc}"


def lexical_context(doc):
    return doc.get("context_text") or "\n".join(
        part
        for part in [
            "SOURCE: neo4j_global_search",
            f"UID: {doc.get('uid')}",
            f"LABELS: {', '.join(doc.get('labels') or [])}",
            f"SCORE: {doc.get('score')}",
            f"DATE: {doc.get('date')}" if doc.get("date") else "",
            f"RUBRIC: {doc.get('rubric')}" if doc.get("rubric") else "",
            f"SUB_RUBRIC: {doc.get('sub_rubric')}" if doc.get("sub_rubric") else "",
            f"NAME: {doc.get('name')}" if doc.get("name") else "",
            f"TEXT: {doc.get('text')}" if doc.get("text") else "",
        ]
        if part
    )


def item_key(source, payload):
    if source == "lexical":
        return f"lexical:{payload.get('uid') or lexical_context(payload)}"
    return f"dense:{str(payload)}"


def build_item(source, payload, rank, rrf_score):
    if source == "lexical":
        item = dict(payload)
        item.update(
            {
                "source": "neo4j_global_search",
                "rank": rank,
                "rrf_score": rrf_score,
                "context_text": lexical_context(payload),
            }
        )
        return item
    return {
        "source": "dense_chroma",
        "rank": rank,
        "rrf_score": rrf_score,
        "text": str(payload),
        "context_text": dense_context(payload),
    }


def reciprocal_rank_fusion(dense_docs, lexical_docs, final_k):
    fused = {}
    source_ranks = {}

    for source, docs in [("dense", dense_docs), ("lexical", lexical_docs)]:
        for rank, payload in enumerate(docs, start=1):
            key = item_key(source, payload)
            fused.setdefault(key, {"source": source, "payload": payload, "score": 0.0})
            fused[key]["score"] += 1.0 / (RRF_CONSTANT + rank)
            source_ranks.setdefault(key, {})[source] = rank

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    output = []
    for item in ranked[:final_k]:
        primary_rank = min(source_ranks[item_key(item["source"], item["payload"])].values())
        built = build_item(item["source"], item["payload"], primary_rank, item["score"])
        built["source_ranks"] = source_ranks[item_key(item["source"], item["payload"])]
        output.append(built)
    return output


class HybridRetriever:
    def __init__(self, concurrency):
        self.dense = NaiveVectorRAG(
            api_key=os.getenv("OPENAI_API_KEY"),
            db_path=str(CHROMA_PATH),
            concurrency=concurrency,
        )
        self.lexical = Neo4jLexicalRetriever()

    def close(self):
        self.lexical.close()

    def retrieve(self, question, final_k):
        candidate_limit = max(final_k * 4, 25)
        dense_docs = self.dense.retrieve_expanded(
            question,
            k_per_query=candidate_limit,
            max_docs=candidate_limit,
        )
        lexical_docs = self.lexical.retrieve_expanded(question, candidate_limit)
        fused_docs = reciprocal_rank_fusion(dense_docs, lexical_docs, final_k)
        return fused_docs, len(dense_docs), len(lexical_docs)


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
        "average_dense_candidates": statistics.mean(row["dense_candidate_count"] for row in rows),
        "average_lexical_candidates": statistics.mean(row["lexical_candidate_count"] for row in rows),
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "fusion_method": f"reciprocal_rank_fusion_c_{RRF_CONSTANT}",
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


async def run_for_k(k, questions, concurrency):
    retriever = HybridRetriever(concurrency)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(concurrency)
    judge_sem = asyncio.Semaphore(concurrency)
    out_path = RESULTS_DIR / f"outputs_hybrid_answer_quality_k{k}.jsonl"

    async def process_question(q):
        async with sem:
            start = time.time()
            context_docs, dense_count, lexical_count = await asyncio.to_thread(
                retriever.retrieve,
                q["question_text"],
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
            return {
                "k": k,
                "question_id": q["question_id"],
                "question": q["question_text"],
                "tier": q["tier"],
                "category": q["category"],
                "expected_answer": q["expected_answer"],
                "baseline_answer": answer,
                "retrieved_context": context_docs,
                "retrieved_doc_count": len(context_docs),
                "dense_candidate_count": dense_count,
                "lexical_candidate_count": lexical_count,
                "context_char_count": len(context_text),
                "latency_seconds": latency,
                "retrieval_backend": "hybrid_dense_chroma_plus_neo4j_global_search",
                "fusion_method": f"reciprocal_rank_fusion_c_{RRF_CONSTANT}",
                "dense_backend": "chroma_all-MiniLM-L6-v2_full_corpus",
                "lexical_backend": "neo4j_global_search_shabdb",
                "answer_model": ANSWER_MODEL,
                "judge_model": JUDGE_MODEL,
                "answer_usage": answer_usage,
                **judge_usages,
                **scores,
            }

    results = []
    try:
        with out_path.open("w", encoding="utf-8") as out:
            tasks = [process_question(q) for q in questions]
            with tqdm.tqdm(total=len(tasks), desc=f"Hybrid answer quality k={k}") as pbar:
                for task in asyncio.as_completed(tasks):
                    row = await task
                    results.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    pbar.update(1)
    finally:
        retriever.close()
    return results, out_path


def write_metrics(rows):
    path = RESULTS_DIR / "results_phase2_hybrid_answer_quality.csv"
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
        "average_dense_candidates",
        "average_lexical_candidates",
        "answer_model",
        "judge_model",
        "fusion_method",
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
    path = RESULTS_DIR / "table_hybrid_answer_quality.tex"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table*}[htbp]\n")
        fh.write("\\centering\n")
        fh.write("\\caption{Hybrid dense+lexical answer-quality sensitivity by retrieval depth on the manually curated benchmark.}\n")
        fh.write("\\label{tab:phase2_hybrid_answer_quality}\n")
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


def read_metric_csv(path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {int(row["k"]): row for row in csv.DictReader(fh)}


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
