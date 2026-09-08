#!/usr/bin/env python3
"""Run Neo4j full-text lexical-RAG answer-quality sensitivity.

This script evaluates a lexical baseline over the 60-question human benchmark.
Retrieval uses the existing Neo4j `global_search` full-text index in `shabdb`.
It does not rebuild embeddings and does not use Chroma. OpenAI is used only for
answer generation and LLM-as-a-judge scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

import tqdm
from neo4j import GraphDatabase, READ_ACCESS
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluation.tier3_regas_human_dataset import load_human_dataset


RESULTS_DIR = ROOT / "evaluation" / "results"
DATASET_PATH = ROOT / "evaluation" / "datasets" / "golden_dataset.json"

ANSWER_MODEL = os.getenv("OPENAI_BASELINE_MODEL", "gpt-4o-mini-2024-07-18")
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-2025-08-07")
MODEL_PRICES_PER_MILLION = {
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
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def build_query_variants(question):
    variants = [question.strip()]
    quoted = re.findall(r'"([^"]+)"', question)
    variants.extend(quoted)

    patterns = [
        r"with whom does (.+?) share an activity\??",
        r"show all the companies with which (.+?) is affiliated\.?",
        r"show all the companies with which (.+?) is connected\.?",
        r"what companies are connected to (.+?)\??",
        r"have any notices of dissolution or bankruptcy been published for (.+?)\??",
        r"what is the registered head office or base of (.+?)\??",
        r"what is the registered address or base of (.+?)\??",
    ]

    extracted = None
    lower_q = question.lower()
    for pattern in patterns:
        match = re.search(pattern, lower_q, flags=re.IGNORECASE)
        if match:
            start, end = match.span(1)
            extracted = question[start:end].strip().strip(".?")
            break

    if extracted:
        variants.append(extracted)
        if "," in extracted:
            parts = [p.strip() for p in extracted.split(",") if p.strip()]
            if len(parts) == 2:
                variants.append(" ".join(reversed(parts)))

    deduped = []
    seen = set()
    for item in variants:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def to_lucene_required_terms(text):
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿ]+", text, flags=re.UNICODE)
    tokens = [token for token in tokens if len(token) > 1]
    if not tokens:
        return ""
    return " ".join(f"+{token}" for token in tokens)


def context_from_record(record):
    labels = record.get("labels") or []
    text = record.get("text") or ""
    name = record.get("name") or ""
    parts = [
        f"UID: {record.get('uid')}",
        f"LABELS: {', '.join(labels)}",
        f"SCORE: {record.get('score')}",
    ]
    if record.get("date"):
        parts.append(f"DATE: {record.get('date')}")
    if record.get("rubric"):
        parts.append(f"RUBRIC: {record.get('rubric')}")
    if record.get("sub_rubric"):
        parts.append(f"SUB_RUBRIC: {record.get('sub_rubric')}")
    if name:
        parts.append(f"NAME: {name}")
    if text:
        parts.append(f"TEXT: {text}")
    return "\n".join(parts)


class Neo4jLexicalRetriever:
    """Small retrieval wrapper around the live Neo4j Lucene full-text index."""

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "shabdb")
        if not self.password:
            raise RuntimeError("NEO4J_PASSWORD is required")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def retrieve(self, query_text, limit):
        lucene_query = to_lucene_required_terms(query_text)
        if not lucene_query:
            return []
        cypher = """
        CALL db.index.fulltext.queryNodes("global_search", $ft_query) YIELD node, score
        WHERE score > 0.3
        RETURN
            labels(node) AS labels,
            score AS score,
            node.uid AS uid,
            node.name AS name,
            node.date AS date,
            node.rubric AS rubric,
            node.sub_rubric AS sub_rubric,
            substring(coalesce(node.text, node.purpose, node.name, ''), 0, 2000) AS text
        ORDER BY score DESC
        LIMIT toInteger($limit)
        """
        with self.driver.session(
            database=self.database,
            default_access_mode=READ_ACCESS,
        ) as session:
            return session.run(cypher, ft_query=lucene_query, limit=limit).data()

    def retrieve_expanded(self, question, k):
        docs = []
        seen = set()
        for variant in build_query_variants(question):
            for record in self.retrieve(variant, k):
                key = record.get("uid") or context_from_record(record)
                if key in seen:
                    continue
                seen.add(key)
                record["query_variant"] = variant
                record["context_text"] = context_from_record(record)
                docs.append(record)
                if len(docs) >= k:
                    return docs
        return docs


def build_answer_prompt(question, context_docs):
    context = "\n\n".join(doc.get("context_text", context_from_record(doc)) for doc in context_docs)
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


async def run_for_k(k, questions, concurrency):
    retriever = Neo4jLexicalRetriever()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(concurrency)
    judge_sem = asyncio.Semaphore(concurrency)
    out_path = RESULTS_DIR / f"outputs_lexical_answer_quality_k{k}.jsonl"

    async def process_question(q):
        async with sem:
            start = time.time()
            context_docs = await asyncio.to_thread(retriever.retrieve_expanded, q["question_text"], k)
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
                "context_char_count": len(context_text),
                "latency_seconds": latency,
                "retrieval_backend": "neo4j_global_search",
                "retrieval_database": os.getenv("NEO4J_DATABASE", "shabdb"),
                "indexed_label": "BaseNode",
                "indexed_properties": ["name", "text", "rubric"],
                "context_fields": ["uid", "labels", "score", "date", "rubric", "sub_rubric", "name", "text"],
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
            with tqdm.tqdm(total=len(tasks), desc=f"Lexical answer quality k={k}") as pbar:
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
    path = RESULTS_DIR / "results_phase2_lexical_answer_quality.csv"
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
    path = RESULTS_DIR / "table_lexical_answer_quality.tex"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table*}[htbp]\n")
        fh.write("\\centering\n")
        fh.write("\\caption{Lexical full-text answer-quality sensitivity by retrieval depth on the manually curated benchmark.}\n")
        fh.write("\\label{tab:phase2_lexical_answer_quality}\n")
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


def load_vector_comparison():
    path = RESULTS_DIR / "results_phase2_vector_topk_answer_quality.csv"
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
