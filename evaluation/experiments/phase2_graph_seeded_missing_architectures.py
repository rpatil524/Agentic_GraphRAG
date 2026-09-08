#!/usr/bin/env python3
"""Run the additional graph-seeded architecture benchmarks.

The script uses the published 300-question benchmark and the same lexical,
hybrid, and graph-ablation implementations used for the human benchmark. It
does not rebuild embeddings, mutate Neo4j, or create new questions.
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
from typing import Any, Optional, Set

import tqdm
from neo4j import GraphDatabase, READ_ACCESS
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_ablation_no_reflection import (  # noqa: E402
    ANSWER_MODEL,
    JUDGE_MODEL,
    count_failed_tool_calls,
    count_tool_calls,
    estimate_judge_cost,
    is_empty_answer,
    judge_all,
    percentile,
    sum_usage,
)
from phase2_ablation_no_router import NoRouterInvestigator  # noqa: E402
from phase2_ablation_no_weak_nodes import StructuredOnlyInvestigator  # noqa: E402
from phase2_hybrid_answer_quality import HybridRetriever  # noqa: E402
from phase2_lexical_answer_quality import (  # noqa: E402
    Neo4jLexicalRetriever,
    estimate_cost,
    generate_answer,
)
from agenticGraphRAG.investigator import SHABInvestigator  # noqa: E402


DATASET_PATH = ROOT / "evaluation" / "datasets" / "automated_dataset.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
DEFAULT_K = 20

ARCHITECTURES = {
    "lexical": {
        "label": "Lexical Full-Text RAG",
        "output": "graph_seeded_lexical_full_text_rag.jsonl",
        "configuration": "Neo4j global_search full-text, k=20",
    },
    "hybrid": {
        "label": "Hybrid Dense+Lexical RAG",
        "output": "graph_seeded_hybrid_dense_lexical_rag.jsonl",
        "configuration": "Chroma dense + Neo4j global_search, reciprocal rank fusion, k=20",
    },
    "no_reflection": {
        "label": "GraphRAG without reflection",
        "output": "graph_seeded_graphrag_without_reflection.jsonl",
        "configuration": "SHABInvestigator with max_iterations=1",
    },
    "no_router": {
        "label": "GraphRAG without intent routing",
        "output": "graph_seeded_graphrag_without_intent_routing.jsonl",
        "configuration": "NoRouterInvestigator exposing the full tool schema",
    },
    "structured_only": {
        "label": "Structured-only GraphRAG",
        "output": "graph_seeded_structured_only_graphrag.jsonl",
        "configuration": "StructuredOnlyInvestigator filtering weak nodes at retrieval time",
    },
}


def load_dataset(
    limit: Optional[int] = None,
    question_ids: Optional[Set[str]] = None,
) -> list[dict[str, Any]]:
    items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows = []
    for item in items:
        qid = item["question_id"]
        if question_ids and qid not in question_ids:
            continue
        rows.append(
            {
                "question_id": qid,
                "question": item["question_text"],
                "expected_answer": item["expected_answer"],
                "tier": item.get("tier"),
                "difficulty_level": item.get("difficulty_level"),
                "category": (item.get("ground_truth_subgraph") or {}).get("type", ""),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def preflight_neo4j() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "shabdb")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(
            database=database,
            default_access_mode=READ_ACCESS,
        ) as session:
            session.run("RETURN 1 AS ok").single()
    except Exception as exc:
        raise RuntimeError(
            "Neo4j preflight failed. The graph-seeded architecture benchmark "
            "requires read-only access to the shabdb Neo4j database in the same "
            "execution context as OPENAI_API_KEY."
        ) from exc
    finally:
        driver.close()


def json_context_text(context: Any) -> str:
    if isinstance(context, str):
        return context
    return json.dumps(context, ensure_ascii=False)


def load_completed_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("question_id"):
                rows[row["question_id"]] = row
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in sorted(rows, key=lambda item: item["question_id"]):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def graph_investigator_for_architecture(architecture: str, max_iterations: int):
    investigator_class = {
        "no_reflection": SHABInvestigator,
        "no_router": NoRouterInvestigator,
        "structured_only": StructuredOnlyInvestigator,
    }[architecture]
    iterations = 1 if architecture == "no_reflection" else max_iterations
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "shabdb")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    investigator = investigator_class(
        driver=driver,
        api_key=os.getenv("OPENAI_API_KEY"),
        model=ANSWER_MODEL,
        iterations=iterations,
        database=database,
    )
    return driver, investigator


def run_graph_answer(architecture: str, question: dict[str, Any], max_iterations: int) -> dict[str, Any]:
    driver, investigator = graph_investigator_for_architecture(architecture, max_iterations)
    trace: list[str] = []

    def trace_callback(message: str) -> None:
        trace.append(message)

    start = time.time()
    try:
        response = investigator.ask(question["question"], trace_callback=trace_callback)
        answer = response.get("answer", "No answer generated.")
        retrieved_context = response.get("data", [])
        new_messages = response.get("new_messages", [])
        error_message = ""
    except Exception as exc:
        answer = f"Error: {exc}"
        retrieved_context = []
        new_messages = []
        error_message = str(exc)
    finally:
        driver.close()

    latency = time.time() - start
    assistant_llm_calls = sum(1 for msg in new_messages if msg.get("role") == "assistant")
    router_calls = 0 if architecture == "no_router" else 1
    return {
        "architecture_key": architecture,
        "architecture": ARCHITECTURES[architecture]["label"],
        "configuration": ARCHITECTURES[architecture]["configuration"],
        "question_id": question["question_id"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "tier": question.get("tier"),
        "difficulty_level": question.get("difficulty_level"),
        "category": question.get("category"),
        "agent_answer": answer,
        "retrieved_context": retrieved_context,
        "trace_log": trace,
        "latency_seconds": latency,
        "tool_calls": count_tool_calls(trace),
        "failed_tool_calls": count_failed_tool_calls(trace),
        "empty_answer": is_empty_answer(answer),
        "agent_llm_calls_estimated": assistant_llm_calls + router_calls,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "retrieval_setting": "graph_tools",
        "error_status": "failure_or_empty" if error_message or is_empty_answer(answer) else "",
        "error_message": error_message,
    }


async def score_rows(rows: list[dict[str, Any]], concurrency: int) -> list[dict[str, Any]]:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(concurrency)

    async def score_row(row: dict[str, Any]) -> dict[str, Any]:
        return await judge_or_preserve(client, sem, row)

    scored: list[dict[str, Any]] = []
    tasks = [score_row(row) for row in rows]
    with tqdm.tqdm(total=len(tasks), desc="Judge graph-seeded rows") as pbar:
        for task in asyncio.as_completed(tasks):
            scored.append(await task)
            pbar.update(1)
    return scored


async def generate_answer_with_retry(
    client: AsyncOpenAI,
    question: str,
    context_docs: list[dict[str, Any]],
    attempts: int = 4,
) -> tuple[str, str, dict[str, Any]]:
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await generate_answer(client, question, context_docs)
        except Exception as exc:
            last_error = exc
            if "429" in str(exc) or "timeout" in str(exc).lower():
                await asyncio.sleep(5 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"answer_generation_failed: {last_error}")


async def judge_or_preserve(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    row: dict[str, Any],
) -> dict[str, Any]:
    answer = row.get("agent_answer") or row.get("baseline_answer") or ""
    try:
        scores, usages = await judge_all(
            client,
            sem,
            row["question"],
            row["expected_answer"],
            answer,
            json_context_text(row.get("retrieved_context")),
        )
        row.update(scores)
        row.update(usages)
    except Exception as exc:
        row["error_status"] = "failure_or_empty"
        row["judge_error"] = str(exc)
    return row


async def run_lexical(questions: list[dict[str, Any]], k: int, concurrency: int) -> list[dict[str, Any]]:
    retriever = Neo4jLexicalRetriever()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    answer_sem = asyncio.Semaphore(concurrency)
    judge_sem = asyncio.Semaphore(concurrency)

    async def process(question: dict[str, Any]) -> dict[str, Any]:
        async with answer_sem:
            start = time.time()
            context_docs: list[dict[str, Any]] = []
            context_text = ""
            answer_usage: dict[str, Any] = {}
            try:
                context_docs = await asyncio.to_thread(retriever.retrieve_expanded, question["question"], k)
                answer, context_text, answer_usage = await generate_answer_with_retry(
                    client,
                    question["question"],
                    context_docs,
                )
                error_status = "failure_or_empty" if is_empty_answer(answer) else ""
                error_message = ""
            except Exception as exc:
                answer = f"Error: {exc}"
                error_status = "failure_or_empty"
                error_message = str(exc)
            row = {
                "architecture_key": "lexical",
                "architecture": ARCHITECTURES["lexical"]["label"],
                "configuration": ARCHITECTURES["lexical"]["configuration"],
                "question_id": question["question_id"],
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "tier": question.get("tier"),
                "difficulty_level": question.get("difficulty_level"),
                "category": question.get("category"),
                "baseline_answer": answer,
                "retrieved_context": context_docs,
                "retrieved_doc_count": len(context_docs),
                "context_char_count": len(context_text),
                "latency_seconds": time.time() - start,
                "retrieval_setting": f"neo4j_global_search_k{k}",
                "answer_model": ANSWER_MODEL,
                "judge_model": JUDGE_MODEL,
                "answer_usage": answer_usage,
                "error_status": error_status,
                "error_message": error_message,
            }
            return await judge_or_preserve(client, judge_sem, row)

    try:
        return await run_async_tasks("Lexical graph-seeded", [process(q) for q in questions])
    finally:
        retriever.close()


async def run_hybrid(questions: list[dict[str, Any]], k: int, concurrency: int) -> list[dict[str, Any]]:
    retriever = HybridRetriever(concurrency)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    answer_sem = asyncio.Semaphore(concurrency)
    judge_sem = asyncio.Semaphore(concurrency)

    async def process(question: dict[str, Any]) -> dict[str, Any]:
        async with answer_sem:
            start = time.time()
            context_docs: list[dict[str, Any]] = []
            context_text = ""
            answer_usage: dict[str, Any] = {}
            dense_count = 0
            lexical_count = 0
            try:
                context_docs, dense_count, lexical_count = await asyncio.to_thread(
                    retriever.retrieve,
                    question["question"],
                    k,
                )
                answer, context_text, answer_usage = await generate_answer_with_retry(
                    client,
                    question["question"],
                    context_docs,
                )
                error_status = "failure_or_empty" if is_empty_answer(answer) else ""
                error_message = ""
            except Exception as exc:
                answer = f"Error: {exc}"
                error_status = "failure_or_empty"
                error_message = str(exc)
            row = {
                "architecture_key": "hybrid",
                "architecture": ARCHITECTURES["hybrid"]["label"],
                "configuration": ARCHITECTURES["hybrid"]["configuration"],
                "question_id": question["question_id"],
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "tier": question.get("tier"),
                "difficulty_level": question.get("difficulty_level"),
                "category": question.get("category"),
                "baseline_answer": answer,
                "retrieved_context": context_docs,
                "retrieved_doc_count": len(context_docs),
                "dense_candidate_count": dense_count,
                "lexical_candidate_count": lexical_count,
                "context_char_count": len(context_text),
                "latency_seconds": time.time() - start,
                "retrieval_setting": f"rrf_dense_plus_lexical_k{k}",
                "answer_model": ANSWER_MODEL,
                "judge_model": JUDGE_MODEL,
                "answer_usage": answer_usage,
                "error_status": error_status,
                "error_message": error_message,
            }
            return await judge_or_preserve(client, judge_sem, row)

    try:
        return await run_async_tasks("Hybrid graph-seeded", [process(q) for q in questions])
    finally:
        retriever.close()


async def run_async_tasks(description: str, tasks: list[asyncio.Future]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tqdm.tqdm(total=len(tasks), desc=description) as pbar:
        for task in asyncio.as_completed(tasks):
            rows.append(await task)
            pbar.update(1)
    return rows


async def run_graph_architecture(
    architecture: str,
    questions: list[dict[str, Any]],
    max_iterations: int,
    judge_concurrency: int,
) -> list[dict[str, Any]]:
    answer_rows: list[dict[str, Any]] = []
    for question in tqdm.tqdm(questions, desc=f"{ARCHITECTURES[architecture]['label']} answers"):
        answer_rows.append(run_graph_answer(architecture, question, max_iterations))
    return await score_rows(answer_rows, judge_concurrency)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metric_mean(metric: str):
        values = [
            float(row[metric])
            for row in rows
            if isinstance(row.get(metric), (int, float))
        ]
        return statistics.mean(values) if values else None

    latencies = [row["latency_seconds"] for row in rows if isinstance(row.get("latency_seconds"), (int, float))]
    usage = sum_usage(rows)
    cost = estimate_cost(rows)
    if cost is None:
        cost = estimate_judge_cost(rows)
    total_tool_calls = sum((row.get("tool_calls") or 0) for row in rows)
    return {
        "architecture": rows[0]["architecture"] if rows else "",
        "n": len(rows),
        "n_scored": sum(
            all(isinstance(row.get(metric), (int, float)) for metric in (
                "Faithfulness", "Correctness", "Answer_Relevance", "Information_Recall"
            ))
            for row in rows
        ),
        "Faithfulness": metric_mean("Faithfulness"),
        "Correctness": metric_mean("Correctness"),
        "Answer_Relevance": metric_mean("Answer_Relevance"),
        "Information_Recall": metric_mean("Information_Recall"),
        "average_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds": percentile(latencies, 0.95) if latencies else None,
        "average_tool_calls": statistics.mean(row.get("tool_calls") or 0 for row in rows),
        "failed_tool_call_rate": (
            sum(row.get("failed_tool_calls") or 0 for row in rows) / total_tool_calls
            if total_tool_calls
            else 0.0
        ),
        "empty_answer_rate": sum(1 for row in rows if row.get("empty_answer")) / len(rows) if rows else None,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "prompt_tokens": usage["prompt_tokens"] if usage["token_usage_available"] else None,
        "completion_tokens": usage["completion_tokens"] if usage["token_usage_available"] else None,
        "total_tokens": usage["total_tokens"] if usage["token_usage_available"] else None,
        "estimated_openai_cost_usd": cost,
    }


def write_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "architecture",
        "n",
        "n_scored",
        "Faithfulness",
        "Correctness",
        "Answer_Relevance",
        "Information_Recall",
        "average_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
        "average_tool_calls",
        "failed_tool_call_rate",
        "empty_answer_rate",
        "answer_model",
        "judge_model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_openai_cost_usd",
        "output_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def select_pending_questions(path: Path, questions: list[dict[str, Any]], force: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed = {} if force else load_completed_rows(path)
    pending = [q for q in questions if q["question_id"] not in completed]
    return pending, list(completed.values())


async def run_architecture(
    architecture: str,
    questions: list[dict[str, Any]],
    output_dir: Path,
    k: int,
    concurrency: int,
    force: bool,
    max_iterations: int,
) -> dict[str, Any]:
    output_path = output_dir / ARCHITECTURES[architecture]["output"]
    pending, completed_rows = select_pending_questions(output_path, questions, force)
    if not pending:
        rows = completed_rows
    elif architecture == "lexical":
        rows = completed_rows + await run_lexical(pending, k, concurrency)
    elif architecture == "hybrid":
        rows = completed_rows + await run_hybrid(pending, k, concurrency)
    else:
        rows = completed_rows + await run_graph_architecture(
            architecture,
            pending,
            max_iterations=max_iterations,
            judge_concurrency=concurrency,
        )
    write_jsonl(output_path, rows)
    summary = summarize_rows(rows)
    try:
        summary["output_file"] = str(output_path.relative_to(ROOT))
    except ValueError:
        summary["output_file"] = str(output_path)
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=sorted(ARCHITECTURES),
        default=sorted(ARCHITECTURES),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-ids", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for answer generation and judging.")
    preflight_neo4j()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    question_ids = set(args.question_ids) if args.question_ids else None
    questions = load_dataset(limit=args.limit, question_ids=question_ids)
    if not questions:
        raise RuntimeError("No graph-seeded benchmark questions selected.")

    summaries = []
    for architecture in args.architectures:
        summaries.append(
            await run_architecture(
                architecture,
                questions,
                args.output_dir,
                k=args.k,
                concurrency=args.concurrency,
                force=args.force,
                max_iterations=args.max_iterations,
            )
        )

    summary_path = args.output_dir / "graph_seeded_missing_architectures_summary.csv"
    write_summary(summary_path, summaries)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
