#!/usr/bin/env python3
"""Run the additional conversational architecture evaluations.

The script evaluates the lexical, hybrid, and three graph-ablation variants on
the published Tier 4 subset. Generated outputs are written to EVAL_OUTPUT_DIR.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

from neo4j import GraphDatabase
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluation.tier4_conversational_eval import (  # noqa: E402
    DEFAULT_SUBSET_IDS,
    ConversationalJudge,
    aggregate_scores,
    evaluate_system_results,
    load_dataset,
)
from phase2_ablation_no_router import NoRouterInvestigator  # noqa: E402
from phase2_ablation_no_weak_nodes import StructuredOnlyInvestigator  # noqa: E402
from phase2_hybrid_answer_quality import HybridRetriever  # noqa: E402
from phase2_lexical_answer_quality import Neo4jLexicalRetriever  # noqa: E402
from agenticGraphRAG.investigator import SHABInvestigator  # noqa: E402


RESULTS_DIR = ROOT / "evaluation" / "results"
ANSWER_MODEL_FLAT = os.getenv("OPENAI_CONVERSATION_MODEL", "gpt-5-2025-08-07")
ANSWER_MODEL_GRAPH = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini-2024-07-18")
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-2025-08-07")

ARCH_CONFIG = {
    "lexical": {
        "label": "Lexical Full-Text RAG",
        "slug": "lexical_full_text_rag",
        "type": "flat_lexical",
    },
    "hybrid": {
        "label": "Hybrid Dense+Lexical RAG",
        "slug": "hybrid_dense_lexical_rag",
        "type": "flat_hybrid",
    },
    "no_reflection": {
        "label": "GraphRAG without reflection",
        "slug": "graphrag_without_reflection",
        "type": "graph",
        "investigator": SHABInvestigator,
        "iterations": 1,
    },
    "no_router": {
        "label": "GraphRAG without intent routing",
        "slug": "graphrag_without_intent_routing",
        "type": "graph",
        "investigator": NoRouterInvestigator,
        "iterations": 4,
    },
    "structured_only": {
        "label": "Structured-only GraphRAG",
        "slug": "structured_only_graphrag",
        "type": "graph",
        "investigator": StructuredOnlyInvestigator,
        "iterations": 4,
    },
}


def output_path(output_dir: Path, arch: str) -> Path:
    return output_dir / f"conversational_{ARCH_CONFIG[arch]['slug']}_results.json"


def load_existing_payload(output_dir: Path, arch: str) -> Optional[dict[str, Any]]:
    path = output_path(output_dir, arch)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_architecture_payload(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arch = payload["architecture_key"]
    output_path(output_dir, arch).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def history_text(history: list[dict[str, str]]) -> str:
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-6:])


def retrieval_query(history: list[dict[str, str]], user_message: str) -> str:
    parts = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-6:]]
    parts.append(f"User: {user_message}")
    return "\n".join(parts)


def context_text(context_docs: list[Any]) -> str:
    parts = []
    for doc in context_docs:
        if isinstance(doc, dict):
            parts.append(doc.get("context_text") or json.dumps(doc, ensure_ascii=False, sort_keys=True))
        else:
            parts.append(str(doc))
    return "\n".join(parts) or "NO CONTEXT RETRIEVED."


def answer_field(config: dict[str, Any]) -> str:
    return "baseline_answer" if config["type"].startswith("flat") else "agent_answer"


async def answer_with_flat(client: AsyncOpenAI, arch: str, retriever: Any, history: list[dict[str, str]], user_message: str, k: int) -> tuple[str, list[Any]]:
    query = retrieval_query(history, user_message)
    if arch == "lexical":
        docs = await asyncio.to_thread(retriever.retrieve_expanded, query, k)
    else:
        docs, _, _ = await asyncio.to_thread(retriever.retrieve, query, k)
    prompt = f"""You are an assistant answering questions about the Swiss Commercial Register.
Use the prior conversation when resolving pronouns or ellipsis.
Answer only based on the provided context and conversation.

Conversation so far:
{history_text(history)}

Retrieved context:
{context_text(docs)}

Current user message: {user_message}

Answer carefully. If the information is not available, say so briefly.
"""
    response = await client.chat.completions.create(
        model=ANSWER_MODEL_FLAT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content, docs


async def with_retries(label: str, fn: Any, max_retries: int, retry_sleep: float) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"{label} failed after {max_retries + 1} attempts: {last_exc}")


def make_graph_investigator(config: dict[str, Any]) -> tuple[Any, Any]:
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), password),
    )
    cls = config["investigator"]
    investigator = cls(
        driver=driver,
        api_key=os.getenv("OPENAI_API_KEY"),
        model=ANSWER_MODEL_GRAPH,
        iterations=config["iterations"],
        database=os.getenv("NEO4J_DATABASE", "shabdb"),
    )
    return driver, investigator


async def run_architecture(
    arch: str,
    dataset: list[dict[str, Any]],
    k: int,
    concurrency: int,
    output_dir: Path,
    max_retries: int,
    retry_sleep: float,
    force: bool,
) -> dict[str, Any]:
    config = ARCH_CONFIG[arch]
    label = config["label"]
    existing = None if force else load_existing_payload(output_dir, arch)
    results = list(existing.get("results", [])) if existing else []
    completed_ids = {row.get("conversation_id") for row in results}
    start_all = time.time()

    flat_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    retriever = None
    driver = None
    investigator = None
    if config["type"] == "flat_lexical":
        retriever = Neo4jLexicalRetriever()
    elif config["type"] == "flat_hybrid":
        retriever = HybridRetriever(concurrency=concurrency)
    else:
        driver, investigator = make_graph_investigator(config)

    try:
        for conversation in dataset:
            if conversation["conversation_id"] in completed_ids:
                continue
            history: list[dict[str, str]] = []
            turns_out = []
            conv_start = time.time()
            for turn in conversation["turns"]:
                trace: list[str] = []
                field = answer_field(config)
                try:
                    if config["type"].startswith("flat"):
                        answer, docs = await with_retries(
                            f"{arch} flat answer turn {conversation['conversation_id']}-{turn['turn_id']}",
                            lambda: answer_with_flat(flat_client, arch, retriever, history, turn["user_message"], k),
                            max_retries,
                            retry_sleep,
                        )
                        retrieved_context = docs
                    else:
                        def trace_callback(msg: str) -> None:
                            trace.append(msg)

                        question = (
                            turn["user_message"]
                            + "\n\n[SYSTEM INSTRUCTION: Answer the user question concisely. Use your tools strictly.]"
                        )
                        response = await with_retries(
                            f"{arch} graph answer turn {conversation['conversation_id']}-{turn['turn_id']}",
                            lambda: asyncio.to_thread(
                                investigator.ask,
                                question,
                                chat_history=history,
                                trace_callback=trace_callback,
                            ),
                            max_retries,
                            retry_sleep,
                        )
                        answer = response.get("answer", "No answer generated.")
                        retrieved_context = response.get("data", [])
                except Exception as exc:
                    answer = f"Error: {exc}"
                    retrieved_context = []

                history.append({"role": "user", "content": turn["user_message"]})
                history.append({"role": "assistant", "content": answer})
                turn_out = {
                    "turn_id": turn["turn_id"],
                    "user_message": turn["user_message"],
                    "expected_answer": turn["expected_answer"],
                    "evaluation_focus": turn["evaluation_focus"],
                    "expected_tool_behavior": turn.get("expected_tool_behavior", []),
                    "retrieved_context": retrieved_context,
                    "trace_log": trace,
                }
                turn_out[field] = answer
                turns_out.append(turn_out)

            results.append({
                "conversation_id": conversation["conversation_id"],
                "scenario_type": conversation["scenario_type"],
                "conversation_goal": conversation["conversation_goal"],
                "latency_seconds": time.time() - conv_start,
                "turns": turns_out,
            })
            save_architecture_payload(output_dir, {
                "architecture": label,
                "architecture_key": arch,
                "status": "partial",
                "results": results,
                "turn_scores": [],
                "conversation_scores": [],
                "summary": {
                    "architecture": label,
                    "architecture_key": arch,
                    "n_conversations_requested": len(dataset),
                    "n_conversations_completed": len(results),
                    "n_turns_completed": sum(len(c["turns"]) for c in results),
                    "answer_model": ANSWER_MODEL_GRAPH if config["type"] == "graph" else ANSWER_MODEL_FLAT,
                    "judge_model": JUDGE_MODEL,
                    "retrieval_setting": f"k={k}" if config["type"].startswith("flat") else f"iterations={config['iterations']}",
                },
            })
    finally:
        if retriever and hasattr(retriever, "close"):
            retriever.close()
        if driver:
            driver.close()

    judge = ConversationalJudge(api_key=os.getenv("OPENAI_API_KEY"), concurrency=concurrency)
    is_graph = config["type"] == "graph"
    turn_scores, convo_scores = await evaluate_system_results(results, is_graph, judge)
    summary = aggregate_scores(turn_scores, convo_scores)
    summary.update({
        "architecture": label,
        "architecture_key": arch,
        "n_conversations_requested": len(dataset),
        "n_conversations_completed": len(results),
        "n_conversations": len(results),
        "n_turns": sum(len(c["turns"]) for c in results),
        "mean_latency_seconds": statistics.mean([c["latency_seconds"] for c in results]) if results else None,
        "total_runtime_seconds": time.time() - start_all,
        "answer_model": ANSWER_MODEL_GRAPH if config["type"] == "graph" else ANSWER_MODEL_FLAT,
        "judge_model": JUDGE_MODEL,
        "retrieval_setting": f"k={k}" if config["type"].startswith("flat") else f"iterations={config['iterations']}",
    })
    return {
        "architecture": label,
        "architecture_key": arch,
        "status": "complete",
        "results": results,
        "turn_scores": turn_scores,
        "conversation_scores": convo_scores,
        "summary": summary,
    }


def write_outputs(payloads: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for payload in payloads:
        save_architecture_payload(output_dir, payload)
        summary_rows.append(payload["summary"])

    fieldnames = [
        "architecture", "architecture_key", "n_conversations", "n_turns",
        "n_conversations_requested", "n_conversations_completed",
        "Correctness", "Answer_Relevance", "Information_Recall",
        "Exact_Match_Normalized", "Turn_Success_Rate",
        "Conversation_Goal_Completion_Rate", "Context_Carryover_Accuracy",
        "Tool_Transition_Accuracy", "mean_latency_seconds", "total_runtime_seconds",
        "answer_model", "judge_model", "retrieval_setting",
    ]
    with (output_dir / "conversational_missing_architectures_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", choices=sorted(ARCH_CONFIG), default=sorted(ARCH_CONFIG))
    parser.add_argument("--conversation-ids", nargs="*", default=DEFAULT_SUBSET_IDS)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", help="Rerun and overwrite existing architecture outputs.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    dataset = load_dataset(conversation_ids=args.conversation_ids)
    payloads = []
    for arch in args.architectures:
        print(f"Running conversational architecture: {ARCH_CONFIG[arch]['label']}")
        payloads.append(await run_architecture(
            arch,
            dataset,
            args.k,
            args.concurrency,
            args.output_dir,
            args.max_retries,
            args.retry_sleep,
            args.force,
        ))
        write_outputs(payloads, args.output_dir)
    print(f"Wrote conversational missing architecture outputs to {args.output_dir}.")


if __name__ == "__main__":
    asyncio.run(main())
