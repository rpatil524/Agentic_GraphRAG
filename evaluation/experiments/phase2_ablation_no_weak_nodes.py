#!/usr/bin/env python3
"""Run the structured-only/no-weak-node ablation on the human benchmark.

Weak nodes are identified by the graph markers confirmed in Neo4j:
`is_weak = true` and UIDs beginning with `weak_` or `weak_comp_`.
The ablation filters these nodes at query/tool execution time and does not
mutate the production graph.
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


def is_weak_uid(value):
    text = str(value or "")
    return text.startswith("weak_") or text.startswith("weak_comp_")


def row_contains_weak_node(row):
    if not isinstance(row, dict):
        return False
    uid_fields = ["uid", "connected_uid", "event_uid"]
    if row.get("is_weak") is True:
        return True
    return any(is_weak_uid(row.get(field)) for field in uid_fields)


def filter_weak_rows(rows):
    filtered = []
    for row in rows or []:
        if row_contains_weak_node(row):
            continue
        if isinstance(row, dict) and isinstance(row.get("sample_entities"), list):
            row = dict(row)
            row["sample_entities"] = [
                item for item in row["sample_entities"] if not row_contains_weak_node(item)
            ]
            row["total_count"] = len(row["sample_entities"])
        filtered.append(row)
    return filtered


class StructuredOnlyInvestigator(SHABInvestigator):
    """Investigator variant that ignores weak nodes during retrieval."""

    WEAK_NODE_FILTER = (
        "NOT coalesce({var}.is_weak, false) = true "
        "AND NOT {var}.uid STARTS WITH 'weak_' "
        "AND NOT {var}.uid STARTS WITH 'weak_comp_'"
    )

    def _strong_filter(self, var):
        return self.WEAK_NODE_FILTER.format(var=var)

    def execute_query(self, cypher_query, params=None):
        cypher_query = self._inject_weak_filters(cypher_query)
        return filter_weak_rows(super().execute_query(cypher_query, params))

    def _inject_weak_filters(self, cypher_query):
        query = cypher_query

        # Search/disambiguation and generic BaseNode scans.
        query = query.replace(
            "conditions.append(\"(n:Company OR n:Person)\")",
            "conditions.append(\"(n:Company OR n:Person)\")",
        )
        query = query.replace(
            "MATCH (n:BaseNode)\n\t\t{where_clause}",
            "MATCH (n:BaseNode)\n\t\t{where_clause}",
        )

        # Concrete Cypher replacements used by investigator tools.
        query = query.replace(
            "(n:Company OR n:Person)",
            f"(n:Company OR n:Person) AND {self._strong_filter('n')}",
        )
        query = query.replace(
            "WHERE score > 0.3",
            f"WHERE score > 0.3 AND ({self._strong_filter('node')} OR node:Event OR node:NameHub)",
        )
        query = query.replace(
            "MATCH (n:BaseNode) \n\t\t\t\tWHERE toLower(n.name) CONTAINS toLower($name)",
            f"MATCH (n:BaseNode) \n\t\t\t\tWHERE toLower(n.name) CONTAINS toLower($name)\n\t\t\t\tAND {self._strong_filter('n')}",
        )
        query = query.replace(
            "MATCH (h:NameHub)-[:HAS_NAME]-(n:Person)",
            "MATCH (h:NameHub)-[:HAS_NAME]-(n:Person)",
        )
        query = query.replace(
            "RETURN h.uid AS uid, count(DISTINCT n) AS support, min(coalesce(n.is_weak, false)) AS has_strong_match",
            f"WITH h, n, query_tokens, candidate_tokens WHERE {self._strong_filter('n')}\n\t\t\t\tRETURN h.uid AS uid, count(DISTINCT n) AS support, min(coalesce(n.is_weak, false)) AS has_strong_match",
        )

        # Network traversal: do not use weak aliases as bridge actors and do not
        # return weak connected entities.
        query = query.replace(
            "WHERE NOT connected:Event AND NOT connected:NameHub",
            f"WHERE NOT connected:Event AND NOT connected:NameHub\n\t\t\tAND {self._strong_filter('connected')}",
        )
        query = query.replace(
            "AND NOT coalesce(neighbor.is_weak, false) = true",
            f"AND {self._strong_filter('neighbor')}",
        )
        query = query.replace(
            "WITH DISTINCT actor WHERE actor IS NOT NULL AND NOT actor:NameHub",
            f"WITH DISTINCT actor WHERE actor IS NOT NULL AND NOT actor:NameHub\n\t\t\tAND {self._strong_filter('actor')}",
        )
        query = query.replace(
            "AND NOT coalesce(connected.is_weak, false) = true",
            f"AND {self._strong_filter('connected')}",
        )
        query = query.replace(
            "MATCH (actor)-[:HAS_EVENT|ACTED_IN]-(e:Event)--(ctx:Company)\n\t\t\tWHERE ctx.uid <> $uid",
            f"MATCH (actor)-[:HAS_EVENT|ACTED_IN]-(e:Event)--(ctx:Company)\n\t\t\tWHERE ctx.uid <> $uid\n\t\t\tAND {self._strong_filter('ctx')}",
        )

        # Event history: aliases connected through NameHub are restricted to
        # strong nodes so weak aliases cannot pull events into the context.
        query = query.replace(
            "WITH DISTINCT anchor, alias\n\t\tMATCH (alias)-[:HAS_EVENT|ACTED_IN]-(e:Event)",
            f"WITH DISTINCT anchor, alias\n\t\tWHERE {self._strong_filter('anchor')} AND {self._strong_filter('alias')}\n\t\tMATCH (alias)-[:HAS_EVENT|ACTED_IN]-(e:Event)",
        )

        # Analytics over Company/Person nodes.
        query = query.replace(
            "MATCH (n:{entity_type})-[:HAS_EVENT|ACTED_IN]-(e:Event)",
            f"MATCH (n:{{entity_type}})-[:HAS_EVENT|ACTED_IN]-(e:Event)\n\t\tWHERE {self._strong_filter('n')}",
        )
        query = query.replace(
            "MATCH (n:{entity_type})\n\t\t\t{where_clause}",
            f"MATCH (n:{{entity_type}})\n\t\t\t{{where_clause}}\n\t\t\tWITH n WHERE {self._strong_filter('n')}",
        )
        query = query.replace(
            "MATCH (n:{entity_type})\n\t\t\t{where_clause}\n\t\t\tMATCH (n)-[r:HAS_EVENT|ACTED_IN]-(e:Event)",
            f"MATCH (n:{{entity_type}})\n\t\t\t{{where_clause}}\n\t\t\tWITH n WHERE {self._strong_filter('n')}\n\t\t\tMATCH (n)-[r:HAS_EVENT|ACTED_IN]-(e:Event)",
        )
        return query

    def search_companies(self, *args, **kwargs):
        return filter_weak_rows(super().search_companies(*args, **kwargs))

    def global_text_search(self, *args, **kwargs):
        return filter_weak_rows(super().global_text_search(*args, **kwargs))

    def explore_network(self, *args, **kwargs):
        return filter_weak_rows(super().explore_network(*args, **kwargs))

    def get_node_history(self, *args, **kwargs):
        return filter_weak_rows(super().get_node_history(*args, **kwargs))

    def get_top_entities(self, *args, **kwargs):
        return filter_weak_rows(super().get_top_entities(*args, **kwargs))

    def count_entities_by_event(self, *args, **kwargs):
        return filter_weak_rows(super().count_entities_by_event(*args, **kwargs))


def run_structured_only_answers(questions, output_path, max_iterations):
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "shabdb")
    api_key = os.getenv("OPENAI_API_KEY")

    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    investigator = StructuredOnlyInvestigator(
        driver=driver,
        api_key=api_key,
        model=ANSWER_MODEL,
        iterations=max_iterations,
        database=database,
    )
    rows = []

    try:
        with output_path.open("w", encoding="utf-8") as out:
            for q in tqdm.tqdm(questions, desc="Structured-only GraphRAG"):
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
                    "system": "Structured-only / no-weak-node Agentic GraphRAG",
                    "ablation": "no_weak_nodes",
                    "weak_node_detection": "coalesce(n.is_weak,false)=true or uid starts with weak_/weak_comp_",
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
                    # Includes router plus planning/synthesis assistant calls.
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
    with tqdm.tqdm(total=len(tasks), desc="Judge structured-only") as pbar:
        for task in asyncio.as_completed(tasks):
            scored.append(await task)
            pbar.update(1)
    return scored


def summarize_structured_only(rows):
    latencies = [row["latency_seconds"] for row in rows]
    total_tool_calls = sum(row["tool_calls"] for row in rows)
    usage = sum_usage(rows)
    cost = estimate_judge_cost(rows)
    return {
        "system": "Structured-only / no-weak-node Agentic GraphRAG",
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
        fh.write("\\caption{Structured-only/no-weak-node ablation on the 60-question manually curated benchmark.}\n")
        fh.write("\\label{tab:phase2_ablation_no_weak_nodes}\n")
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
    parser.add_argument("--structured-only", action="store_true", default=True)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_human_dataset(str(DATASET_PATH))

    answers_path = RESULTS_DIR / "outputs_ablation_no_weak_nodes_human.jsonl"
    metrics_path = RESULTS_DIR / "results_phase2_ablation_no_weak_nodes.csv"
    table_path = RESULTS_DIR / "table_ablation_no_weak_nodes.tex"

    structured_rows = run_structured_only_answers(
        questions,
        answers_path,
        max_iterations=args.max_iterations,
    )
    structured_rows = await score_rows(structured_rows, args.concurrency)
    write_outputs_jsonl(answers_path, structured_rows)

    summaries = [
        load_full_summary(),
        summarize_structured_only(structured_rows),
    ]
    write_metrics_csv(metrics_path, summaries)
    write_latex(table_path, summaries)

    print(f"Wrote {answers_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    asyncio.run(main())
