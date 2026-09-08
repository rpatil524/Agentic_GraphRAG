#!/usr/bin/env python3
"""Retrieval-only top-k sensitivity for the existing Chroma vector baseline.

This script deliberately avoids OpenAI calls. It evaluates whether the retrieved
documents contain the expected answer string, or enough answer tokens, for a set
of top-k values. It is therefore a retrieval diagnostic, not an answer-quality
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "evaluation" / "results"


DATASETS = {
    "human": ROOT / "evaluation" / "datasets" / "golden_dataset.json",
    "automated": ROOT / "evaluation" / "datasets" / "automated_dataset.json",
}


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9äöüàáâçèéêìíîòóôùúûñß]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_tokens(expected: str) -> set[str]:
    stop = {
        "the", "a", "an", "of", "is", "in", "on", "for", "to", "and",
        "was", "were", "with", "company", "registered", "address", "base",
        "notice", "published", "dissolution", "bankruptcy",
    }
    return {tok for tok in normalize(expected).split() if len(tok) >= 3 and tok not in stop}


def load_dataset(name: str, limit: Optional[int]):
    path = DATASETS[name]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        question = item.get("question") or item.get("question_text")
        expected = item.get("answer") or item.get("expected_answer") or item.get("raw_answer")
        rows.append(
            {
                "question_id": item.get("question_id"),
                "question": question,
                "expected_answer": expected,
            }
        )
    return rows[:limit] if limit else rows


def retrieve(collection, question: str, k: int):
    result = collection.query(query_texts=[question], n_results=k)
    return result.get("documents", [[]])[0] or []


def evaluate(dataset_name: str, db_path: Path, ks: list[int], limit: Optional[int]):
    local_ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("shab_events_local", embedding_function=local_ef)
    questions = load_dataset(dataset_name, limit)

    rows = []
    for k in ks:
        start = time.time()
        exact_hits = 0
        token_recalls = []
        token_hit_50 = 0
        retrieved_nonempty = 0

        for item in questions:
            docs = retrieve(collection, item["question"], k)
            joined = normalize(" ".join(docs))
            expected_norm = normalize(item["expected_answer"])
            tokens = answer_tokens(item["expected_answer"])

            exact_hit = bool(expected_norm and expected_norm in joined)
            token_recall = (
                sum(1 for token in tokens if token in joined) / len(tokens)
                if tokens else 0.0
            )

            exact_hits += int(exact_hit)
            token_hit_50 += int(token_recall >= 0.5)
            token_recalls.append(token_recall)
            retrieved_nonempty += int(bool(docs))

        n = len(questions)
        rows.append(
            {
                "dataset": dataset_name,
                "k": k,
                "n": n,
                "exact_expected_answer_hit_at_k": exact_hits / n if n else 0.0,
                "token_recall_at_k": sum(token_recalls) / n if n else 0.0,
                "token_hit_50_at_k": token_hit_50 / n if n else 0.0,
                "retrieved_nonempty_at_k": retrieved_nonempty / n if n else 0.0,
                "runtime_seconds": time.time() - start,
                "notes": "Retrieval-only diagnostic; no answer generation or LLM judging.",
            }
        )
    return rows


def write_outputs(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "results_phase2_vector_topk_retrieval_only.csv"
    fieldnames = [
        "dataset",
        "k",
        "n",
        "exact_expected_answer_hit_at_k",
        "token_recall_at_k",
        "token_hit_50_at_k",
        "retrieved_nonempty_at_k",
        "runtime_seconds",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tex_path = RESULTS_DIR / "table_vector_topk_retrieval_only.tex"
    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table}[htbp]\n")
        fh.write("\\centering\n")
        fh.write("\\caption{Retrieval-only top-k sensitivity for the vector baseline. No answer generation or LLM judging is used.}\n")
        fh.write("\\label{tab:phase2_vector_topk_retrieval_only}\n")
        fh.write("\\small\n")
        fh.write("\\begin{tabular}{lcccc}\n")
        fh.write("\\toprule\n")
        fh.write("$k$ & Exact Hit & Token Recall & Token Hit@50\\% & Nonempty \\\\\n")
        fh.write("\\midrule\n")
        for row in rows:
            fh.write(
                f"{row['k']} & "
                f"{row['exact_expected_answer_hit_at_k']:.3f} & "
                f"{row['token_recall_at_k']:.3f} & "
                f"{row['token_hit_50_at_k']:.3f} & "
                f"{row['retrieved_nonempty_at_k']:.3f} \\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")
        fh.write("\\end{table}\n")
    return csv_path, tex_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="human")
    parser.add_argument("--db-path", default=str(ROOT / "evaluation" / "baseline_rag" / "chroma_db"))
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = evaluate(args.dataset, Path(args.db_path), args.k, args.limit)
    csv_path, tex_path = write_outputs(rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
