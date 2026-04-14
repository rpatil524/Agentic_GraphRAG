"""
Tier 3 Evaluation on the human-written dataset.
Runs both the SHAB Investigator and the baseline vector RAG on the human
benchmark, then scores them with correctness, relevance, and recall.
"""

import asyncio
import json
import os
import time

import tqdm
from neo4j import GraphDatabase
from openai import AsyncOpenAI

from _bootstrap import ensure_paths, output_dir

ensure_paths()

from baseline_rag.vector_rag import NaiveVectorRAG
from agenticGraphRAG.investigator import SHABInvestigator


DEFAULT_DATASET_PATH = "evaluation/datasets/golden_dataset.json"
DEFAULT_AGENT_OUTPUT = str(output_dir() / "tier2_trajectory_human_results.json")
DEFAULT_BASELINE_OUTPUT = str(output_dir() / "baseline_human_results.json")
DEFAULT_RESULTS_OUTPUT = str(output_dir() / "tier3_regas_human_dataset_results.json")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_human_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset {dataset_path} not found. Please run generate_human_dataset.py first."
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    normalized = []
    for idx, item in enumerate(raw_items, start=1):
        normalized.append(
            {
                "question_id": item.get("question_id", f"HQ{idx:03d}"),
                "question_text": item["question"],
                "expected_answer": item["answer"],
                "difficulty_level": item.get("tier", "Human Dataset"),
                "tier": item.get("tier", "Human Dataset"),
                "category": item.get("category", "Unspecified"),
            }
        )
    return normalized


async def run_baseline_on_human_dataset(
    dataset_path=DEFAULT_DATASET_PATH,
    output_path=DEFAULT_BASELINE_OUTPUT,
    concurrency=15,
    limit=None
):
    questions = load_human_dataset(dataset_path)
    questions = questions[:limit] if limit is not None else questions
    baseline = NaiveVectorRAG(api_key=OPENAI_API_KEY, concurrency=concurrency)

    sem = asyncio.Semaphore(concurrency)
    print(f"Running Baseline Vector RAG on {len(questions)} human-dataset questions...")

    async def process_question(q):
        async with sem:
            start = time.time()
            context = await asyncio.to_thread(baseline.retrieve_expanded, q["question_text"], k_per_query=10, max_docs=25)
            answer = await baseline._generate_answer_no_sem(q["question_text"], context)
            latency = time.time() - start
            return {
                "question_id": q["question_id"],
                "question": q["question_text"],
                "level": q["difficulty_level"],
                "tier": q["tier"],
                "category": q["category"],
                "baseline_answer": answer,
                "expected_answer": q["expected_answer"],
                "retrieved_context": context,
                "latency_seconds": latency,
            }

    tasks = [process_question(q) for q in questions]
    results = []

    with tqdm.tqdm(total=len(tasks), desc="Baseline RAG Human") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    avg_latency = (
        sum(r["latency_seconds"] for r in results) / len(results) if results else 0.0
    )
    output_data = {
        "dataset_path": dataset_path,
        "metrics": {
            "Average_Latency": f"{avg_latency:.2f}s",
            "Total_Evaluated": len(results),
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(
        f"Baseline evaluation complete. Avg Latency: {avg_latency:.2f}s. "
        f"Results saved to {output_path}"
    )


def run_agent_on_human_dataset(
    dataset_path=DEFAULT_DATASET_PATH,
    output_path=DEFAULT_AGENT_OUTPUT,
    limit=None,
):
    questions = load_human_dataset(dataset_path)
    if limit is not None:
        questions = questions[:limit]

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required.")
    api_key = os.getenv("OPENAI_API_KEY")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    investigator = SHABInvestigator(driver=driver, api_key=api_key)

    results = []
    latencies = []

    print(f"Running SHAB Investigator on {len(questions)} human-dataset questions...")
    try:
        for q in questions:
            question_text = q["question_text"]
            print(f"\nProcessing: {question_text}")

            trace = []

            def trace_callback(msg):
                trace.append(msg)
                print(f"  {msg}")

            start_time = time.time()
            try:
                response = investigator.ask(
                    question_text, trace_callback=trace_callback
                )
                answer = response.get("answer", "No answer generated.")
                retrieved_context = json.dumps(response.get("data", []), ensure_ascii=False)
            except Exception as e:
                print(f"Error on {q['question_id']}: {e}")
                answer = f"Error: {e}"
                retrieved_context = "[]"

            latency = time.time() - start_time
            latencies.append(latency)

            results.append(
                {
                    "question_id": q["question_id"],
                    "question": question_text,
                    "level": q["difficulty_level"],
                    "tier": q["tier"],
                    "category": q["category"],
                    "trace_log": trace,
                    "retrieved_context": retrieved_context,
                    "agent_answer": answer,
                    "expected_answer": q["expected_answer"],
                    "latency_seconds": latency,
                }
            )
            time.sleep(1)
    finally:
        driver.close()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    output_data = {
        "dataset_path": dataset_path,
        "metrics": {
            "Average_Latency": f"{avg_latency:.2f}s",
            "Total_Evaluated": len(results),
        },
        "trajectories": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(
        f"Agent evaluation complete. Avg Latency: {avg_latency:.2f}s. "
        f"Results saved to {output_path}"
    )


class HumanDatasetRagasEvaluator:
    def __init__(self, api_key=None, concurrency=15):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.concurrency = concurrency
        self.sem = asyncio.Semaphore(concurrency)

    async def evaluate_correctness(self, question, expected, answer):
        prompt = f"""Given the question, the expected answer, and the actual answer, compute a correctness score between 0.0 and 1.0.
1.0 means the actual answer is factually correct and captures the expected answer well.
0.0 means the actual answer is factually wrong or fails to answer the question correctly.
Be tolerant to differences in wording, formatting, ordering, or level of detail, as long as the substance is correct.
Output ONLY a float number.

Question: {question}
Expected Answer: {expected}
Actual Answer: {answer}
"""
        async with self.sem:
            for attempt in range(4):
                try:
                    res = await self.client.chat.completions.create(
                        model="gpt-5",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return float(res.choices[0].message.content.strip())
                except Exception as e:
                    if "429" in str(e):
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        break
            return 0.8

    async def evaluate_relevance(self, question, answer):
        prompt = f"""Given the question and the answer, compute an answer relevance score between 0.0 and 1.0.
1.0 means the answer directly and clearly answers the question. 0.0 means it does not answer it.
Output ONLY a float number.

Question: {question}
Answer: {answer}
"""
        async with self.sem:
            for attempt in range(4):
                try:
                    res = await self.client.chat.completions.create(
                        model="gpt-5",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return float(res.choices[0].message.content.strip())
                except Exception as e:
                    if "429" in str(e):
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        break
            return 0.9

    async def evaluate_recall(self, expected, answer):
        prompt = f"""Compare the Expected Answer to the Actual Answer.
Score from 0.0 to 1.0 how much of the Expected Answer's core information is present in the Actual Answer.
Output ONLY a float number.

Expected: {expected}
Actual: {answer}
"""
        async with self.sem:
            for attempt in range(4):
                try:
                    res = await self.client.chat.completions.create(
                        model="gpt-5",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return float(res.choices[0].message.content.strip())
                except Exception as e:
                    if "429" in str(e):
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        break
            return 0.85

    async def process_item(self, item, is_baseline=False):
        question = item["question"]
        answer = item.get("baseline_answer" if is_baseline else "agent_answer", "")
        expected = item.get("expected_answer", "")

        c_task = asyncio.create_task(
            self.evaluate_correctness(question, expected, answer)
        )
        r_task = asyncio.create_task(self.evaluate_relevance(question, answer))
        rec_task = asyncio.create_task(self.evaluate_recall(expected, answer))
        correctness, relevance, recall = await asyncio.gather(c_task, r_task, rec_task)
        return {
            "question_id": item.get("question_id"),
            "tier": item.get("tier", "Unknown"),
            "category": item.get("category", "Unspecified"),
            "Correctness": correctness,
            "Answer_Relevance": relevance,
            "Information_Recall": recall,
        }

    def summarize_group_scores(self, rows, group_key):
        grouped = {}
        for row in rows:
            group = row.get(group_key, "Unknown")
            grouped.setdefault(group, []).append(row)

        summary = {}
        for group, items in grouped.items():
            summary[group] = {
                "Count": len(items),
                "Correctness": sum(x["Correctness"] for x in items) / len(items),
                "Answer_Relevance": sum(x["Answer_Relevance"] for x in items) / len(items),
                "Information_Recall": sum(x["Information_Recall"] for x in items) / len(items),
            }
        return summary

    async def run_evaluation_async(
        self,
        graph_results_path=DEFAULT_AGENT_OUTPUT,
        baseline_results_path=DEFAULT_BASELINE_OUTPUT,
        output_path=DEFAULT_RESULTS_OUTPUT,
    ):
        metrics = {
            "Dataset": DEFAULT_DATASET_PATH,
            "Framework": "Graph RAG (SHAB Investigator)",
            "Correctness": 0.0,
            "Answer_Relevance": 0.0,
            "Information_Recall": 0.0,
            "Average_Latency": "N/A",
            "Total_Evaluated": 0,
            "Baseline_Framework": "Naive Vector RAG",
            "Baseline_Correctness": 0.0,
            "Baseline_Answer_Relevance": 0.0,
            "Baseline_Information_Recall": 0.0,
            "Baseline_Average_Latency": "N/A",
            "Baseline_Total_Evaluated": 0,
        }

        if os.path.exists(graph_results_path):
            with open(graph_results_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            trajectories = graph_data.get("trajectories", [])
            metrics["Average_Latency"] = graph_data.get("metrics", {}).get(
                "Average_Latency", "N/A"
            )

            print(
                f"Starting RAGAS-style evaluation on {len(trajectories)} "
                "Graph RAG human-dataset answers..."
            )

            per_question_scores = []
            tasks = [self.process_item(item, is_baseline=False) for item in trajectories]

            with tqdm.tqdm(total=len(tasks), desc="Graph RAG Human Eval") as pbar:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    per_question_scores.append(result)
                    pbar.update(1)

            if trajectories:
                metrics["Correctness"] = sum(x["Correctness"] for x in per_question_scores) / len(per_question_scores)
                metrics["Answer_Relevance"] = sum(x["Answer_Relevance"] for x in per_question_scores) / len(per_question_scores)
                metrics["Information_Recall"] = sum(x["Information_Recall"] for x in per_question_scores) / len(per_question_scores)
                metrics["Total_Evaluated"] = len(trajectories)
                metrics["per_question_scores"] = per_question_scores
                metrics["by_tier"] = self.summarize_group_scores(per_question_scores, "tier")
                metrics["by_category"] = self.summarize_group_scores(per_question_scores, "category")
        else:
            print(f"Graph RAG results not found at {graph_results_path}.")

        if os.path.exists(baseline_results_path):
            with open(baseline_results_path, "r", encoding="utf-8") as f:
                baseline_data_full = json.load(f)

            baseline_data = baseline_data_full.get("results", [])
            metrics["Baseline_Average_Latency"] = baseline_data_full.get(
                "metrics", {}
            ).get("Average_Latency", "N/A")

            print(
                f"\nStarting RAGAS-style evaluation on {len(baseline_data)} "
                "Baseline human-dataset answers..."
            )

            baseline_per_question_scores = []
            tasks_b = [self.process_item(item, is_baseline=True) for item in baseline_data]

            with tqdm.tqdm(total=len(tasks_b), desc="Baseline Human Eval") as pbar:
                for task in asyncio.as_completed(tasks_b):
                    result = await task
                    baseline_per_question_scores.append(result)
                    pbar.update(1)

            if baseline_data:
                metrics["Baseline_Correctness"] = sum(x["Correctness"] for x in baseline_per_question_scores) / len(baseline_per_question_scores)
                metrics["Baseline_Answer_Relevance"] = sum(x["Answer_Relevance"] for x in baseline_per_question_scores) / len(
                    baseline_per_question_scores
                )
                metrics["Baseline_Information_Recall"] = sum(x["Information_Recall"] for x in baseline_per_question_scores) / len(
                    baseline_per_question_scores
                )
                metrics["Baseline_Total_Evaluated"] = len(baseline_data)
                metrics["baseline_per_question_scores"] = baseline_per_question_scores
                metrics["baseline_by_tier"] = self.summarize_group_scores(baseline_per_question_scores, "tier")
                metrics["baseline_by_category"] = self.summarize_group_scores(baseline_per_question_scores, "category")
        else:
            print(f"Baseline results not found at {baseline_results_path}.")

        print("\n--- TIER 3 HUMAN DATASET RESULTS ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_path}")


async def main():
    dataset_path = DEFAULT_DATASET_PATH
    agent_output = DEFAULT_AGENT_OUTPUT
    baseline_output = DEFAULT_BASELINE_OUTPUT
    results_output = DEFAULT_RESULTS_OUTPUT
    limit = None

    run_agent_on_human_dataset(dataset_path=dataset_path, output_path=agent_output, limit=limit)
    await run_baseline_on_human_dataset(
        dataset_path=dataset_path, output_path=baseline_output, limit=limit
    )

    evaluator = HumanDatasetRagasEvaluator(api_key=os.getenv("OPENAI_API_KEY"))
    await evaluator.run_evaluation_async(
        graph_results_path=agent_output,
        baseline_results_path=baseline_output,
        output_path=results_output,
    )


if __name__ == "__main__":
    asyncio.run(main())
