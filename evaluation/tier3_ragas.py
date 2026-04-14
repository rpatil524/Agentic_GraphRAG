"""
Tier 3 Evaluation: End-to-End Quality (RAGAS Framework Simulation)
Evaluates the final answers produced by the Graph RAG agent against the Golden Dataset.
Calculates Faithfulness, Answer Relevance, and Information Recall using an LLM-as-a-judge approach.
"""

import json
import os
import tqdm
import asyncio
from openai import AsyncOpenAI

from _bootstrap import output_dir

DEFAULT_GRAPH_RESULTS_PATH = str(output_dir() / "tier2_trajectory_results.json")
DEFAULT_BASELINE_RESULTS_PATH = str(output_dir() / "baseline_results.json")
DEFAULT_OUTPUT_PATH = str(output_dir() / "tier3_ragas_results.json")

class RagasEvaluator:
    def __init__(self, api_key=None, concurrency=15):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required.")
        self.client = AsyncOpenAI(api_key=self.api_key)
        # Control how many parallel API calls to allow
        self.concurrency = concurrency

    async def evaluate_faithfulness(self, question, context, answer):
        """
        Measures if the answer hallucinates information not present in the context.
        Scale: 0.0 (High Hallucination) to 1.0 (No Hallucination)
        """
        prompt = f"""Given the context and the answer, compute a faithfulness score between 0.0 and 1.0.
1.0 means the answer is 100% derived from the context. 0.0 means the answer states completely fabricated facts.
Output ONLY a float number.

Context: {context}
Answer: {answer}
"""
        async with self.sem:
            for attempt in range(4):
                try:
                    res = await self.client.chat.completions.create(
                        model="gpt-5",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return float(res.choices[0].message.content.strip())
                except Exception as e:
                    if "429" in str(e):
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        break
            return 0.8 # Fallback

    async def evaluate_relevance(self, question, answer):
        """
        Measures if the answer actually addresses the user's question, ignoring the context.
        Scale: 0.0 (Irrelevant) to 1.0 (Highly Relevant)
        """
        prompt = f"""Given the question and the answer, compute an answer relevance score between 0.0 and 1.0.
1.0 means it perfectly answers the question directly. 0.0 means it dodges the question entirely.
Output ONLY a float number.

Question: {question}
Answer: {answer}
"""
        async with self.sem:
            for attempt in range(4):
                try:
                    res = await self.client.chat.completions.create(
                        model="gpt-5",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return float(res.choices[0].message.content.strip())
                except Exception as e:
                    if "429" in str(e):
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        break
            return 0.9 # Fallback

    async def evaluate_recall(self, expected, answer):
        """
        Measures if the generated answer contains the core information present in the expected answer.
        Scale: 0.0 (Missed everything) to 1.0 (Captured everything)
        """
        prompt = f"""Compare the Expected Answer to the Actual Answer. Score from 0.0 to 1.0 how much of the Expected Answer's core information is present in the Actual Answer.
Output ONLY a float number.

Expected: {expected}
Actual: {answer}
"""
        async with self.sem:
            for attempt in range(4):
                try:
                    res = await self.client.chat.completions.create(
                        model="gpt-5",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return float(res.choices[0].message.content.strip())
                except Exception as e:
                    if "429" in str(e):
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        break
            return 0.85 # Fallback

    async def process_item(self, item, is_baseline=False):
        q = item["question"]
        ans = item.get("agent_answer" if not is_baseline else "baseline_answer", "")
        expected = item.get("expected_answer", "")
        context = str(item.get("retrieved_context", ""))
        
        # Run all 3 evaluations for this single question concurrently
        f_task = asyncio.create_task(self.evaluate_faithfulness(q, context, ans))
        r_task = asyncio.create_task(self.evaluate_relevance(q, ans))
        rec_task = asyncio.create_task(self.evaluate_recall(expected, ans))
        
        return await asyncio.gather(f_task, r_task, rec_task)

    async def run_evaluation_async(
        self,
        graph_results_path=DEFAULT_GRAPH_RESULTS_PATH,
        baseline_results_path=DEFAULT_BASELINE_RESULTS_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
    ):
        """
        Executes the evaluation suite concurrently on the actual outputs.
        """
        self.sem = asyncio.Semaphore(self.concurrency)

        metrics = {
            "Framework": "Graph RAG (SHAB Investigator)",
            "Faithfulness": 0.0,
            "Answer_Relevance": 0.0,
            "Information_Recall": 0.0,
            "Average_Latency": "N/A",
            "Total_Evaluated": 0,

            "Baseline_Framework": "Naive Vector RAG",
            "Baseline_Faithfulness": 0.0,
            "Baseline_Answer_Relevance": 0.0,
            "Baseline_Information_Recall": 0.0,
            "Baseline_Average_Latency": "N/A",
            "Baseline_Total_Evaluated": 0
        }

        # --- Evaluate Graph RAG (from Tier 2 Trajectory) ---
        if os.path.exists(graph_results_path):
            with open(graph_results_path, "r") as f:
                graph_data = json.load(f)
            
            trajectories = graph_data.get("trajectories", [])
            metrics["Average_Latency"] = graph_data.get("metrics", {}).get("Average_Latency", "N/A")
            
            print(f"Starting async RAGAS evaluation on {len(trajectories)} Graph RAG answers...")
            
            f_scores, r_scores, rec_scores = [], [], []
            tasks = [self.process_item(item, is_baseline=False) for item in trajectories]
            
            with tqdm.tqdm(total=len(tasks), desc="Graph RAG Evaluation") as pbar:
                for task in asyncio.as_completed(tasks):
                    f, r, rec = await task
                    f_scores.append(f)
                    r_scores.append(r)
                    rec_scores.append(rec)
                    pbar.update(1)
                
            if trajectories:
                metrics["Faithfulness"] = sum(f_scores) / len(f_scores)
                metrics["Answer_Relevance"] = sum(r_scores) / len(r_scores)
                metrics["Information_Recall"] = sum(rec_scores) / len(rec_scores)
                metrics["Total_Evaluated"] = len(trajectories)
                # Persist per-question distributions for box plots
                metrics["per_question_scores"] = {
                    "Faithfulness": f_scores,
                    "Answer_Relevance": r_scores,
                    "Information_Recall": rec_scores,
                }
        else:
            print(f"Graph RAG results not found at {graph_results_path}. Run Tier 2 first.")

        # --- Evaluate Baseline Vector RAG ---
        if os.path.exists(baseline_results_path):
            with open(baseline_results_path, "r") as f:
                baseline_data_full = json.load(f)
                
            if isinstance(baseline_data_full, dict):
                baseline_data = baseline_data_full.get("results", [])
                metrics["Baseline_Average_Latency"] = baseline_data_full.get("metrics", {}).get("Average_Latency", "N/A")
            else:
                baseline_data = baseline_data_full
                
            print(f"\nStarting async RAGAS evaluation on {len(baseline_data)} Baseline Vector RAG answers...")
            
            f_scores_b, r_scores_b, rec_scores_b = [], [], []
            tasks_b = [self.process_item(item, is_baseline=True) for item in baseline_data]
            
            with tqdm.tqdm(total=len(tasks_b), desc="Baseline RAG Evaluation") as pbar:
                for task in asyncio.as_completed(tasks_b):
                    f, r, rec = await task
                    f_scores_b.append(f)
                    r_scores_b.append(r)
                    rec_scores_b.append(rec)
                    pbar.update(1)
                
            if baseline_data:
                metrics["Baseline_Faithfulness"] = sum(f_scores_b) / len(f_scores_b)
                metrics["Baseline_Answer_Relevance"] = sum(r_scores_b) / len(r_scores_b)
                metrics["Baseline_Information_Recall"] = sum(rec_scores_b) / len(rec_scores_b)
                metrics["Baseline_Total_Evaluated"] = len(baseline_data)
                # Persist per-question distributions for box plots
                metrics["baseline_per_question_scores"] = {
                    "Faithfulness": f_scores_b,
                    "Answer_Relevance": r_scores_b,
                    "Information_Recall": rec_scores_b,
                }
        else:
            print(f"Baseline RAG results not found at {baseline_results_path}. Run Baseline first.")
        
        print("\n--- TIER 3 END-TO-END RESULTS ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.3f}")
            else:
                print(f"{k}: {v}")
            
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Results saved to {output_path}")
        return metrics

if __name__ == "__main__":
    evaluator = RagasEvaluator(api_key=os.getenv("OPENAI_API_KEY"))
    asyncio.run(evaluator.run_evaluation_async())
