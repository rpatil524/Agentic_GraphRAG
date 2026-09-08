"""
Tier 2 Evaluation: Agent State Machine Trajectory Tracking.

Runs the SHAB investigator on a dataset and records:
- final answers
- retrieved context
- execution traces
- simple trajectory metrics
"""

import json
import os
import time
from neo4j import GraphDatabase

from _bootstrap import REPO_ROOT, ensure_paths, output_dir

ensure_paths()

from agenticGraphRAG.investigator import SHABInvestigator

def track_trajectory(trace_log):
    """
    Parses the execution trace to score trajectory adherence.
    """
    trajectory = {
        "called_search_first": False,
        "used_fallback": False,
        "total_steps": 0,
        "success": False
    }
    
    for i, step in enumerate(trace_log):
        if "🛠️ Called:" in step:
            trajectory["total_steps"] += 1
            # e.g., "🛠️ Called: search_companies | Args: {...}"
            tool_name = step.split("Called: ")[1].split(" |")[0].strip()
            
            if trajectory["total_steps"] == 1 and tool_name in ["search_companies", "get_node_history", "explore_network"]:
                trajectory["called_search_first"] = True
            if tool_name == "global_text_search":
                trajectory["used_fallback"] = True
                
        if "✅ Result: Found" in step or "Result: 1" in step or "Got" in step:
            trajectory["success"] = True
            
    return trajectory

DEFAULT_DATASET_PATH = os.getenv(
    "AUTOMATED_DATASET_PATH",
    str(REPO_ROOT / "evaluation" / "datasets" / "automated_dataset.json"),
)
DEFAULT_OUTPUT_PATH = str(output_dir() / "tier2_trajectory_results.json")


def evaluate_tier_2(dataset_path=DEFAULT_DATASET_PATH, limit=None, output_path=DEFAULT_OUTPUT_PATH):
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found! Please run generate_dataset.py first.")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    
    # Initialize Neo4j Driver
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required.")
    database = os.getenv("NEO4J_DATABASE", "shabdb")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for agent execution.")
    
    investigator = SHABInvestigator(driver=driver, api_key=api_key, database=database)
    results = []
    latencies = []
    
    # We evaluate sequentially since SHABInvestigator maintains context per-instance if not careful.
    # But ask() handles statelessness well.
    selected_questions = questions[:limit] if limit is not None else questions
    if not selected_questions:
        driver.close()
        raise ValueError("The selected Tier 2 benchmark contains no questions.")

    print(f"Starting Tier 2 Trajectory Evaluation on {len(selected_questions)} real questions...")
    try:
        for q in selected_questions:
            question_text = q["question_text"]
            print(f"\nProcessing: {question_text}")

            trace = []

            def trace_callback(msg):
                trace.append(msg)
                print(f"  {msg}")

            start_time = time.time()
            try:
                response = investigator.ask(question_text, trace_callback=trace_callback)
                answer = response.get("answer", "No answer generated.")
                retrieved_context = json.dumps(response.get("data", []))
            except Exception as exc:
                print(f"Error on {q['question_id']}: {exc}")
                answer = f"Error: {exc}"
                retrieved_context = "[]"
            latency = time.time() - start_time
            latencies.append(latency)

            results.append({
                "question_id": q["question_id"],
                "question": question_text,
                "level": q["difficulty_level"],
                "trajectory": track_trajectory(trace),
                "trace_log": trace,
                "retrieved_context": retrieved_context,
                "agent_answer": answer,
                "expected_answer": q["expected_answer"],
                "latency_seconds": latency,
            })
            time.sleep(1)  # Avoid bursts against the OpenAI API.
    finally:
        driver.close()

    # Aggregate Metrics
    success_rate = sum(1 for r in results if r["trajectory"]["success"]) / len(results)
    called_search_first_rate = sum(1 for r in results if r["trajectory"]["called_search_first"]) / len(results)
    fallback_rate = sum(1 for r in results if r["trajectory"]["used_fallback"]) / len(results)
    avg_steps = sum(r["trajectory"]["total_steps"] for r in results) / len(results)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    metrics = {
        "Dataset_Size": len(results),
        "Tool_Selection_Accuracy (Search First)": f"{called_search_first_rate*100:.1f}%",
        "Fallback_Activation_Rate": f"{fallback_rate*100:.1f}%",
        "Average_Reasoning_Steps": f"{avg_steps:.1f}",
        "Query_Success_Rate": f"{success_rate*100:.1f}%",
        "Average_Latency": f"{avg_latency:.2f}s"
    }
    
    print("\n--- TIER 2 TRAJECTORY RESULTS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "trajectories": results}, f, indent=4)
        
    print(f"\nResults saved to {output_path}")
    return {"metrics": metrics, "trajectories": results}

if __name__ == "__main__":
    evaluate_tier_2()
