"""
run_job_parallel.py
===================
Process SHAB CSV batches into graph checkpoint JSON files in parallel.

This script runs the full processing pipeline:
1. Load and filter raw CSV data.
2. Run structured ingestion (strong nodes).
3. Run structured edge wiring.
4. Identify high-value events for LLM enrichment.
5. Run unstructured extraction (weak nodes) using OPENAI_API_KEY.
6. Save one checkpoint JSON per input folder.

Environment variables:
    INPUT_GLOB         Glob pattern for input folders containing monthly CSV files.
                       Default: shab_data/*
    OUTPUT_DIR         Output directory for checkpoint JSON files.
                       Default: processed_data/processed_graph_data_archive
    MAX_WORKERS        Parallel workers for folder-level processing. Default: 4
    ROW_LIMIT          Optional row cap per CSV for testing. Default: empty (no limit)
    OPENAI_API_KEY     Required for Phase 2 LLM enrichment.
"""

import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agenticGraphRAG import SHABPipeline


INPUT_GLOB = os.getenv("INPUT_GLOB", str(REPO_ROOT / "shab_data" / "*"))
OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(REPO_ROOT / "processed_data" / "processed_graph_data_archive"),
    )
).expanduser()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
ROW_LIMIT_RAW = os.getenv("ROW_LIMIT", "").strip()
ROW_LIMIT = int(ROW_LIMIT_RAW) if ROW_LIMIT_RAW else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_folder(folder_path):
    """
    Process one folder containing SHAB CSV data and output one graph checkpoint file.
    """
    folder_name = os.path.basename(folder_path)
    output_file = OUTPUT_DIR / f"graph_{folder_name}.json"

    try:
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        if not csv_files:
            return f"⚠️ No CSV in {folder_name}"
        csv_path = csv_files[0]

        df = SHABPipeline.load_skeleton_data(csv_path, limit=ROW_LIMIT)

        if df.empty:
            return f"⚠️ Empty {folder_name}"

        pipeline = SHABPipeline(api_key=OPENAI_API_KEY)
        pipeline.run_phase_1_structured_ingestion(df)
        pipeline.run_structured_edge_creation()

        llm_candidates = pipeline.identify_candidates()

        enrichment_count = 0
        if llm_candidates and OPENAI_API_KEY:
            enrichment_count = len(llm_candidates)
            pipeline.run_phase_2_unstructured_ingestion(
                batch_size=20,
                subset_events=llm_candidates,
            )
        elif llm_candidates and not OPENAI_API_KEY:
            return f"❌ {folder_name}: OPENAI_API_KEY missing (cannot run Phase 2 LLM extraction)."

        stats = pipeline.save_checkpoint(output_file)

        return (
            f"✅ {folder_name} | Rows: {len(df)} | LLM: {enrichment_count} | "
            f"Nodes: {stats['companies']} Comp, {stats['people']} Ppl | "
            f"Edges: {stats['edges']}"
        )

    except Exception as exc:
        return f"❌ Error {folder_name}: {exc}"


def main():
    all_folders = sorted(glob.glob(INPUT_GLOB))
    pending_folders = [
        f
        for f in all_folders
        if not (OUTPUT_DIR / f"graph_{os.path.basename(f)}.json").exists()
    ]

    print(
        f"🚀 STARTING PARALLEL RUN on {len(pending_folders)} folders "
        f"({len(all_folders) - len(pending_folders)} already processed)"
    )
    print(f"   - Workers: {MAX_WORKERS}")
    print(f"   - Input glob: {INPUT_GLOB}")
    print(f"   - Output dir: {OUTPUT_DIR}")
    print(f"   - Row limit: {ROW_LIMIT if ROW_LIMIT is not None else 'None'}")

    if not all_folders:
        raise FileNotFoundError(
            f"No input folders matched INPUT_GLOB={INPUT_GLOB!r}. "
            "Run the downloader or set INPUT_GLOB to the monthly SHAB folders."
        )
    if pending_folders and not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is required for Phase 2 weak-node extraction."
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_folder = {
            executor.submit(process_folder, folder): folder
            for folder in pending_folders
        }

        for future in tqdm.tqdm(
            as_completed(future_to_folder),
            total=len(pending_folders),
            desc="Processing",
        ):
            try:
                result_msg = future.result()
                tqdm.tqdm.write(result_msg)
            except Exception as exc:
                tqdm.tqdm.write(f"🔥 CRITICAL THREAD ERROR: {exc}")

    print("🏁 PARALLEL RUN COMPLETE.")


if __name__ == "__main__":
    main()
