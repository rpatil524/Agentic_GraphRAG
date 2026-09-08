"""
ingest_db.py
============
Populates the Neo4j database from processed SHAB graph JSON checkpoint files.

Usage:
    python ingest_db.py

Steps:
    1. Creates constraints and indexes.
    2. Ingests all *.json files from DATA_FOLDER into Neo4j.
    3. Deduplicates Name Hubs (merges token-order variants, e.g. "Kauter Martin" == "Martin Kauter").
    4. Resolves weak nodes (removes LLM-extracted duplicates that exist in the official register).
    5. Runs graph analytics (PageRank risk scores + community detection).
"""

import glob
import os
import sys
from pathlib import Path

import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agenticGraphRAG.graph_db import Neo4jIngester

DATA_FOLDER = Path(
    os.getenv(
        "PROCESSED_GRAPH_DIR",
        str(REPO_ROOT / "processed_data" / "processed_graph_data_archive"),
    )
).expanduser()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", ""),
)
DATABASE = os.getenv("NEO4J_DATABASE", "shabdb")
ANONYMIZE_DATA = os.getenv("ANONYMIZE_DATA", "false").lower() in {"1", "true", "yes", "y"}
ANONYMIZATION_SECRET_KEY = os.getenv("ANONYMIZATION_SECRET_KEY", "")
ANONYMIZATION_OUTPUT = os.getenv(
    "ANONYMIZATION_OUTPUT",
    str(REPO_ROOT / "processed_data" / "anonymization_mapping.json"),
)
SKIP_ANALYTICS = os.getenv("SKIP_ANALYTICS", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def main():
    print("=" * 60)
    print("  Agentic GraphRAG — Database Ingestion Pipeline")
    print("=" * 60)

    if not AUTH[1]:
        print("❌ Missing NEO4J_PASSWORD. Export it and retry.")
        return

    ingester = Neo4jIngester(URI, AUTH, database=DATABASE)

    # ── STEP 1: Schema Setup ──────────────────────────────────
    print("\n📐 STEP 1: Creating constraints and indexes...")
    ingester.create_constraints()
    print("   ✅ Schema ready.")

    # ── STEP 2: Find Files ────────────────────────────────────
    files = sorted(glob.glob(str(DATA_FOLDER / "*.json")))
    if not files:
        print(f"\n❌ No JSON files found in: {DATA_FOLDER}")
        print("   Check that PROCESSED_GRAPH_DIR points to the processed checkpoints.")
        ingester.close()
        return

    print(f"\n📂 STEP 2: Found {len(files)} checkpoint file(s) to ingest.")

    # ── STEP 3: Ingest ────────────────────────────────────────
    print("\n⚡ STEP 3: Ingesting graph data into Neo4j...")
    for f in tqdm.tqdm(files, desc="Ingesting files"):
        ingester.ingest_file(f)
    print("   ✅ All files ingested.")

    # ── STEP 4: Hub Deduplication ─────────────────────────────
    print("\n🔤 STEP 4: Deduplicating Name Hubs...")
    print("   (Merges token-order variants, e.g. 'Kauter Martin' ↔ 'Martin Kauter')")
    ingester.deduplicate_name_hubs()

    # ── STEP 5: Weak Node Resolution ─────────────────────────
    print("\n🧹 STEP 5: Resolving weak nodes...")
    print("   (Removes LLM-extracted entities that are duplicates of official register entries)")
    ingester.resolve_weak_nodes()

    # ── STEP 5.5: Privacy Abstraction Layer (Optional) ───────
    if ANONYMIZE_DATA:
        if not ANONYMIZATION_SECRET_KEY:
            print("❌ ANONYMIZE_DATA is enabled but ANONYMIZATION_SECRET_KEY is missing.")
            ingester.close()
            return
        print("\n🔒 STEP 5.5: Applying Privacy Abstraction Layer...")
        ingester.anonymize_graph(
            secret_key=ANONYMIZATION_SECRET_KEY,
            output_path=ANONYMIZATION_OUTPUT,
        )

    # ── STEP 6: Graph Analytics ───────────────────────────────
    if SKIP_ANALYTICS:
        print("\nSTEP 6: Skipping graph analytics (SKIP_ANALYTICS=true).")
    else:
        print("\n📊 STEP 6: Running graph analytics...")
        print("   - PageRank → risk_rank (identifies high-centrality entities)")
        print("   - Louvain  → community_id (detects corporate clusters)")
        ingester.run_analytics(algorithms=["risk", "communities"])

    ingester.close()

    print("\n" + "=" * 60)
    print("  🎉 Ingestion Pipeline Complete!")
    print("  Open the app at http://localhost:3000 to explore the graph.")
    print("=" * 60)


if __name__ == "__main__":
    main()
