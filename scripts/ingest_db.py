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
import tqdm
import sys

# Add repo root to sys.path so agenticGraphRAG can be imported.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticGraphRAG.graph_db import Neo4jIngester

# ─────────────────────────────────────────────
# CONFIGURATION — update these before running
# ─────────────────────────────────────────────
# DATA_FOLDER defaults to a processed_data directory under the repo root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FOLDER = os.getenv("PROCESSED_GRAPH_DIR", os.path.join(PROJECT_ROOT, "processed_data/processed_graph_data_archive"))
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
    os.path.join(PROJECT_ROOT, "processed_data", "anonymization_mapping.json"),
)
# ─────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  SHAB Risk Radar — Database Ingestion Pipeline")
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
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.json")))
    if not files:
        print(f"\n❌ No JSON files found in: {DATA_FOLDER}")
        print("   Make sure DATA_FOLDER points to your processed checkpoint outputs.")
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
