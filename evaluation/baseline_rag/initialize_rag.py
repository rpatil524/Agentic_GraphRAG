"""Initialize (or refresh) the baseline vector DB from Neo4j events."""

import argparse
import os

from vector_rag import NaiveVectorRAG


def main():
    parser = argparse.ArgumentParser(description="Build baseline vector DB from Neo4j.")
    parser.add_argument("--reset", action="store_true", help="Reset the Chroma collection before ingestion.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    baseline = NaiveVectorRAG(api_key=api_key)
    baseline.ingest_from_neo4j(reset=args.reset)


if __name__ == "__main__":
    main()
