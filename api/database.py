"""Shared Neo4j connection and analytical-agent instance for the API."""

import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase, READ_ACCESS

# Ensure local package imports work when running the API directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agenticGraphRAG.investigator import SHABInvestigator

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", ""),
)
DATABASE = os.getenv("NEO4J_DATABASE", "shabdb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Database:
    def __init__(self):
        if not AUTH[1]:
            raise RuntimeError("NEO4J_PASSWORD is required.")
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        if not OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY is not set.")
        # Strip quotes and whitespace that might be accidentally included
        clean_key = OPENAI_API_KEY.strip("\"' ") if OPENAI_API_KEY else None
        self.investigator = SHABInvestigator(self.driver, clean_key, database=DATABASE)

    def close(self):
        self.driver.close()

    def run_query(self, query, params=None):
        with self.driver.session(
            database=DATABASE,
            default_access_mode=READ_ACCESS,
        ) as session:
            result = session.run(query, params or {})
            return [r.data() for r in result]


db = Database()
