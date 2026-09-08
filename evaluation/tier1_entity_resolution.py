"""
Tier 1: Orthographic Consistency Evaluation
Evaluates whether names grouped by the alphabetical tokenization logic
(`generate_hub_key`) satisfy a conservative orthographic criterion.
"""

import os
import json
from neo4j import GraphDatabase, READ_ACCESS
import Levenshtein

from _bootstrap import output_dir

SIMILARITY_THRESHOLD = 0.7


class EntityResolutionEvaluator:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        if not password:
            raise RuntimeError("NEO4J_PASSWORD is required.")
        self.database = os.getenv("NEO4J_DATABASE", "shabdb")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def generate_sample(self, limit=1000):
        """
        Extracts NameHubs that group multiple distinct entities.
        We check if the grouped entity names are actually lexically similar.
        """
        query = """
        MATCH (h:NameHub)<-[:HAS_NAME]-(n:BaseNode)
        WITH h, collect(DISTINCT n.name) AS connected_names, count(DISTINCT n) AS degree
        WHERE degree >= 2 AND degree <= 20
        RETURN h.uid AS hub_uid, h.name AS hub_name, connected_names, degree
        LIMIT $limit
        """
        
        sample_data = []
        try:
            with self.driver.session(
                database=self.database,
                default_access_mode=READ_ACCESS,
            ) as session:
                result = session.run(query, limit=limit)
                for record in result:
                    sample_data.append({
                        "hub_uid": record["hub_uid"],
                        "hub_name": record["hub_name"],
                        "merged_names": record["connected_names"],
                        "connection_count": record["degree"]
                    })
                    
            print(f"Extracted {len(sample_data)} NameHub samples from the database.")
        except Exception as exc:
            print(f"Failed to pull from Neo4j: {exc}")
            return []

        return sample_data

    def evaluate_orthographic_consistency(self, sample_data):
        """
        Calculates the share of grouped names that satisfy the orthographic rule.
        """
        if not sample_data:
            print("No data to evaluate.")
            return
            
        acceptable_comparisons = 0
        total_pairs_evaluated = 0
        evaluated_pairs = []
        
        # We evaluate the pairwise similarity of all names merged under a single Hub
        for item in sample_data:
            names = [str(name).lower() for name in item["merged_names"]]
            hub_name = str(item["hub_name"]).lower()
            
            for name in names:
                total_pairs_evaluated += 1
                
                # Compare each observed name with the canonical NameHub form.
                similarity = Levenshtein.ratio(name, hub_name)
                
                # If similarity is high enough, or if sorting the words makes them identical
                tokens_name = sorted(name.replace(",", "").split())
                tokens_hub = sorted(hub_name.replace(",", "").split())
                
                is_acceptable = similarity > SIMILARITY_THRESHOLD or tokens_name == tokens_hub
                if is_acceptable:
                    acceptable_comparisons += 1
                    
                evaluated_pairs.append({
                    "hub_name": hub_name,
                    "merged_name": name,
                    "similarity_score": similarity,
                    "meets_orthographic_criterion": is_acceptable
                })

        consistency_rate = (
            acceptable_comparisons / total_pairs_evaluated
        ) * 100 if total_pairs_evaluated > 0 else 0
        
        report = {
            "Total_Hubs_Sampled": len(sample_data),
            "Total_Name_Pairs_Evaluated": total_pairs_evaluated,
            "Orthographic_Consistency_Rate": f"{consistency_rate:.2f}%"
        }
        
        out_path = output_dir() / "tier1_sample_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": report,
                "detailed_evaluations": evaluated_pairs,
                "sample_data": sample_data
            }, f, indent=2, ensure_ascii=False)
            
        print(f"Tier 1 Evaluation Complete. Results written to {out_path}")
        for k, v in report.items():
            print(f"{k}: {v}")
            
if __name__ == "__main__":
    evaluator = EntityResolutionEvaluator()
    try:
        sample = evaluator.generate_sample(limit=1000)
        evaluator.evaluate_orthographic_consistency(sample)
    finally:
        evaluator.close()
