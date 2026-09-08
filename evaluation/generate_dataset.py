import os
import json
import time
import random
import asyncio
import aiohttp
from tqdm import tqdm
from neo4j import GraphDatabase, READ_ACCESS

from _bootstrap import REPO_ROOT

# --- CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "shabdb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATASET_MODEL = os.getenv("OPENAI_DATASET_MODEL", "gpt-5-2025-08-07")
OUTPUT_PATH = os.getenv(
    "AUTOMATED_DATASET_PATH",
    str(REPO_ROOT / "evaluation" / "datasets" / "automated_dataset.json"),
)

def get_neo4j_session():
    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is required.")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

def run_cypher_query(driver, query, params=None):
    with driver.session(
        database=NEO4J_DATABASE,
        default_access_mode=READ_ACCESS,
    ) as session:
        result = session.run(query, params or {})
        return [r.data() for r in result]

async def generate_qa_pair_async(session, subgraph_json, question_type, sem):
    """
    Calls OpenAI asynchronously to generate a Q&A pair based strictly on the provided subgraph.
    """
    system_prompt = """
    You are an expert Forensic Investigator and Dataset Generator for a Swiss Corporate Graph Database.
    You will be provided with a JSON subgraph extracted from the real database.
    Your task is to generate ONE highly realistic, challenging question that a real investigator would ask, which requires understanding the provided subgraph to answer.
    Then, provide the exact expected answer, using ONLY the facts present in the JSON.
    
    Difficulty Levels:
    Level 1: Direct (e.g., "What is the purpose of Company X?", "Who is the liquidator of Y?")
    Level 2: Multi-hop (e.g., "Which companies share board members with Company X?", "Does Person Y own any bankrupt companies?")
    Level 3: Temporal/Complex (e.g., "Summarize the history of events for Company X", "Trace the address changes of Company Y over time.")

    Return ONLY a valid JSON object with these keys:
    {
        "question_text": "...",
        "difficulty_level": "Level X: [Name]",
        "expected_answer": "..."
    }
    """
    
    prompt = f"Target Question Type: {question_type}\n\nSUBGRAPH DATA:\n{json.dumps(subgraph_json, indent=2)}\n\nGenerate the JSON Q&A pair now."
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": DATASET_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    async with sem:
        for attempt in range(4):
            try:
                async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
                    if response.status == 429:
                        print("Rate limited by OpenAI. Waiting to retry...")
                        await asyncio.sleep(10 * (attempt + 1))
                        continue
                    response.raise_for_status()
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    return json.loads(content)
            except Exception as e:
                print(f"Error calling OpenAI (Attempt {attempt+1}/4): {e}")
                await asyncio.sleep(2)
        return None

def fetch_subgraphs(driver):
    """
    Fetches different types of interesting subgraphs from Neo4j to seed the questions.
    """
    subgraphs = []
    print("Fetching subgraphs from Neo4j...")
    
    # Type 1: Multi-hop Corporate Hierarchy (Company -> Person -> Company)
    query_2hop = """
    MATCH (c1:Company)-[:HAS_EVENT]->(e1:Event)<-[r1:ACTED_IN|INVOLVED_IN]-(p:Person)-[r2:ACTED_IN|INVOLVED_IN]->(e2:Event)<-[:HAS_EVENT]-(c2:Company)
    WHERE c1 <> c2
    WITH c1, p, c2, r1, r2, e1, e2 LIMIT 150
    RETURN {
        type: "2-hop Network",
        company1: c1.name,
        company1_uid: c1.uid,
        person: p.name,
        person_role_c1: type(r1),
        event1_rubric: e1.rubric,
        company2: c2.name,
        company2_uid: c2.uid,
        person_role_c2: type(r2),
        event2_rubric: e2.rubric
    } AS subgraph
    """
    res = run_cypher_query(driver, query_2hop)
    for r in res:
        subgraphs.append((r['subgraph'], 'Multi-hop Corporate Connection'))

    # Type 2: Temporal Event History (Company with >3 events)
    query_events = """
    MATCH (c:Company)-[r:HAS_EVENT|INVOLVED_IN]->(e:Event)
    WITH c, collect({date: e.date, rubric: e.rubric, text: e.text}) AS events
    WHERE size(events) >= 3
    WITH c, events LIMIT 150
    RETURN {
        type: "Temporal History",
        company: c.name,
        uid: c.uid,
        events: events
    } AS subgraph
    """
    res = run_cypher_query(driver, query_events)
    for r in res:
        subgraphs.append((r['subgraph'], 'Temporal Event Summary'))

    # Type 3: Direct Extraction (Company Properties)
    query_direct = """
    MATCH (c:Company)
    WHERE c.purpose IS NOT NULL OR c.address IS NOT NULL
    WITH c LIMIT 150
    RETURN {
        type: "Direct Entity Data",
        company: c.name,
        uid: c.uid,
        purpose: c.purpose,
        address: c.address,
        capital_nominal: c.capital_nominal
    } AS subgraph
    """
    res = run_cypher_query(driver, query_direct)
    for r in res:
        subgraphs.append((r['subgraph'], 'Direct Property Extraction'))

    # Type 4: NameHub Resolution
    query_hub = """
    MATCH (n:NameHub)<-[:HAS_NAME]-(p:Person)-[r:ACTED_IN|INVOLVED_IN]->(e:Event)<-[:HAS_EVENT]-(c:Company)
    WITH n, collect(DISTINCT {person_uid: p.uid, company: c.name, role: type(r)}) AS connections
    WHERE size(connections) > 1
    WITH n, connections LIMIT 150
    RETURN {
        type: "NameHub Entity Resolution",
        namehub: n.name,
        hub_uid: n.uid,
        connections: connections
    } AS subgraph
    """
    res = run_cypher_query(driver, query_hub)
    for r in res:
        subgraphs.append((r['subgraph'], 'NameHub Disambiguation'))

    print(f"Fetched {len(subgraphs)} source subgraphs.")
    return subgraphs

async def process_subgraph(session, subgraph, q_type, tier, sem):
    try:
        qa_pair = await generate_qa_pair_async(session, subgraph, q_type, sem)
        if qa_pair:
            qa_pair['tier'] = tier
            qa_pair['ground_truth_subgraph'] = subgraph
            return qa_pair
    except Exception as e:
        print(f"Error processing {q_type}: {e}")
    return None

async def main_async():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to generate benchmark questions.")
        
    driver = get_neo4j_session()
    try:
        subgraphs = fetch_subgraphs(driver)
        if not subgraphs:
            raise RuntimeError("No source subgraphs were found. Is shabdb populated?")

        dataset = []
        tier_mapping = {
            "Direct Property Extraction": "Level 1",
            "Multi-hop Corporate Connection": "Level 2",
            "NameHub Disambiguation": "Level 2",
            "Temporal Event Summary": "Level 3",
        }

        subgraphs_by_tier = {"Level 1": [], "Level 2": [], "Level 3": []}
        random.shuffle(subgraphs)
        for subgraph, question_type in subgraphs:
            tier = tier_mapping.get(question_type, "Level 1")
            subgraphs_by_tier[tier].append((subgraph, question_type))

        print("Generating automated benchmark (target: 100 questions per level)...")
        sem = asyncio.Semaphore(5)
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for tier in ["Level 1", "Level 2", "Level 3"]:
                print(f"Generating for {tier}...")
                tier_dataset = []
                chunk_size = 50
                index = 0
                tier_subgraphs = subgraphs_by_tier[tier]

                with tqdm(total=100, desc=tier) as progress:
                    while len(tier_dataset) < 100:
                        if index >= len(tier_subgraphs):
                            print(
                                f"\nWarning: exhausted {len(tier_subgraphs)} source "
                                f"subgraphs for {tier}; generated {len(tier_dataset)} questions."
                            )
                            break

                        chunk = tier_subgraphs[index : index + chunk_size]
                        index += chunk_size
                        tasks = [
                            process_subgraph(session, subgraph, question_type, tier, sem)
                            for subgraph, question_type in chunk
                        ]
                        for task in asyncio.as_completed(tasks):
                            result = await task
                            if result and len(tier_dataset) < 100:
                                tier_dataset.append(result)
                                result["question_id"] = (
                                    f"Q{len(dataset) + len(tier_dataset):03d}"
                                )
                                progress.update(1)

                dataset.extend(tier_dataset)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
            json.dump(dataset, output_file, indent=2, ensure_ascii=False)

        print(f"\nCreated automated benchmark with {len(dataset)} questions at {OUTPUT_PATH}")
    finally:
        driver.close()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
