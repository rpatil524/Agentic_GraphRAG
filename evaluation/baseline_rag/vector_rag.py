"""
Naive Vector RAG Baseline for the SHAB domain.
Simulates traditional RAG architecture (chunking -> embeddings -> vector DB -> LLM).
This serves as the comparative baseline to prove the efficacy of the Graph RAG system.
"""

import json
import os
import time
import asyncio
import re
from tqdm import tqdm
from openai import AsyncOpenAI, OpenAI
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from neo4j import GraphDatabase

class NaiveVectorRAG:
    def __init__(self, api_key=None, db_path="evaluation/baseline_rag/chroma_db", concurrency=15):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required.")
        self.client = OpenAI(api_key=self.api_key)           # sync, used for non-eval calls
        self.async_client = AsyncOpenAI(api_key=self.api_key) # async, used for eval
        self.concurrency = concurrency
        
        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # We use ChromaDB's default local embedding function (all-MiniLM-L6-v2)
        # This allows you to ingest millions of records locally for free without OpenAI rate limits.
        local_ef = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="shab_events_local",
            embedding_function=local_ef
        )

    def ingest_from_neo4j(self, reset=False):
        """
        Pulls ALL raw event strings from the graph database and embeds them into the Vector DB.
        Uses a proper driver streaming architecture and Python-side batching to handle 
        millions of rows without causing Neo4j Memory Pool OOMs or crashing Python.
        """
        if reset:
            print("Resetting ChromaDB Collection...")
            try:
                self.chroma_client.delete_collection("shab_events_local")
            except:
                pass
            local_ef = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.chroma_client.get_or_create_collection(
                name="shab_events_local",
                embedding_function=local_ef
            )
            
        print("Connecting to Neo4j to pull ALL raw events for Vector Embeddings extraction...")
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        if not password:
            raise RuntimeError("NEO4J_PASSWORD is required.")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        try:
            with driver.session() as session:
                # 1. Get exact total count for the progress bar
                count_res = session.run("MATCH (e:Event) WHERE e.text IS NOT NULL RETURN count(e) as total")
                total_events = count_res.single()["total"]
                
                if total_events == 0:
                    print("No events found in Neo4j to ingest!")
                    return
                    
                print(f"Database contains {total_events:,} events with text.")
                
                # Pre-fetch existing IDs to make ingestion resumable without re-embedding
                print("Checking ChromaDB for existing embedded documents...")
                existing_data = self.collection.get(include=[])
                existing_ids = set(existing_data.get("ids", []))
                
                if existing_ids:
                    print(f"Found {len(existing_ids):,} existing documents. Resuming from where we left off...")
                else:
                    print("Starting fresh ingestion...")
                
                print("Streaming events from Neo4j and Embedding into ChromaDB (Ctrl+C to stop)...")
                
                # 2. Execute a stream query. By default, the Python driver fetches 1000 records at a time,
                # which keeps memory usage extremely low for both Python and Neo4j.
                result = session.run("MATCH (e:Event) WHERE e.text IS NOT NULL RETURN e.uid AS uid, e.text AS text")
                
                batch_size = 500
                batch_ids = []
                batch_docs = []
                
                # 3. Stream and Upsert in chunks
                with tqdm(total=total_events, initial=len(existing_ids)) as pbar:
                    for record in result:
                        uid = record["uid"]
                        
                        # Skip if already embedded in a previous run
                        if uid in existing_ids:
                            continue
                            
                        batch_ids.append(uid)
                        batch_docs.append(record["text"])
                        
                        if len(batch_ids) >= batch_size:
                            # Upsert overwrites existing IDs so we don't infinitely duplicate on re-runs
                            self.collection.upsert(
                                documents=batch_docs,
                                ids=batch_ids
                            )
                            pbar.update(len(batch_ids))
                            batch_ids = []
                            batch_docs = []
                            
                    # Flush the remainder
                    if batch_ids:
                        self.collection.upsert(
                            documents=batch_docs,
                            ids=batch_ids
                        )
                        pbar.update(len(batch_ids))
                        
        finally:
            driver.close()
            
        print("✅ Vector DB Ingestion Complete.")

    def retrieve(self, query_text, k=3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k
        )
        # Results['documents'] is a list of lists (one list per query)
        if results['documents'] and results['documents'][0]:
            return results['documents'][0]
        return []

    def build_query_variants(self, question):
        variants = [question.strip()]
        quoted = re.findall(r'"([^"]+)"', question)
        variants.extend(quoted)

        patterns = [
            r"with whom does (.+?) share an activity\??",
            r"show all the companies with which (.+?) is affiliated\.?",
            r"show all the companies with which (.+?) is connected\.?",
            r"what companies are connected to (.+?)\??",
            r"have any notices of dissolution or bankruptcy been published for (.+?)\??",
            r"what is the registered head office or base of (.+?)\??",
        ]

        extracted = None
        lower_q = question.lower()
        for pattern in patterns:
            match = re.search(pattern, lower_q, flags=re.IGNORECASE)
            if match:
                start, end = match.span(1)
                extracted = question[start:end].strip().strip(".?")
                break

        if extracted:
            variants.append(extracted)
            variants.append(f'"{extracted}"')
            if "," in extracted:
                parts = [p.strip() for p in extracted.split(",") if p.strip()]
                if len(parts) == 2:
                    variants.append(" ".join(reversed(parts)))

        # preserve order while deduplicating
        deduped = []
        seen = set()
        for item in variants:
            key = item.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def retrieve_expanded(self, question, k_per_query=10, max_docs=25):
        docs = []
        seen = set()
        for query in self.build_query_variants(question):
            for doc in self.retrieve(query, k=k_per_query):
                key = str(doc)
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
                if len(docs) >= max_docs:
                    return docs
        return docs

    def generate_answer(self, question, context_docs):
        # Determine format of context docs (string vs dict)
        parsed_docs = []
        for doc in context_docs:
            if isinstance(doc, dict):
                parsed_docs.append(str(doc.get("text", doc)))
            else:
                parsed_docs.append(str(doc))
                
        context = "\n".join(parsed_docs)
        if not context.strip():
            context = "NO CONTEXT RETRIEVED."
            
        prompt = f"""You are an assistant answering questions about the Swiss Commercial Register based ONLY on the provided context.

Your job is to extract the answer as directly as possible from the context.
If the question asks for a list of companies or people, list all matching names you can support from the context.
If the question asks for a date or location, return only the relevant date or location.
Do not add extra commentary. Do not say more than the context supports.
If the information is not in the context, say briefly that you do not know.

Context:
{context}

Question: {question}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    async def generate_answer_async(self, question, context_docs, sem):
        """Async version of generate_answer with semaphore-controlled concurrency."""
        parsed_docs = []
        for doc in context_docs:
            if isinstance(doc, dict):
                parsed_docs.append(str(doc.get("text", doc)))
            else:
                parsed_docs.append(str(doc))

        context = "\n".join(parsed_docs)
        if not context.strip():
            context = "NO CONTEXT RETRIEVED."

        prompt = f"""You are an assistant answering questions about the Swiss Commercial Register based ONLY on the provided context.

Your job is to extract the answer as directly as possible from the context.
If the question asks for a list of companies or people, list all matching names you can support from the context.
If the question asks for a date or location, return only the relevant date or location.
Do not add extra commentary. Do not say more than the context supports.
If the information is not in the context, say briefly that you do not know.

Context:
{context}

Question: {question}
"""
        async with sem:
            for attempt in range(4):
                try:
                    response = await self.async_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if "429" in str(e):
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        break
        return "Error: failed after retries."

    async def _generate_answer_no_sem(self, question, context_docs):
        """Internal helper without semaphore to avoid double locking."""
        parsed_docs = []
        for doc in context_docs:
            if isinstance(doc, dict):
                parsed_docs.append(str(doc.get("text", doc)))
            else:
                parsed_docs.append(str(doc))

        context = "\n".join(parsed_docs)
        if not context.strip():
            context = "NO CONTEXT RETRIEVED."

        prompt = f"""You are an assistant answering questions about the Swiss Commercial Register based ONLY on the provided context.

Your job is to extract the answer as directly as possible from the context.
If the question asks for a list of companies or people, list all matching names you can support from the context.
If the question asks for a date or location, return only the relevant date or location.
Do not add extra commentary. Do not say more than the context supports.
If the information is not in the context, say briefly that you do not know.

Context:
{context}

Question: {question}
"""
        for attempt in range(4):
            try:
                response = await self.async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    break
        return "Error: failed after retries."

    async def evaluate_golden_dataset_async(
        self,
        dataset_path="evaluation/datasets/automated_dataset.json",
        output_path="evaluation/baseline_results.json",
        limit=None,
    ):
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found! Please run generate_dataset.py first.")
            return

        with open(dataset_path, "r") as f:
            questions = json.load(f)
        if limit is not None:
            questions = questions[:limit]

        sem = asyncio.Semaphore(self.concurrency)
        print(f"Running async Baseline Vector RAG Evaluation on {len(questions)} queries...")

        async def process_question(q):
            async with sem:
                start = time.time()
                # Run retrieval in thread to prevent blocking event loop
                context = await asyncio.to_thread(self.retrieve_expanded, q["question_text"], k_per_query=10, max_docs=25)
                answer  = await self._generate_answer_no_sem(q["question_text"], context)
                latency = time.time() - start
                return {
                    "question_id":       q["question_id"],
                    "question":          q["question_text"],
                    "level":             q["difficulty_level"],
                    "baseline_answer":   answer,
                    "expected_answer":   q["expected_answer"],
                    "retrieved_context": context,
                    "latency_seconds":   latency,
                }

        tasks   = [process_question(q) for q in questions]
        results = []

        with tqdm(total=len(tasks), desc="Baseline RAG") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        avg_latency = sum(r["latency_seconds"] for r in results) / len(results) if results else 0
        output_data = {
            "metrics": {
                "Average_Latency": f"{avg_latency:.2f}s",
                "Total_Evaluated": len(results)
            },
            "results": results
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Baseline evaluation complete. Avg Latency: {avg_latency:.2f}s. Results saved to {output_path}")
        return output_data

if __name__ == "__main__":
    baseline = NaiveVectorRAG(api_key=os.getenv("OPENAI_API_KEY"))

    # Normally, you run ingestion once:
    # asyncio.run(baseline.ingest_from_neo4j(limit=5000))

    asyncio.run(baseline.evaluate_golden_dataset_async())
