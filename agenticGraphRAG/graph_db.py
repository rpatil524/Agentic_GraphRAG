import json
import time
import tqdm
from neo4j import GraphDatabase
from collections import defaultdict
from .utils import generate_hub_key

class Neo4jIngester:
	"""
	Handles bulk ingestion of SHAB JSON data into Neo4j.
	Now includes:
	1. Extended Company Properties (address, legal_form, etc.)
	2. Event Text Fix (prioritizing 'full_text')
	3. Built-in Analytics (PageRank & Louvain)
	4. Entity Resolution (Deduplication of Weak Nodes)
	"""
	def __init__(self, uri, auth, database="neo4j"):
		self.driver = GraphDatabase.driver(uri, auth=auth)
		self.database = database
		self.verify_connection()

	def verify_connection(self):
		try:
			self.driver.verify_connectivity()
			print(f"✅ Connected to Neo4j ({self.database})")
		except Exception as e:
			print(f"❌ Connection Failed: {e}")
			raise e

	def close(self):
		self.driver.close()

	def create_constraints(self):
		print("⚡ Configuring database indexes...")
		queries = [
			"CREATE CONSTRAINT uid_global IF NOT EXISTS FOR (n:BaseNode) REQUIRE n.uid IS UNIQUE",
			"CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
			"CREATE INDEX person_name  IF NOT EXISTS FOR (p:Person)  ON (p.name)",
			"CREATE INDEX event_date   IF NOT EXISTS FOR (e:Event)   ON (e.date)",
			
			# --- NEW: GLOBAL SEARCH INDEX ---
			# This enables "Google-style" search across names and raw text.
			"CREATE FULLTEXT INDEX global_search IF NOT EXISTS FOR (n:BaseNode) ON EACH [n.name, n.text, n.rubric]"
		]
		with self.driver.session(database=self.database) as session:
			for q in queries:
				session.run(q)
		print("✅ Database optimized with Full-Text Search.")

	def deduplicate_name_hubs(self):
		"""
		Consolidates duplicate NameHubs (e.g., 'Martin Kauter' and 'Kauter Martin')
		by applying the alphabetized token sorting directly in the database.
		Uses Cypher UNWIND for lightning-fast bulk processing.
		"""
		print("🔤 Deduplicating Name Hubs (Alphabetical Tokenization)...")
		
		with self.driver.session(database=self.database) as session:
			result = session.run("MATCH (n:NameHub) RETURN n.uid AS uid, n.name AS name")
			hubs = [{"uid": r["uid"], "name": r["name"]} for r in result]
			
		print(f"   📊 Found {len(hubs)} total Name Hubs. Grouping by token order...")
		
		grouped_hubs = defaultdict(list)
		for hub in hubs:
			new_key = generate_hub_key(hub["name"])
			grouped_hubs[new_key].append(hub)

		duplicates_to_merge = {k: v for k, v in grouped_hubs.items() if len(v) > 1}
		
		if not duplicates_to_merge:
			print("   ✅ No duplicates found! Hubs are perfectly clean.")
			return

		print(f"   ⚠️ Found {len(duplicates_to_merge)} sets of duplicated hubs. Preparing bulk payload...")
		
		batch = []
		for new_key, duplicates in duplicates_to_merge.items():
			master_uid = duplicates[0]["uid"]
			old_uids = [d["uid"] for d in duplicates[1:]]
			for old_uid in old_uids:
				batch.append({"master_uid": master_uid, "old_uid": old_uid})

		print(f"   ⚡ Sending {len(batch)} merge operations to Neo4j in chunks...")

		chunk_size = 1000 
		with self.driver.session(database=self.database) as session:
			for i in tqdm.tqdm(range(0, len(batch), chunk_size), desc="Bulk Merging"):
				chunk = batch[i:i + chunk_size]
				
				# Step A: Rewire edges in bulk
				session.run("""
					UNWIND $chunk AS row
					MATCH (old:BaseNode {uid: row.old_uid})
					MATCH (master:BaseNode {uid: row.master_uid})
					MATCH (spoke)-[r:HAS_NAME]-(old)
					MERGE (spoke)-[:HAS_NAME]->(master)
					DELETE r
				""", chunk=chunk)
				
				# Step B: Delete isolated old hubs in bulk
				session.run("""
					UNWIND $chunk AS row
					MATCH (old:BaseNode {uid: row.old_uid})
					DELETE old
				""", chunk=chunk)

		print("   🎉 Hub Deduplication Complete!")

	def resolve_weak_nodes(self):
		"""
		Cleans up the database by deleting redundant 'Weak' nodes that were
		extracted by the LLM but already exist as 'Strong' nodes in the same event.
		"""
		print("🧹 Running Entity Resolution (Deduplication)...")
		# We generalize the query to work for both Companies and People.
		# It finds any Weak node sharing a NameHub and an Event with a Strong node.
		query = """
		MATCH (weak {is_weak: True})-[:HAS_NAME]->(hub:NameHub)<-[:HAS_NAME]-(strong {is_weak: False})
		MATCH (weak)-[r_weak:ACTED_IN]->(e:Event)<-[r_strong]-(strong)
		WITH weak, r_weak
		DELETE r_weak
		DETACH DELETE weak
		RETURN count(weak) AS deleted_nodes
		"""
		try:
			with self.driver.session(database=self.database) as session:
				result = session.run(query)
				record = result.single()
				deleted_count = record["deleted_nodes"] if record else 0
				print(f"✅ Resolution complete! Removed {deleted_count} redundant weak nodes.")
		except Exception as e:
			print(f"❌ Error during entity resolution: {e}")

	def run_analytics(self, algorithms=["risk", "communities"]):
		"""
		Runs Graph Data Science algorithms on the current data.
		:param algorithms: List of algos to run. Options: 'risk' (PageRank), 'communities' (Louvain)
		"""
		print(f"📊 Starting Analytics: {algorithms}...")
		graph_name = "shab_analytics"
		
		with self.driver.session(database=self.database) as session:
			# 1. CLEANUP OLD GRAPHS
			session.run(f"CALL gds.graph.drop('{graph_name}', false)")

			# 2. PROJECT GRAPH
			print("   - Projecting graph into memory...")
			session.run(f"""
				CALL gds.graph.project(
					'{graph_name}',
					'BaseNode',
					{{
						RELATIONSHIP: {{
							type: '*',
							orientation: 'UNDIRECTED'
						}}
					}}
				)
			""")

			# 3. COMPUTE RISK (PageRank)
			if "risk" in algorithms:
				print("   - Calculating Risk Scores (PageRank)...")
				session.run(f"""
					CALL gds.pageRank.write('{graph_name}', {{
						writeProperty: 'risk_rank',
						maxIterations: 20,
						dampingFactor: 0.85
					}})
				""")

			# 4. COMPUTE COMMUNITIES (Louvain)
			if "communities" in algorithms:
				print("   - Detecting Communities (Louvain)...")
				session.run(f"""
					CALL gds.louvain.write('{graph_name}', {{
						writeProperty: 'community_id'
					}})
				""")

			# 5. CLEANUP
			session.run(f"CALL gds.graph.drop('{graph_name}', false)")
		
		print("✅ Analytics Complete.")

	def ingest_file(self, file_path):
		try:
			with open(file_path, 'r', encoding='utf-8') as f:
				data = json.load(f)
		except Exception as e:
			print(f"⚠️ Read Error {file_path}: {e}")
			return

		with self.driver.session(database=self.database) as session:
			# --- 1. NODES ---
			
			# Companies (With your extended fields)
			if data.get('companies'):
				batch = []
				for uid, props in data['companies'].items():
					p = props.get('properties', {})
					batch.append({
						'uid': uid, 
						'name': p.get('name'), 
						'city': p.get('city'),
						'address': p.get('address'),
						'legal_form': p.get('legal_form'),
						'deletion_date': p.get('deletion_date'),
						'purpose': p.get('purpose'),
						'capital_nominal': p.get('capital_nominal'),
						'capital_paid': p.get('capital_paid'),
						'is_weak': p.get('is_weak'),
						'source': p.get('source')
					})
				
				session.run("""
					UNWIND $batch AS row
					MERGE (n:BaseNode {uid: row.uid})
					SET n:Company, n.name = row.name, n.city = row.city,
						n.address = row.address, n.legal_form = row.legal_form,
						n.deletion_date = row.deletion_date, n.purpose = row.purpose,
						n.capital_nominal = row.capital_nominal, n.capital_paid = row.capital_paid,
						n.is_weak = row.is_weak, n.source = row.source
				""", batch=batch)

			# People
			if data.get('people'):
				batch = []
				for uid, props in data['people'].items():
					p = props.get('properties', {})
					batch.append({
						'uid': uid, 
						'name': p.get('name'), 
						'origin': p.get('origin'),
						'town': p.get('town'),
						'dob': p.get('dob'),
						'is_weak': p.get('is_weak'),
						'source': p.get('source')
					})

				session.run("""
					UNWIND $batch AS row
					MERGE (n:BaseNode {uid: row.uid})
					SET n:Person, n.name = row.name, n.origin = row.origin,
						n.town = row.town, n.dob = row.dob,
						n.is_weak = row.is_weak, n.source = row.source
				""", batch=batch)

			# Events (With FULL_TEXT priority)
			if data.get('events'):
				batch = []
				for evt in data['events']:
					props = evt.get('properties', {})
					# Priority: full_text -> text -> publication -> summary
					text_content = props.get('full_text') or props.get('text') or props.get('publication') or props.get('summary')
					
					batch.append({
						'uid': evt['id'], 
						'date': props.get('date'), 
						'rubric': props.get('rubric'),
						'sub_rubric': props.get('sub_rubric'),
						'text': text_content
					})
					
				session.run("""
					UNWIND $batch AS row
					MERGE (n:BaseNode {uid: row.uid})
					SET n:Event, n.date = row.date, n.rubric = row.rubric, n.text = row.text, n.sub_rubric = row.sub_rubric
				""", batch=batch)
				
			# Name Hubs
			if data.get('name_hub'):
				batch = [
					{'uid': hub['id'], 'name': hub['properties'].get('value')}
					for hub in data['name_hub'].values()
				]
				session.run("""
					UNWIND $batch AS row
					MERGE (n:BaseNode {uid: row.uid})
					SET n:NameHub, n.name = row.name
				""", batch=batch)

			# --- 2. EDGES ---
			if data.get('edges'):
				edges_by_type = defaultdict(list)
				for edge in data['edges']:
					src = edge.get('source_id') or edge.get('source')
					tgt = edge.get('target_id') or edge.get('target')
					raw_type = edge.get('relation_type') or edge.get('type') or "RELATED_TO"
					safe_type = "".join(c for c in raw_type if c.isalnum() or c == "_").upper()
					
					if src and tgt:
						edges_by_type[safe_type].append({'source': src, 'target': tgt})

				for rtype, batch in edges_by_type.items():
					query = f"""
						UNWIND $batch AS row
						MATCH (s:BaseNode {{uid: row.source}}), (t:BaseNode {{uid: row.target}})
						MERGE (s)-[r:{rtype}]->(t)
					"""
					session.run(query, batch=batch)

	def anonymize_graph(self, secret_key: str, output_path: str):
		"""
		Anonymizes sensitive properties in the Neo4j database using HMAC-SHA256.
		Updates 'name' and 'address' on BaseNodes, and 'text' on Events.
		Saves a local JSON mapping of {hashed_value: original_value}.
		"""
		import hmac
		import hashlib
		import os

		print("🔐 Anonymizing Graph (HMAC-SHA256)...")
		if not secret_key:
			print("❌ Secret key is empty, cannot anonymize.")
			return

		mapping = {}

		def compute_hash(val):
			if not val:
				return None
			return hmac.new(secret_key.encode('utf-8'), str(val).encode('utf-8'), hashlib.sha256).hexdigest()

		# Step 1: Collect properties to anonymize
		# We collect node ID, name, address, and text. Events have 'text', Companies have 'address', Names are across nodes.
		with self.driver.session(database=self.database) as session:
			print("   - Fetching properties to anonymize...")
			result = session.run("MATCH (n:BaseNode) RETURN id(n) AS node_id, n.name AS name, n.address AS address, n.text AS text")
			
			updates = []
			for record in tqdm.tqdm(result, desc="Computing Hashes"):
				node_id = record["node_id"]
				updates_for_node = {'node_id': node_id}
				changed = False
				
				if record["name"]:
					hashed_name = compute_hash(record["name"])
					mapping[hashed_name] = record["name"]
					updates_for_node['name'] = hashed_name
					changed = True
					
				if record["address"]:
					hashed_address = compute_hash(record["address"])
					mapping[hashed_address] = record["address"]
					updates_for_node['address'] = hashed_address
					changed = True
					
				if record["text"]:
					hashed_text = compute_hash(record["text"])
					mapping[hashed_text] = record["text"]
					updates_for_node['text'] = hashed_text
					changed = True
					
				if changed:
					updates.append(updates_for_node)

		if not updates:
			print("   ✅ No properties found to anonymize.")
			return

		# Step 2: Bulk update the Neo4j nodes
		print(f"   - Writing {len(updates)} node updates back to Neo4j...")
		chunk_size = 1000
		with self.driver.session(database=self.database) as session:
			for i in tqdm.tqdm(range(0, len(updates), chunk_size), desc="Bulk Updating DB"):
				chunk = updates[i:i+chunk_size]
				session.run("""
					UNWIND $chunk AS row
					MATCH (n) WHERE id(n) = row.node_id
					SET 
						n.name = coalesce(row.name, n.name),
						n.address = coalesce(row.address, n.address),
						n.text = coalesce(row.text, n.text)
				""", chunk=chunk)

		# Step 3: Save mapping
		print(f"   - Saving translation mapping to {output_path}...")
		try:
			os.makedirs(os.path.dirname(output_path), exist_ok=True)
			with open(output_path, 'w', encoding='utf-8') as f:
				json.dump(mapping, f, indent=2, ensure_ascii=False)
			print("✅ Anonymization complete.")
		except Exception as e:
			print(f"❌ Error saving translation mapping: {e}")
