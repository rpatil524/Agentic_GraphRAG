import pandas as pd
import tqdm
import json
from .models import SHABCompany, SHABPerson, SHABEvent, SHABEdge
from .extractors import real_llm_extractor, mock_llm_extractor
# Updated import to include new constants
from .constants import PUBLISHER_STOPLIST, SKELETON_RUBRICS, LLM_TARGET_SUBRUBRICS
from .utils import clean_text, generate_hub_key

class SHABPipeline:
	def __init__(self, api_key=None):
		self.companies = {} 
		self.people = {}    
		self.events = []
		self.name_hub = {}        
		self.edge_objects = []
		self.api_key = api_key

	@staticmethod
	def scan_rubric_statistics(folder_path):
		"""
		Scans all CSVs in a folder and returns counts for every sub-rubric found.
		Useful for analyzing data distribution before processing.
		
		Args:
			folder_path (str): Path to the folder containing CSV files.
			
		Returns:
			dict: A nested dictionary {rubric: {sub_rubric: count}}
		"""
		import glob
		import os
		from collections import defaultdict

		stats = defaultdict(lambda: defaultdict(int))
		csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

		if not csv_files:
			print(f"⚠️ No CSV files found in {folder_path}")
			return {}

		print(f"📊 Scanning {len(csv_files)} files in {os.path.basename(folder_path)}...")

		for csv_file in csv_files:
			try:
				# Read ONLY the rubric columns to be fast and save RAM
				df = pd.read_csv(csv_file, usecols=['meta_rubric', 'meta_subRubric'], low_memory=False)
				
				# aggregate counts
				counts = df.groupby(['meta_rubric', 'meta_subRubric']).size()
				
				for (rubric, sub_rubric), count in counts.items():
					stats[rubric][sub_rubric] += count
					
			except Exception as e:
				print(f"   ❌ Error reading {os.path.basename(csv_file)}: {e}")

		# Convert to standard dict for clean output
		final_stats = {k: dict(v) for k, v in stats.items()}
		return final_stats

	# --- STEP 0: STATIC LOADER ---
	@staticmethod
	def load_skeleton_data(csv_path, limit=None):
		"""
		Step 0: Loads CSV and filters for SKELETON_RUBRICS.
		Args:
			csv_path (str): Full path to the CSV file.
			limit (int, optional): If set, returns only the top N rows (for testing).
		Returns:
			pd.DataFrame: The filtered DataFrame ready for Phase 1.
		"""
		try:
			# 1. Load
			df = pd.read_csv(csv_path, low_memory=False)
			
			# 2. Filter using the Constant
			df = df[df['meta_rubric'].isin(SKELETON_RUBRICS)].reset_index(drop=True)
			
			# 3. Apply Limit (if testing)
			if limit:
				df = df.head(limit)
				
			return df
		except Exception as e:
			print(f"❌ Error loading skeleton data: {e}")
			return pd.DataFrame() # Return empty DF on failure

	def identify_candidates(self):
		"""
		Helper: Returns a list of events that match LLM_TARGET_SUBRUBRICS.
		"""
		return [
			evt for evt in self.events 
			if evt['properties'].get('sub_rubric') in LLM_TARGET_SUBRUBRICS
		]

	def link_to_name_hub(self, entity_id, raw_name, clean_name_key):
		"""
		Creates the Name Node if missing, and links the Entity to it.
		Schema: (:Person)-[:HAS_NAME]->(:Name)
		"""
		# 1. Create/Get Name Node (The Hub)
		hub_id = f"name_{clean_name_key}"
		if clean_name_key not in self.name_hub:
			self.name_hub[clean_name_key] = {
				'id': hub_id,
				'label': 'Name',
				'properties': {'value': raw_name}
			}
		
		# 2. Create the Spoke (Edge)
		self.edge_objects.append(SHABEdge(
			source_id=entity_id,
			target_id=hub_id,
			relation_type='HAS_NAME'
		))

	def run_phase_1_structured_ingestion(self, df):
		"""
		Phase 1: Process CSV Rows (Strong Nodes)
		"""
		# print("Processing Phase 1: CSV Rows (Strong Nodes)...")
		
		for index, row in df.iterrows():
			# 1. Process Strong Company
			comp = SHABCompany.from_row(row)
			if comp:
				self.companies[comp.uid] = comp.to_dict()
				# Use generate_hub_key instead of clean_text
				clean_comp_name = generate_hub_key(comp.name) 
				self.link_to_name_hub(comp.uid, comp.name, clean_comp_name)
				
			# 2. Process Strong Person
			pers = SHABPerson.from_row(row)
			if pers:
				self.people[pers.id] = pers.to_dict()
				
				# Link Strong Person to Name Hub
				if pers.name_key:
					self.link_to_name_hub(pers.id, pers.name, pers.name_key)
					
			# 3. Process Event
			evt = SHABEvent(row)
			self.events.append(evt.to_dict())

	def run_phase_2_unstructured_ingestion(self, use_mock=False, batch_size=10, subset_events=None):
		"""
		Phase 2: LLM Extraction
		Args:
			subset_events (list): Optional. If provided, ONLY process these events. 
								  If None, process ALL events in self.events.
		"""
		# 1. Determine Scope (The Logic You Need)
		target_list = subset_events if subset_events is not None else self.events
		
		if not target_list:
			# print("Phase 2: No events to process.")
			return

		# print(f"Processing Phase 2: LLM Extraction on {len(target_list)} events (Batch Size: {batch_size})...")
		
		if use_mock:
			iterator = self.events
			for evt in iterator:
				entities = mock_llm_extractor(evt['id'], evt['properties']['full_text'])
				self._process_extracted_entities(evt['id'], entities, is_mock=True)
			return

		# --- Real LLM Batch Processing with Adaptive Retry ---
		from .extractors import batch_real_llm_extractor_openai
		
		# --- 1. Define the Recursive Safety Wrapper ---
		def process_safe_batch(current_batch):
			"""
			Tries to process a batch. If it fails (returns empty dict), 
			splits the batch in half and tries the halves recursively.
			"""
			# A. Try processing the batch
			results = batch_real_llm_extractor_openai(current_batch, api_key=self.api_key)
			
			# B. Success Check
			# If 'results' is not empty (it contains keys), it worked.
			if results: 
				return results
			
			# C. Failure Handling (Recursive Split)
			# If results is empty {} and we have more than 1 item, we split.
			if len(current_batch) > 1:
				mid = len(current_batch) // 2
				print(f"   [!] Batch of {len(current_batch)} failed or timed out. Splitting ({mid}/{len(current_batch)-mid})...")
				
				left_batch = current_batch[:mid]
				right_batch = current_batch[mid:]
				
				# Recursive calls
				results_left = process_safe_batch(left_batch)
				results_right = process_safe_batch(right_batch)
				
				# Merge results dictionaries
				return {**results_left, **results_right}
			
			# D. Poison Pill
			# If we are down to 1 item and it still fails, log and skip it.
			if len(current_batch) == 1:
				bad_id = current_batch[0]['id']
				print(f"   [❌] Item {bad_id} completely failed LLM processing.")
			
			return {}

		# --- 2. Main Processing Loop ---
		# Prepare all items
		all_items = [{'id': e['id'], 'text': e['properties']['full_text']} for e in target_list]
		
		# Slice into initial large batches
		batches = [all_items[i:i + batch_size] for i in range(0, len(all_items), batch_size)]
		
		# NOTE: Removed TQDM for parallel workers to avoid messy console output
		for batch in batches:
			# Call the adaptive wrapper
			batch_results = process_safe_batch(batch)
			
			if not isinstance(batch_results, dict):
				batch_results = {}
				
			# Process results
			for item in batch:
				event_id = item['id']
				entities = batch_results.get(event_id, [])
				self._process_extracted_entities(event_id, entities, is_mock=False)

	def _process_extracted_entities(self, event_id, extracted_entities, is_mock=False):
		"""Helper to create nodes and edges from raw entities"""
		# Guard: entities must be a list of dicts, not a string/None/etc.
		if not isinstance(extracted_entities, list):
			return
		for entity in extracted_entities:
			if not isinstance(entity, dict):
				continue
			try:
				# [SAFETY FIX] Skip Publishers immediately
				if entity['name'].upper() in PUBLISHER_STOPLIST:
					continue

				source_id = None

				if entity['type'] == 'Person':
					weak_pers = SHABPerson.from_text(entity['name'], event_id)
					self.people[weak_pers.id] = weak_pers.to_dict()
					
					if weak_pers.name_key:
						self.link_to_name_hub(weak_pers.id, weak_pers.name, weak_pers.name_key)
					source_id = weak_pers.id
		
				elif entity['type'] == 'Organization':
					weak_comp = SHABCompany.from_text(entity['name'], event_id)
					if weak_comp.uid not in self.companies:
						self.companies[weak_comp.uid] = weak_comp.to_dict()
						# Use generate_hub_key instead of clean_text
						clean_name_comp = generate_hub_key(weak_comp.name)
						self.link_to_name_hub(weak_comp.uid, weak_comp.name, clean_name_comp)
					source_id = weak_comp.uid
		
				if source_id:
					self.edge_objects.append(SHABEdge(
						source_id=source_id,
						target_id=event_id,
						relation_type='ACTED_IN',
						properties={
							'role': entity['role'], 
							'confidence': entity.get('confidence'),
							'extraction_source': 'MOCK' if is_mock else 'LLM'
						}
					))

			except Exception as e:
				print(f"Error processing entity: {entity}")
				print(e)

	def run_phase_3_structured_edges(self):
		"""
		Phase 3: Generate Structured Edges from Event connections
		"""
		# print("Generating Structured Edges (Appending to existing graph)...")
		
		count_new_edges = 0
		
		for evt in self.events:
			event_id = evt['id']
			connections = evt['connections']
			
			# Iterate through our defined connection roles
			for role, uid in connections.items():
				if uid:
					
					# --- VALIDATION: NO WEAK NODES ---
					is_person = (role == 'DEBTOR_PERSON')
					exists_in_companies = (not is_person and uid in self.companies)
					exists_in_people = (is_person and uid in self.people)
					
					if exists_in_companies or exists_in_people:
						
						# --- MAPPING ROLES TO EDGE TYPES ---
						edge_type = 'RELATED_TO' # Default
						
						if role == 'SUBJECT':        edge_type = 'HAS_EVENT'
						elif role == 'PARENT':       edge_type = 'HEAD_OFFICE_OF'
						elif role == 'SELLER':       edge_type = 'TRANSFERRED_TO'
						elif role == 'BUYER':        edge_type = 'ACQUIRED_FROM'
						elif role == 'DEBTOR_COMPANY': edge_type = 'HAS_EVENT' 
						elif role == 'DEBTOR_PERSON':  edge_type = 'INVOLVED_IN'
						elif role == 'DISSOLVED':    edge_type = 'DISSOLVED_IN'
						elif role == 'ASSURANCE':    edge_type = 'PROVIDES_ASSURANCE_TO'

						# Create Edge
						self.edge_objects.append(SHABEdge(
							source_id=uid,
							target_id=event_id,
							relation_type=edge_type,
							properties={'role': role}
						))
						count_new_edges += 1
						
		# print(f"✅ Added {count_new_edges} structured edges.")
		# print(f"   Total Graph Edges: {len(self.edge_objects)}")

	def save_checkpoint(self, filename="shab_graph_checkpoint.json"):
		# print("Saving Graph Data to JSON...")
		
		# Convert edge objects to dicts for JSON serialization
		serialized_edges = [e.to_dict() for e in self.edge_objects]
		
		graph_dump = {
			"companies": self.companies,
			"people": self.people,
			"events": self.events,
			"name_hub": self.name_hub,
			"edges": serialized_edges
		}
		
		with open(filename, 'w', encoding='utf-8') as f:
			json.dump(graph_dump, f, ensure_ascii=False, indent=2)
			
		# RETURN the stats so the script can print them
		return {
			"companies": len(self.companies),
			"people": len(self.people),
			"name_hub": len(self.name_hub),
			"edges": len(serialized_edges)
		}