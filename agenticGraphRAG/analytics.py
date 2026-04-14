import os
import glob
import json
import tqdm
import csv
from collections import Counter

class GlobalGraphAnalyzer:
	def __init__(self, target_uid=None):
		self.target_uid = target_uid
		self.data_folder = None
		
		# Global Statistics
		self.stats = {
			"files_processed": 0,
			"total_companies": 0,
			"total_people": 0,
			"total_events": 0,
			"total_edges": 0
		}
		
		# Activity & Metadata Cache
		self.activity_counter = Counter()
		self.entity_props = {}   # UID -> {name, type, city}
		
		# Target Deep Dive (if target_uid is set)
		self.target_edges = []
		self.target_neighbors = {}

	def process_folder(self, folder_path):
		"""Scans all JSON files to build stats and index the graph."""
		self.data_folder = folder_path
		json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
		
		print(f"🚀 Starting Global Scan on {len(json_files)} files...")
		
		for fpath in tqdm.tqdm(json_files, desc="Indexing Graph"):
			try:
				with open(fpath, 'r', encoding='utf-8') as f:
					data = json.load(f)
					self._ingest_file(data)
			except Exception as e:
				print(f"❌ Error reading {os.path.basename(fpath)}: {e}")

	def _ingest_file(self, data):
		"""Ingests a single file's data into the global state."""
		# 1. Update Global Counts
		self.stats["files_processed"] += 1
		self.stats["total_companies"] += len(data.get("companies", {}))
		self.stats["total_people"] += len(data.get("people", {}))
		self.stats["total_events"] += len(data.get("events", []))
		self.stats["total_edges"] += len(data.get("edges", []))

		# 2. Cache Metadata (Name, Type) for Companies & People
		# We merge 'companies' and 'people' dicts for easier lookup
		for group, label in [('companies', 'Company'), ('people', 'Person')]:
			for uid, node in data.get(group, {}).items():
				if uid not in self.entity_props:
					self.entity_props[uid] = {
						'name': node['properties'].get('name', 'Unknown'),
						'type': label,
						'city': node['properties'].get('city', node['properties'].get('town', '-'))
					}
				
				# If this entity is a neighbor of our target, save it specifically
				if self.target_uid and uid in self.target_neighbors:
					 self.target_neighbors[uid] = self.entity_props[uid]

		# 3. Process Edges (Activity + Target Search)
		for edge in data.get("edges", []):
			# Robust Key Handling (source_id vs source)
			src = edge.get('source_id') or edge.get('source')
			tgt = edge.get('target_id') or edge.get('target')
			
			if not src or not tgt: continue
			
			# A. Count Activity (Skip metadata nodes)
			if not src.startswith("name_") and not src.startswith("SHAB"):
				self.activity_counter[src] += 1
			if not tgt.startswith("name_") and not tgt.startswith("SHAB"):
				 self.activity_counter[tgt] += 1

			# B. Target Analysis (Only if target_uid is set)
			if self.target_uid and (src == self.target_uid or tgt == self.target_uid):
				self.target_edges.append(edge)
				neighbor = tgt if src == self.target_uid else src
				# Mark neighbor to fetch metadata in step 2 (or next file)
				if neighbor not in self.target_neighbors:
					self.target_neighbors[neighbor] = {} 

	def print_report(self, top_n=15):
		"""Prints the full analysis report to console."""
		print("\n" + "="*60)
		print("🌍 GLOBAL GRAPH STATISTICS")
		print("="*60)
		print(f"Files Processed: {self.stats['files_processed']}")
		print(f"Total Companies: {self.stats['total_companies']:,} (Occurrences)")
		print(f"Total People:    {self.stats['total_people']:,} (Occurrences)")
		print(f"Total Events:    {self.stats['total_events']:,}")
		print(f"Total Edges:     {self.stats['total_edges']:,}")
		
		print(f"\n🏆 TOP {top_n} MOST ACTIVE ENTITIES")
		print(f"{'RANK':<5} | {'TYPE':<10} | {'COUNT':<8} | {'NAME'}")
		print("-" * 60)
		
		# Get Top N
		for i, (uid, count) in enumerate(self.activity_counter.most_common(top_n), 1):
			props = self.entity_props.get(uid, {})
			name = props.get('name', uid)
			etype = props.get('type', '?')
			print(f"{i:<5} | {etype:<10} | {count:<8} | {name}")

		if self.target_uid:
			self._print_target_deep_dive()

	def _print_target_deep_dive(self):
		t_props = self.entity_props.get(self.target_uid, {})
		t_name = t_props.get('name', self.target_uid)
		
		print("\n" + "="*80)
		print(f"🔎 DEEP DIVE: {t_name} ({self.target_uid})")
		print("="*80)
		print(f"{'ROLE':<25} | {'NEIGHBOR':<20} | {'CITY':<15} | {'NAME'}")
		print("-" * 80)
		
		count = 0
		for edge in self.target_edges:
			src = edge.get('source_id') or edge.get('source')
			tgt = edge.get('target_id') or edge.get('target')
			
			is_source = (src == self.target_uid)
			neighbor_id = tgt if is_source else src
			role = edge.get('relation_type') or edge.get('type')
			
			props = self.target_neighbors.get(neighbor_id, self.entity_props.get(neighbor_id, {}))
			n_name = props.get('name', neighbor_id)
			n_city = props.get('city', '-')
			
			if neighbor_id.startswith("name_"): continue
			
			print(f"{role:<25} | {neighbor_id:<20} | {n_city:<15} | {n_name}")
			count += 1
		print(f"\n✅ Found {count} direct connections.")

	def export_to_gephi(self, top_n=1000, output_dir="."):
		"""Exports the Ego Graph of the Top N entities to CSVs."""
		if not self.data_folder:
			print("❌ Please run process_folder() first!")
			return

		# 1. Identify VIPs
		top_uids = set(uid for uid, _ in self.activity_counter.most_common(top_n))
		print(f"💎 Identified Top {top_n} VIPs. Streaming to CSV...")

		# 2. Prepare Writers
		nodes_path = os.path.join(output_dir, f"gephi_nodes_top{top_n}.csv")
		edges_path = os.path.join(output_dir, f"gephi_edges_top{top_n}.csv")
		
		written_nodes = set()
		
		with open(nodes_path, 'w', newline='', encoding='utf-8') as f_nodes, \
			 open(edges_path, 'w', newline='', encoding='utf-8') as f_edges:
			
			writer_nodes = csv.writer(f_nodes)
			writer_edges = csv.writer(f_edges)
			
			writer_nodes.writerow(["Id", "Label", "Type", "Degree"])
			writer_edges.writerow(["Source", "Target", "Type", "Label"])
			
			# 3. Rescan files to find connections
			json_files = sorted(glob.glob(os.path.join(self.data_folder, "*.json")))
			
			for fpath in tqdm.tqdm(json_files, desc="Exporting to CSV"):
				try:
					with open(fpath, 'r', encoding='utf-8') as f:
						data = json.load(f)
						
						for edge in data.get("edges", []):
							src = edge.get('source_id') or edge.get('source')
							tgt = edge.get('target_id') or edge.get('target')
							etype = edge.get('relation_type') or edge.get('type')
							
							if not src or not tgt: continue

							# If edge involves a VIP, keep it
							if src in top_uids or tgt in top_uids:
								writer_edges.writerow([src, tgt, "Directed", etype])
								
								for node_id in [src, tgt]:
									if node_id not in written_nodes:
										self._write_node(node_id, writer_nodes, data)
										written_nodes.add(node_id)
				except: pass

		print(f"\n✅ Export Complete!\n   -> {nodes_path}\n   -> {edges_path}")

	def _write_node(self, uid, writer, current_file_data):
		"""Helper to find label and write node row."""
		# Try Cache
		if uid in self.entity_props:
			info = self.entity_props[uid]
			writer.writerow([uid, info['name'], info['type'], self.activity_counter[uid]])
			return

		# Try Events in current file
		for evt in current_file_data.get('events', []):
			if evt['id'] == uid:
				lbl = f"{evt['properties'].get('rubric', 'Event')} {evt['properties'].get('date', '')}"
				writer.writerow([uid, lbl, "Event", 1])
				return
		
		# Try Name Hubs
		if uid.startswith("name_"):
			writer.writerow([uid, uid.replace("name_", ""), "NameHub", 1])
			return

		# Unknown
		writer.writerow([uid, uid, "Unknown", 0])

	def export_snowball_gephi(self, start_uid, hops=2, output_dir="."):
		"""
		Extracts a connected subgraph starting from ONE node and expanding outwards.
		Guarantees a connected component.
		"""
		if not self.data_folder:
			print("❌ Please run process_folder() first!")
			return

		print(f"\n❄️ STARTING SNOWBALL SAMPLING")
		print(f"   - Seed: {start_uid}")
		print(f"   - Hops: {hops}")

		# Sets to track the graph
		collected_edges = []
		collected_nodes = set([start_uid])
		
		# The "Frontier" is the set of nodes we need to expand from in the current pass
		current_frontier = set([start_uid])
		
		json_files = sorted(glob.glob(os.path.join(self.data_folder, "*.json")))

		# --- MULTI-PASS SCAN (One pass per hop) ---
		for current_hop in range(1, hops + 1):
			print(f"\n🌊 HOP {current_hop}/{hops}: Scanning files for neighbors of {len(current_frontier)} nodes...")
			
			next_frontier = set()
			
			for fpath in tqdm.tqdm(json_files, desc=f"Hop {current_hop}"):
				try:
					with open(fpath, 'r', encoding='utf-8') as f:
						data = json.load(f)
						
						# Cache nodes in this file just in case we need their labels later
						self._cache_file_props(data)

						for edge in data.get("edges", []):
							src = edge.get('source_id') or edge.get('source')
							tgt = edge.get('target_id') or edge.get('target')
							etype = edge.get('relation_type') or edge.get('type')
							
							if not src or not tgt: continue

							# CHECK: Is this edge connected to our current frontier?
							# And avoid 'Name Hubs' to keep the graph interesting (optional, but recommended)
							if "name_" in src or "name_" in tgt: continue

							is_connected = False
							neighbor = None

							if src in current_frontier:
								is_connected = True
								neighbor = tgt
							elif tgt in current_frontier:
								is_connected = True
								neighbor = src
							
							if is_connected:
								# Found a relevant edge!
								edge_tuple = (src, tgt, etype)
								collected_edges.append(edge_tuple)
								
								# Add neighbor to next frontier IF we haven't seen it yet
								if neighbor not in collected_nodes:
									next_frontier.add(neighbor)
									collected_nodes.add(neighbor)

				except: pass
			
			print(f"   ✅ Hop {current_hop} finished. Found {len(next_frontier)} new neighbors.")
			if not next_frontier:
				print("   ⚠️ Dead end reached. Stopping early.")
				break
			
			current_frontier = next_frontier

		# --- WRITE TO CSV ---
		print(f"\n💾 Writing {len(collected_nodes)} nodes and {len(collected_edges)} edges to CSV...")
		
		nodes_path = os.path.join(output_dir, f"snowball_{start_uid}_h{hops}_nodes.csv")
		edges_path = os.path.join(output_dir, f"snowball_{start_uid}_h{hops}_edges.csv")

		with open(nodes_path, 'w', newline='', encoding='utf-8') as f_n, \
			 open(edges_path, 'w', newline='', encoding='utf-8') as f_e:
			
			wn = csv.writer(f_n)
			we = csv.writer(f_e)
			
			wn.writerow(["Id", "Label", "Type", "Degree"])
			we.writerow(["Source", "Target", "Type", "Label"])
			
			# Write Edges
			for s, t, rel in collected_edges:
				we.writerow([s, t, "Directed", rel])
			
			# Write Nodes
			for uid in collected_nodes:
				props = self.entity_props.get(uid, {})
				lbl = props.get('name', uid)
				typ = props.get('type', 'Unknown')
				
				# Try to pretty-print Events
				if uid.startswith("SHAB"): 
					typ = "Event"
					lbl = "Event"
				
				wn.writerow([uid, lbl, typ, 1])

		print(f"✅ DONE. Download these files: \n   {nodes_path}\n   {edges_path}")

	def _cache_file_props(self, data):
		"""Helper to cache node props on the fly during snowballing"""
		for group, label in [('companies', 'Company'), ('people', 'Person')]:
			for uid, node in data.get(group, {}).items():
				if uid not in self.entity_props:
					self.entity_props[uid] = {
						'name': node['properties'].get('name', uid),
						'type': label
					}