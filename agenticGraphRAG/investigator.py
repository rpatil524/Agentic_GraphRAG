import os
import json
import requests
import re
from datetime import datetime

class SHABInvestigator:
	"""
	The Semantic Agent Engine for the SHAB Network.
	Uses Tool Calling, Reflexion, and Secure Sandboxed Analytics.
	"""
	def __init__(self, driver, api_key, model="gpt-4o-mini", iterations=4, database=None):
		self.driver = driver
		self.api_key = api_key
		self.model = model
		self.iterations = iterations
		self.database = database
		
		# --- SEMANTIC LAYER: TOOL SCHEMAS ---
		self.tools_schema = [
			{
				"type": "function",
				"function": {
					"name": "search_companies",
					"description": "Find companies by combining various optional filters like name, location, or purpose.",
					"parameters": {
						"type": "object",
						"properties": {
							"name": {"type": "string", "description": "Name of the company"},
							"uid": {"type": "string", "description": "Unique ID (CHE-...) of the company"},
							"location": {"type": "string", "description": "City or location of the company"},
							"purpose": {"type": "string", "description": "Keywords related to the company's business purpose (e.g., crypto, real estate)"},
							"limit": {"type": "integer", "description": "How many results to return. Default is 15. Maximum is 25. If user asks for more than 25, use 25."},
							"offset": {"type": "integer", "description": "Pagination offset. Increase this by the limit to get the next page of results."}
						},
						"additionalProperties": False
					}
				}
			},
			{
				"type": "function",
				"function": {
					"name": "explore_network",
					"description": "Find all related entities connected to a specific entity. For a Person, this ALREADY finds co-board members and associates across ALL their companies in one go. Do NOT iterate companies manually.",
					"parameters": {
						"type": "object",
						"properties": {
							"uid": {
								"type": "string", 
								"description": "The UID of the Company or Person you want to investigate."
							},
							"person_name": {
								"type": "string", 
								"description": "Anchor person's full name (optional, use uid if known)."
							},
							"filter_entity_type": {
								"type": "string",
								"enum": ["Company", "Person", "All"],
								"description": "CRITICAL: Filter output to only show Companies or People. Use 'Company' if the user asks 'Which companies...', use 'Person' if 'Who is involved...'."
							},
							"connection_type": {
								"type": "string", 
								"enum": ["owners", "founders", "liquidators", "bankruptcy", "structure", "mergers", "all"],
								"description": "Filter for specific relationship types. Use 'structure' for parent/child links, 'mergers' for M&A, or 'all' for everything."
							},
							"limit": {"type": "integer", "description": "How many results to return. Default 15. Maximum 25. If user asks for more than 25, use 25."},
							"offset": {"type": "integer", "description": "Pagination offset. MANDATORY for paging. Page 1 = 0. Page 2 = limit. Page 3 = limit*2. ALWAYS increment by limit."}
						},
						"additionalProperties": False
					}
				}
			},
			{
				"type": "function",
				"function": {
					"name": "get_node_history",
					"description": "Get the chronological timeline of events and legal text for a specific company or person.",
					"parameters": {
						"type": "object",
						"properties": {
							"uid": {"type": "string", "description": "The unique ID (UID) of the entity"},
							"limit": {"type": "integer", "description": "How many results to return. Default 15. Maximum 25. If user asks for more than 25, use 25."},
							"offset": {"type": "integer", "description": "Pagination offset. MANDATORY for paging. Page 1 = 0. Page 2 = limit. Page 3 = limit*2."}
						},
						"required": ["uid"],
						"additionalProperties": False
					}
				}
			},
			{
				"type": "function",
				"function": {
					"name": "get_top_entities",
					"description": "Canned analytics: Get a ranked list of entities based on a specific metric (e.g., most events, highest risk score), optionally filtered by location or keyword.",
					"parameters": {
						"type": "object",
						"properties": {
							"entity_type": {"type": "string", "enum": ["Company", "Person"], "description": "Type of entity to rank."},
							"metric": {"type": "string", "enum": ["event_count", "risk_rank", "capital_nominal"], "description": "The metric to sort by."},
							"limit": {"type": "integer", "description": "How many results to return. Default 15. Maximum 25. If user asks for more than 25, use 25."},
							"offset": {"type": "integer", "description": "Pagination offset. MANDATORY for paging. Page 1 = 0. Page 2 = limit. Page 3 = limit*2."},
							"location": {"type": "string", "description": "Optional city or location filter (e.g., 'Zug', 'Zurich')."},
							"keyword": {"type": "string", "description": "Optional keyword to filter the company purpose or name (e.g., 'crypto', 'real estate')."}
						},
						"required": ["entity_type", "metric"],
						"additionalProperties": False
					}
				}
			},
			{
				"type": "function",
				"function": {
					"name": "global_text_search",
					"description": "Deep search across the entire database text. ONLY use this if the user explicitly asks for 'text search' or 'mentions'.",
					"parameters": {
						"type": "object",
						"properties": {
							"query": {"type": "string", "description": "The exact name or keyword to search for."},
							"limit": {"type": "integer", "description": "Max results. Default 20. Maximum 25."},
							"offset": {"type": "integer", "description": "Pagination offset. Increase this by the limit to get the next page of results."}
						},
						"required": ["query"],
						"additionalProperties": False
					}
				}
			},
			{
				"type": "function",
				"function": {
					"name": "count_entities_by_event",
					"description": "Count the number of companies or people involved in specific legal events.",
					"parameters": {
						"type": "object",
						"properties": {
							"event_type": {
								"type": "string",
								"enum": ["bankruptcy", "liquidation", "new_foundation", "all"],
								"description": "The category of the legal event."
							},
							"entity_type": {
								"type": "string",
								"enum": ["Company", "Person"],
								"description": "Whether to count companies or people. Defaults to Company."
							},
							"keyword": {
								"type": "string",
								"description": "Optional keyword to filter by industry or purpose (e.g., 'manufacturing', 'crypto')."
							},
							"location": {
								"type": "string",
								"description": "Optional city or canton filter."
							},
							"limit": {"type": "integer", "description": "Number of examples to return in the sample list. Default 10. Maximum 25."},
							"offset": {"type": "integer", "description": "Pagination offset to view more examples in the sample list."}
						},
						"required": ["event_type"],
						"additionalProperties": False
					}
				}
			},
			{
				"type": "function",
				"function": {
					"name": "execute_custom_cypher",
					"description": "LAST RESORT Analytics: Write custom Cypher for complex counting or unmapped aggregations.",
					"parameters": {
						"type": "object",
						"properties": {
							"cypher_query": {"type": "string", "description": "The raw Cypher query to execute. MUST NOT contain mutation keywords like DELETE or SET."}
						},
						"required": ["cypher_query"],
						"additionalProperties": False
					}
				}
			}
		]

	def _call_llm(self, messages, temperature=0.0, tools=None):
		"""Helper to call OpenAI API via requests, now supporting Tool Calling"""
		headers = {
			"Content-Type": "application/json",
			"Authorization": f"Bearer {self.api_key}"
		}
		payload = {
			"model": self.model,
			"messages": messages,
			"temperature": temperature
		}
		
		if tools:
			payload["tools"] = tools
			payload["parallel_tool_calls"] = False
			
		try:
			resp = requests.post("https://api.openai.com/v1/chat/completions", 
							   headers=headers, json=payload, timeout=90)
			resp.raise_for_status()
			return resp.json()['choices'][0]['message']
		except Exception as e:
			return {"role": "assistant", "content": f"Error calling LLM: {e}"}

	def execute_query(self, cypher_query, params=None):
		"""Retriever. Runs the Cypher query."""
		try:
			with self.driver.session(database=self.database) as session:
				result = session.run(cypher_query, params or {})
				return [r.data() for r in result]
		except Exception as e:
			return [{"error": str(e)}]

	# ---------------------------------------------------------
	# --- SEMANTIC LAYER: CORE TOOLS ---
	# ---------------------------------------------------------

	def _paginate_results(self, results, limit, offset=0):
		"""Helper to enforce strict limit and detect if more results exist."""
		# Logic: If we got more than the requested limit, we know there's a next page.
		# We truncate the list to the requested limit so the LLM doesn't get overwhelmed.
		if len(results) > limit:
			next_offset = offset + limit
			paginated = results[:limit]
			# Append explicit instruction for the LLM
			paginated.append({
				"SYSTEM_NOTE": "🔴 MORE RESULTS AVAILABLE 🔴",
				"NEXT_STEP_INSTRUCTION": f"The user has NOT seen the full list. You MUST ask: 'Would you like to see the next {limit} results?'.",
				"TOOL_usage": f"If user says 'yes', call THIS SAME TOOL with offset={next_offset}."
			})
			return paginated
		return results

	def search_companies(self, name=None, uid=None, location=None, purpose=None, limit=15, offset=0):
		limit = min(int(limit), 25)
		conditions = []
		# FETCH ONE EXTRA to detect if there are more pages
		params = {
			"limit": limit + 1,
			"offset": int(offset)
		}
		
		conditions.append("(n:Company OR n:Person)")

		# --- 🟢 WHAT CHANGED: ADDED THE ORPHAN WEAK NODE FILTER ---
		# This pattern comprehension checks if the node is Strong (is_weak is false/null).
		# If it IS weak, it traverses 2 hops (Weak Node -> Name Hub -> Strong Node) 
		# to see if a Strong twin exists. If size() is 0, it's an orphan and we keep it.
		conditions.append("""
		(NOT n.uid STARTS WITH 'weak_' OR 
		size([(n)-[:HAS_NAME*2]-(strong) WHERE NOT strong.uid STARTS WITH 'weak_' | strong]) = 0)
		""")
		# ----------------------------------------------------------

		if uid:
			conditions.append("n.uid = $uid")
			params["uid"] = uid
		if name:
			tokens = name.split()
			for i, token in enumerate(tokens):
				param_name = f"name_{i}"
				conditions.append(f"toLower(n.name) CONTAINS toLower(${param_name})")
				params[param_name] = token
		if location:
			conditions.append("toLower(n.city) CONTAINS toLower($location)")
			params["location"] = location
		if purpose:
			conditions.append("toLower(n.purpose) CONTAINS toLower($purpose)")
			params["purpose"] = purpose

		where_clause = " WHERE " + " AND ".join(conditions)

		cypher = f"""
		MATCH (n:BaseNode)
		{where_clause}
		
		// 1. Find the central Name Hub for this entity (if it exists)
		OPTIONAL MATCH (n)-[:HAS_NAME]-(hub)
		
		// 2. Group by the Hub's ID (or fall back to the raw name if no hub exists)
		WITH coalesce(hub.uid, n.name) AS deduplication_key, collect(n)[0] AS n 
		
		RETURN n.uid AS uid, n.name AS name, n.city AS location, n.address AS address, n.purpose AS purpose, n.risk_rank AS risk_score, n.uid STARTS WITH 'weak_' AS is_weak
		ORDER BY risk_score DESC, name ASC, uid ASC
		SKIP $offset LIMIT $limit
		"""
		results = self.execute_query(cypher, params)
		return self._paginate_results(results, limit, int(offset))

	def global_text_search(self, query, limit=20, offset=0):
		limit = min(int(limit), 25)
		lucene_query = " ".join([f"+{word}" for word in query.split()])

		cypher = """
		CALL db.index.fulltext.queryNodes("global_search", $query) YIELD node, score
		WHERE score > 0.3 
		RETURN 
			labels(node)[0] as type,
			coalesce(node.name, 'Event') as name,
			node.uid as uid,
			substring(coalesce(node.text, node.purpose, ''), 0, 1000) as snippet,
			node.date as date
		SKIP toInteger($offset)
		LIMIT toInteger($limit)
		"""
		# Update the params to use our new strictly-formatted lucene_query
		return self.execute_query(cypher, {"query": lucene_query, "limit": limit, "offset": offset})

	def _expand_analytics_keyword_terms(self, keyword):
		base = str(keyword or "").strip().lower()
		if not base:
			return []
		synonyms = {
			"crypto": [
				"crypto",
				"cryptocurrency",
				"cryptocurrencies",
				"blockchain",
				"distributed ledger",
				"token",
				"tokenization",
				"digital asset",
				"digital assets",
				"virtual asset",
				"virtual assets",
				"web3",
				"defi",
			],
			"blockchain": [
				"blockchain",
				"distributed ledger",
				"crypto",
				"token",
				"tokenization",
				"digital asset",
				"digital assets",
				"web3",
			],
		}
		return synonyms.get(base, [base])

	def count_entities_by_event(self, event_type, entity_type="Company", keyword=None, location=None, limit=10, offset=0):
		# This one is tricky because of 'collect'. We apply limit inside the collect logic usually.
		# For simplicity, we just pass the raw limit here as it returns a 'sample' list, not a full paginated set usually.
		conditions = []
		params = {
			"limit": min(int(limit), 25),
			"offset": int(offset)
		}

		if event_type == "bankruptcy":
			conditions.append("(e.rubric = 'KK' OR e.sub_rubric IN ['KK02', 'KK03', 'KK06'])")
		elif event_type == "liquidation":
			conditions.append("(e.rubric = 'LS' OR e.sub_rubric IN ['LS01', 'LS02'])")
		elif event_type == "new_foundation":
			conditions.append("e.sub_rubric = 'HR01'")
			
		if location:
			conditions.append("toLower(n.city) CONTAINS toLower($location)")
			params["location"] = location

		if keyword:
			if entity_type == "Company":
				keyword_terms = self._expand_analytics_keyword_terms(keyword)
				conditions.append("""
				(
					any(term IN $keyword_terms WHERE toLower(coalesce(n.purpose, '')) CONTAINS term)
					OR any(term IN $keyword_terms WHERE toLower(coalesce(n.name, '')) CONTAINS term)
					OR any(term IN $keyword_terms WHERE toLower(coalesce(e.text, '')) CONTAINS term)
				)
				""")
				params["keyword_terms"] = keyword_terms
			else:
				conditions.append("toLower(n.name) CONTAINS toLower($keyword)")
			params["keyword"] = keyword

		where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
		
		cypher = f"""
		MATCH (n:{entity_type})-[:HAS_EVENT|ACTED_IN]-(e:Event)
		{where_clause}
		WITH DISTINCT n
		ORDER BY n.name ASC, n.uid ASC
		WITH count(n) AS total_count, collect({{uid: n.uid, name: n.name}}) AS all_matches
		RETURN total_count, 
			   '{event_type}' AS event_type, 
			   '{keyword or 'all'}' AS industry_filter,
			   all_matches[$offset .. $offset + $limit] AS sample_entities
		"""
		return self.execute_query(cypher, params)

	def explore_network(self, uid=None, person_name=None, filter_entity_type="All", connection_type="all", limit=15, offset=0):
		limit = min(int(limit), 25)
		
		# FETCH ONE EXTRA for pagination detection
		params = {
			"limit": limit + 1,
			"offset": int(offset),
			"filter": filter_entity_type
		}

		if person_name and not uid:
			# Fix: If the router accidentally passes a NameHub UID as person_name, just use it directly
			if person_name.startswith("name_"):
				uid = person_name
			else:
				from agenticGraphRAG.utils import generate_hub_key
				hub_key = "name_" + generate_hub_key(person_name)
				
				resolve_query = """
				WITH [token IN split(toLower(replace(replace($name, ',', ' '), '.', ' ')), ' ')
				      WHERE token <> ''] AS query_tokens
				MATCH (h:NameHub)-[:HAS_NAME]-(n:Person)
				WITH h, n, query_tokens,
				     [token IN split(toLower(replace(replace(coalesce(h.name, n.name), ',', ' '), '.', ' ')), ' ')
				      WHERE token <> ''] AS candidate_tokens
				WHERE size(query_tokens) > 0
				  AND all(token IN query_tokens WHERE token IN candidate_tokens)
				RETURN h.uid AS uid, count(DISTINCT n) AS support, min(coalesce(n.is_weak, false)) AS has_strong_match
				ORDER BY has_strong_match ASC, support DESC, uid ASC
				LIMIT 1
				"""
				resolution = self.execute_query(resolve_query, {"name": person_name})
				if resolution:
					# When a matching NameHub exists, prefer querying the hub itself so
					# global mode explores the full shared-name cluster just like dossier mode.
					uid = resolution[0]["uid"]
				elif hub_key.startswith("name_"):
					# Backward-compatible fallback for databases where hub UIDs happen
					# to match the generated key directly.
					direct_hub_query = """
					MATCH (h:NameHub {uid: $hub_key})
					RETURN h.uid AS uid
					LIMIT 1
					"""
					resolution = self.execute_query(direct_hub_query, {"hub_key": hub_key})
					if resolution:
						uid = resolution[0]["uid"]
			
			if not resolution:
				resolve_query2 = """
				MATCH (n:BaseNode) 
				WHERE toLower(n.name) CONTAINS toLower($name)
				WITH n 
				ORDER BY 
					CASE WHEN 'Person' IN labels(n) THEN 1 ELSE 2 END ASC,
					coalesce(n.is_weak, true) ASC
				LIMIT 1
				RETURN n.uid AS uid
				"""
				resolution = self.execute_query(resolve_query2, {"name": person_name})
				
				if resolution: uid = resolution[0]['uid']
				else: return [{"error": f"Could not find any entity matching '{person_name}'."}]

		if uid:
			params["uid"] = uid
			
			cypher = """
			// === BRANCH 1: Identity (Event -> Subject) ===
			MATCH (startNode {uid: $uid})
			WHERE startNode:Event
			MATCH (startNode)-[r]-(connected:BaseNode)
			WHERE NOT connected:Event AND NOT connected:NameHub
			AND ($filter = 'All' OR $filter IN labels(connected))
			
			RETURN DISTINCT
				connected.uid AS connected_uid,
				connected.name AS connected_name,
				[l IN labels(connected) WHERE l <> 'BaseNode'][0] AS node_type,
				type(r) AS connection_role,
				"01_IDENTITY" AS sort_rank,
				"Direct Subject" AS rubric, 
				null AS event_date,
				"This is the main entity referenced by the event." AS event_text,
				"N/A" AS context_name

			UNION ALL

			// === BRANCH 2: Structure (Direct Neighbors) ===
			MATCH (startNode {uid: $uid})
			WHERE NOT startNode:NameHub
			OPTIONAL MATCH (startNode)-[r1]-(neighbor:BaseNode)
			WHERE NOT neighbor:Event
			AND NOT neighbor:NameHub
			AND NOT coalesce(neighbor.is_weak, false) = true
			
			WITH neighbor, type(r1) AS role, startNode
			WHERE neighbor IS NOT NULL
			AND ($filter = 'All' OR $filter IN labels(neighbor))
			
			RETURN DISTINCT
				neighbor.uid AS connected_uid,
				neighbor.name AS connected_name,
				[l IN labels(neighbor) WHERE l <> 'BaseNode'][0] AS node_type,
				role AS connection_role,
				"02_STRUCTURE" AS sort_rank,
				"Structural Link" AS rubric,
				null AS event_date,
				"Connected via direct legal structure." AS event_text,
				"Direct Link" AS context_name

			UNION ALL

			// === BRANCH 3: Full Cluster Activity (Hub <-> Spokes) ===
			MATCH (startNode {uid: $uid})
			
			// Collect siblings whether the start node IS a NameHub or is linked to one.
			OPTIONAL MATCH (startNode:NameHub)-[:HAS_NAME]-(direct_hub_sibling:BaseNode)
			OPTIONAL MATCH (startNode)-[:HAS_NAME]-(startHub:NameHub)-[:HAS_NAME]-(indirect_hub_sibling:BaseNode)
			
			// Combine startNode and all siblings into a distinct list of actors.
			// IMPORTANT: preserve startNode even if no sibling rows exist, otherwise
			// weak/company dossier exploration collapses to empty and falls back to history.
			WITH startNode,
			     [candidate IN collect(DISTINCT direct_hub_sibling) + collect(DISTINCT indirect_hub_sibling)
			      WHERE candidate IS NOT NULL] AS siblings
			WITH [startNode] + siblings AS rawCluster
			UNWIND rawCluster AS actor
			WITH DISTINCT actor WHERE actor IS NOT NULL AND NOT actor:NameHub
			
			// Step C: Find Events for this Actor
			MATCH (actor)-[:HAS_EVENT|ACTED_IN]-(e:Event)-[r]-(connected:BaseNode)
			WHERE actor <> connected AND NOT connected:NameHub
			AND connected.uid <> $uid
			AND NOT coalesce(connected.is_weak, false) = true
			AND ($filter = 'All' OR $filter IN labels(connected))

			// NEW: Find Context (Company) for the event to answer "Where?"
			OPTIONAL MATCH (e)--(ctx:Company)
			
			RETURN DISTINCT
				connected.uid AS connected_uid,
				connected.name AS connected_name,
				[l IN labels(connected) WHERE l <> 'BaseNode'][0] AS node_type,
				
				// Rename ACTED_IN to user-friendly term
				CASE 
					WHEN type(r) = 'ACTED_IN' THEN 'Associated with'
					ELSE type(r)
				END AS connection_role,
				
				"03_ACTIVITY" AS sort_rank,
				e.rubric AS rubric,
				e.date AS event_date,
				
				CASE 
					WHEN actor.is_weak = true THEN "(Via Weak Node) " + coalesce(e.text, "")
					ELSE coalesce(e.text, "")
				END AS event_text,

				coalesce(ctx.name, 'Unknown Company') AS context_name

			UNION ALL

			// === BRANCH 4: Event Context Companies (for weak/name-hub driven office nodes) ===
			MATCH (startNode {uid: $uid})
			OPTIONAL MATCH (startNode:NameHub)-[:HAS_NAME]-(direct_hub_sibling:BaseNode)
			OPTIONAL MATCH (startNode)-[:HAS_NAME]-(startHub:NameHub)-[:HAS_NAME]-(indirect_hub_sibling:BaseNode)
			WITH startNode,
			     [candidate IN collect(DISTINCT direct_hub_sibling) + collect(DISTINCT indirect_hub_sibling)
			      WHERE candidate IS NOT NULL] AS siblings
			WITH [startNode] + siblings AS rawCluster
			UNWIND rawCluster AS actor
			WITH DISTINCT actor WHERE actor IS NOT NULL AND NOT actor:NameHub
			MATCH (actor)-[:HAS_EVENT|ACTED_IN]-(e:Event)--(ctx:Company)
			WHERE ctx.uid <> $uid
			AND ($filter = 'All' OR $filter = 'Company')

			RETURN DISTINCT
				ctx.uid AS connected_uid,
				ctx.name AS connected_name,
				"Company" AS node_type,
				"HAS_EVENT" AS connection_role,
				"03_ACTIVITY" AS sort_rank,
				e.rubric AS rubric,
				e.date AS event_date,
				CASE
					WHEN actor.is_weak = true THEN "(Via Weak Node) " + coalesce(e.text, "")
					ELSE coalesce(e.text, "")
				END AS event_text,
				ctx.name AS context_name
			
			ORDER BY sort_rank ASC, event_date DESC, connected_uid ASC
			SKIP $offset LIMIT $limit
			"""
			results = self.execute_query(cypher, params)

			# Detect hub linkage independently from visible result rows. We do want to
			# traverse through a NameHub to collect sibling entities, but we do NOT want
			# to show the NameHub itself as a connected entity row.
			found_namehub = False
			if not str(uid).startswith("name_"):
				hub_link_query = """
				MATCH (n:BaseNode {uid: $uid})-[:HAS_NAME]-(h:NameHub)
				RETURN count(h) > 0 AS has_hub
				"""
				hub_link_res = self.execute_query(hub_link_query, {"uid": uid})
				found_namehub = bool(hub_link_res and hub_link_res[0].get("has_hub"))

			final_results = [r for r in results if r.get("node_type") != "NameHub"]

			# For "connected entities", keep one row per connected entity and prefer the
			# most recent supporting event. This prevents the network view from degrading
			# into an event list with duplicate companies.
			deduped_results = []
			seen_entities = set()
			for row in sorted(
				final_results,
				key=lambda r: (
					str(r.get("sort_rank", "")),
					str(r.get("event_date", "")),
					str(r.get("connected_name", "")),
				),
				reverse=True,
			):
				entity_key = row.get("connected_uid") or row.get("connected_name")
				if entity_key in seen_entities:
					continue
				seen_entities.add(entity_key)
				deduped_results.append(row)
			final_results = deduped_results
					
			notes = []
			if not str(uid).startswith("name_") and found_namehub:
				notes.append({
					"_SYSTEM_INSTRUCTION_": "DO NOT PUT THIS ROW IN THE TABLE. Append this note text AFTER the table:",
					"NOTE": "This entity is connected to a NameHub. You can open the NameHub to explore other records that might belong to the same underlying name cluster."
				})
			elif str(uid).startswith("name_"):
				# Check if this NameHub actually connects to > 1 physical node
				count_query = """
				MATCH (n:NameHub {uid: $uid})-[:HAS_NAME]-(linked:BaseNode)
				RETURN count(DISTINCT linked) AS hub_degree
				"""
				degree_res = self.execute_query(count_query, {"uid": uid})
				hub_degree = degree_res[0]['hub_degree'] if degree_res else 0
				
				if hub_degree > 1:
					notes.append({
						"_SYSTEM_INSTRUCTION_": "DO NOT PUT THIS ROW IN THE TABLE. Print this exact note text BEFORE the table:",
						"NOTE": "The entities shown below are connected to various records sharing this exact name, but the database cannot guarantee with 100% certainty that they belong to the exact same physical person."
					})
				
			# If we found no real connected entities, do not return a note-only payload.
			# A note without rows leads the LLM to fabricate placeholders instead of
			# correctly treating the structured result as empty.
			if not final_results:
				return []

			# Insert instruction notes at the top so they are never paginated out.
			final_results = notes + final_results

			return self._paginate_results(final_results, limit, int(offset))
		
		else:
			 return [{"error": "Must provide either uid or person_name"}]

	def get_node_history(self, uid, limit=15, offset=0):
		limit = min(int(limit), 25)
		params = {
			"uid": uid,
			"limit": limit + 1, # Fetch one extra
			"offset": int(offset)
		}
		# RETURN DISTINCT anchor.name AS entity_name, e.uid AS event_uid, e.date AS date, e.rubric AS rubric, e.sub_rubric AS sub_rubric, e.text AS text
		cypher = """
		MATCH (anchor:BaseNode {uid: $uid})
		MATCH (anchor)-[:HAS_NAME*0..2]-(alias:BaseNode)
		WITH DISTINCT anchor, alias
		MATCH (alias)-[:HAS_EVENT|ACTED_IN]-(e:Event)
		RETURN DISTINCT anchor.name AS entity_name, e.uid AS event_uid, e.date AS date, e.rubric AS rubric, e.sub_rubric AS sub_rubric, substring(coalesce(e.text, ''), 0, 800) AS text
		ORDER BY date DESC, event_uid ASC
		SKIP $offset LIMIT $limit
		"""
		results = self.execute_query(cypher, params)
		return self._paginate_results(results, limit, int(offset))

	def get_top_entities(self, entity_type="Company", metric="event_count", limit=15, location=None, keyword=None, offset=0):
		limit = min(int(limit), 25)
		# Fetch one extra
		offset = int(offset)
		conditions = []
		params = {}

		if metric != "event_count":
			conditions.append(f"n.{metric} IS NOT NULL")

		if location:
			conditions.append("toLower(n.city) CONTAINS toLower($location)")
			params["location"] = location

		if keyword:
			if entity_type == "Company":
				keyword_terms = self._expand_analytics_keyword_terms(keyword)
				conditions.append("""
				(
					any(term IN $keyword_terms WHERE toLower(coalesce(n.purpose, '')) CONTAINS term)
					OR any(term IN $keyword_terms WHERE toLower(coalesce(n.name, '')) CONTAINS term)
					OR EXISTS {
						MATCH (n)-[:HAS_EVENT|ACTED_IN]-(evt:Event)
						WHERE any(term IN $keyword_terms WHERE toLower(coalesce(evt.text, '')) CONTAINS term)
					}
				)
				""")
				params["keyword_terms"] = keyword_terms
			else:
				conditions.append("toLower(n.name) CONTAINS toLower($keyword)")
			params["keyword"] = keyword

		where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
		
		if metric == "event_count":
			cypher = f"""
			MATCH (n:{entity_type})
			{where_clause}
			MATCH (n)-[r:HAS_EVENT|ACTED_IN]-(e:Event)
			RETURN n.uid AS uid, n.name AS name, n.city AS location, count(e) AS total_events
			ORDER BY total_events DESC, uid ASC
			SKIP {offset} LIMIT {limit + 1}
			"""
		else:
			cypher = f"""
			MATCH (n:{entity_type})
			{where_clause}
			RETURN n.uid AS uid, n.name AS name, n.city AS location, n.{metric} AS {metric}
			ORDER BY {metric} DESC, uid ASC
			SKIP {offset} LIMIT {limit + 1}
			"""

			print(cypher)
			
		results = self.execute_query(cypher, params)
		return self._paginate_results(results, limit, int(offset))

	def execute_custom_cypher(self, cypher_query):
		upper_query = cypher_query.upper()
		blocked_keywords = r'\b(CREATE|DELETE|SET|REMOVE|MERGE|DROP|CALL|DETACH)\b'
		if re.search(blocked_keywords, upper_query):
			return [{"error": "SECURITY BLOCK: The query contains restricted mutation keywords. Only read-operations are allowed."}]
		if "LIMIT " not in upper_query:
			cypher_query += "\nLIMIT 50"
		return self.execute_query(cypher_query)

	# ---------------------------------------------------------
	# --- DYNAMIC SCHEMA & PIPELINE ---
	# ---------------------------------------------------------

	def _route_intent(self, user_question):
		router_prompt = """
		You are an Intent Classification Router. Categorize the user's question into ONE of these precise strings:
		- search_companies: Finding a company or a person by name, location, or purpose.
		- explore_network: Finding owners, founders, or network connections.
		- get_node_history: Events, history, or timeline for a specific entity.
		- analytics: Questions asking for "most", "top", "average", counting, or aggregating data across the whole database.
		- all: Complex or ambiguous queries.
		"""
		messages = [{"role": "system", "content": router_prompt}, {"role": "user", "content": user_question}]
		try:
			resp = self._call_llm(messages, temperature=0.0)
			intent = resp.get("content", "").strip().lower()
			if intent in ["search_companies", "explore_network", "get_node_history", "analytics"]:
				return intent
			return "all"
		except:
			return "all"

	def ask(self, user_question, current_uid=None, chat_history=None, trace_callback=None):
		# --- UPDATED PROMPT: STRICT PAGINATION & NO PREMATURE TEXT SEARCH ---
		# agent_prompt = """You are an expert Forensic Investigator analyzing the Swiss Commercial Register.

		# INTERACTION FLOW (STRICT):
		# 1. **Identification (Start Here)**: Always begin by using `search_companies` to find the entity.
		# - **FALLBACK RULE**: If result is empty, YOU MUST IMMEDIATELY use `global_text_search` in the same turn (with disclaimer "Results derived from unstructured text analysis").
		# - **WEAK NODE RULE**: If a result has `is_weak=True` (e.g. "weak_..."), DO NOT display it. Instead, AUTOMATICALLY use `explore_network` on that UID to find connected real companies, and display those.
		# - **FOLLOW-UP RULE**: After showing Step 1 results, YOU MUST ASK: "Do you want to explore the network connections (If yes, use `explore_network`), or the history of the company (If yes, use `get_node_history`)?"
		# 2. **Dynamic Exploration (Cross-Prompting)**: Based on the user's choice in Step 1, display the results and always ask about the option that has *not yet been shown*:
		# - If you just used `explore_network` to show the network, YOU MUST ASK: "Do you want to see the history of the company?" (If yes, use `get_node_history`).
		# - If you just used `get_node_history` to show the history, YOU MUST ASK: "Do you want to explore the network connections?" (If yes, use `explore_network`).
		# 3. **Text Deep Dive (Final Step)**: Once both the network and history have been explored (or if the user declines further node exploration), YOU MUST ASK: "Do you want me to go deeper and perform a text-based research?" (If yes, use `global_text_search` as a LAST RESORT).

		# PRESENTATION RULES (DUAL-MODE):
		# - **DOSSIER MODE (1 Result)**:
		# - Display a **Detailed Card** (Name, Purpose, Address). **NO UIDs**.
		# - **CRITICAL**: Ask about **History** or **Network**.

		# - **LIST MODE (>1 Results)**:
		# - Display a **Markdown Table**. **COLUMNS**: Select columns RELEVANT to the user's question. Always include **Name** and **Location**.
		# 	- *Example*: If user asks about "capital", show "Name | Location | Capital".
		# 	- *Example*: If user asks about "network" or "involvement", show "Name | Role | Company Context".
		# 	- *Default*: "Name | Location | Purpose | Role".
		# - **STRICT RULE**: You MUST display EVERY SINGLE result returned by the tool in your table. Do not sample, truncate, or skip items. If the tool returns 20 records, your table MUST contain exactly 20 rows!
		# - **NO UIDs**.
		# - **PAGINATION**: "Showing results [offset+1] - [offset+limit]. Would you like to see the next [limit]?"

		# PAGINATION PROTOCOL (MANDATORY):
		# - **STATE TRACKING**: You must track the `offset` of your last tool call.
		# - **NEXT PAGE**: If user says "more" or "next page":
		# 1. **SAME TOOL**: You MUST call the EXACT SAME tool as the previous turn (e.g., if you used `get_top_entities`, use it again). Do NOT switch to `get_node_history` or `explore_network`.
		# 2. **CALCULATION**: New `offset` = Old `offset` + `limit`. (e.g., 0 -> 15 -> 30).
		# 3. **EXECUTION**: Call the same tool with the new offset.

		# GENERAL RULES:
		# - **LIMIT RULE**: If the user requests a specific number of results, you MUST set the `limit` parameter to that number (up to a maximum of 25). If they request > 25, set limit=25 and explicitly state in your text response that you can only return a maximum of 25 elements per query.
		# - **EFFICIENCY RULE**: When asked "Who is [Person] involved with?", use `explore_network(person_name='...')` **ONCE** with `filter_entity_type='Person'`. Do **NOT** look up companies first and then loop. The tool manages the multi-hop lookup.
		# - **NO UIDs**: Never show the 'uid' or 'id' field in the final output.
		# - **EVENT HISTORY**: When showing history, **SUMMARIZE** the `text` field into an informative paragraph (2-3 sentences). Do not paste raw text.
		# - **FACTS ONLY**: List entities exactly as returned (unless resolving weak nodes).
		# """

		# context_str = f"\nCONTEXT: The user is currently looking at node UID='{current_uid}'." if current_uid else ""
		# messages = [{"role": "system", "content": agent_prompt + context_str}]
	
		agent_prompt = """You are an expert Forensic Investigator analyzing the Swiss Commercial Register.

			INTERACTION FLOW AND NUDGING (STRICT STATE MACHINE):
			You must guide the user through a specific investigation pipeline. Always end your response with the exact question dictated by your current state:

			CRITICAL RULE: You must ALWAYS begin an investigation by calling the `search_companies` tool. DO NOT use `global_text_search` unless `search_companies` returns absolutely zero results.

			1. **STATE 0: Disambiguation (List Mode)**
			- *Condition:* You returned multiple results (e.g., a list of companies or people).
			- *Rule:* DO NOT ask about history or networks yet.
			- *Nudge:* You must explicitly ask the user for clarification: "Which of these specific entities would you like to investigate further?"

			**DOSSIER MODE DYNAMIC STATE MACHINE:**
			When evaluating a single entity (Dossier Mode), the backend programmatically evaluates the investigation state and injects exactly what you should say next. 
			Whenever you successfully execute `explore_network` or `get_node_history`, looking at the tool results you will see a `_SYSTEM_INSTRUCTION_`. 
			You MUST follow the `_SYSTEM_INSTRUCTION_` explicitly and use it as your concluding question to the user. Do NOT invent your own follow-up questions in Dossier Mode.

			TOOL BEHAVIOR RULES:
			- **EXHAUSTIVE SEARCH RULE**: Before giving up, you MUST try every still-relevant action available in this mode. If one structured tool returns empty, you must try the other structured tools that could still answer the question before stopping.
			- **FALLBACK RULE**: If `search_companies` is empty, YOU MUST IMMEDIATELY use `global_text_search` in the same turn (with disclaimer "Results derived from unstructured text analysis").
			- **WEAK NODE RULE**: If `search_companies` returns a weak node (for example `is_weak=true` or a UID starting with `weak_`), do NOT stop there. Immediately call `explore_network` on that UID to recover connected real entities.
			- **EXHAUSTION RULE**: If the user asks for "more" connections or "more" history, or repeats a request for something you have ALREADY explored using `explore_network` or `get_node_history`, DO NOT call those tools again. Instead, you MUST immediately fall back to `global_text_search` to find deeper unstructured mentions.

			PRESENTATION RULES (DUAL-MODE):
			- **DOSSIER MODE (1 Result)**:
			- Display a **Detailed Card** (Name, Purpose, Address). **NO UIDs**.

			- **LIST MODE (>1 Results)**:
			- Display a **Markdown Table**. **COLUMNS**: Select columns RELEVANT to the user's question. Always include **Name** and **Location**.
			- **SMART FILTERING**: If a tool returns broad matches (e.g., text search returns other people with similar first names), filter them out. BUT you MUST add a note saying: *"Note: The database returned X results, but I filtered out Y partial matches."*
			- **STRICT RULE**: You MUST display EVERY SINGLE relevant result returned by the tool in your table. Do not sample, truncate, or skip items. If the tool returns 20 relevant records, your table MUST contain exactly 20 rows!
			- **NO UIDs**.
			- **PAGINATION**: "Showing results [offset+1] - [offset+limit]. Would you like to see the next [limit]?"

			PAGINATION PROTOCOL (MANDATORY):
			- **STATE TRACKING**: You must track the `offset` of your last tool call.
			- **NEXT PAGE**: If user says "more" or "next page":
			1. **SAME TOOL**: You MUST call the EXACT SAME tool as the previous turn (e.g., if you used `get_top_entities`, use it again). Do NOT switch to `get_node_history` or `explore_network`.
			2. **CALCULATION**: New `offset` = Old `offset` + `limit`. (e.g., 0 -> 15 -> 30).
			3. **EXECUTION**: Call the same tool with the new offset.

			GENERAL RULES:
			- **LIMIT RULE**: If the user requests a specific number of results, you MUST set the `limit` parameter to that number (up to a maximum of 25). If they request > 25, set limit=25 and explicitly state in your text response that you can only return a maximum of 25 elements per query.
			- **EFFICIENCY RULE**: When asked "Who is [Person] involved with?", use `explore_network(person_name='...')` **ONCE** with `filter_entity_type='Person'`. Do **NOT** look up companies first and then loop. The tool manages the multi-hop lookup.
			- **NO UIDs**: Never show the 'uid' or 'id' field in the final output.
			- **EVENT HISTORY**: When showing history, **SUMMARIZE** the `text` field into an informative paragraph (2-3 sentences). Do not paste raw text.
			- **FACTS ONLY**: List entities exactly as returned (unless resolving weak nodes).
			"""
			# - **WEAK NODE RULE**: If a result has `is_weak=True` (e.g. "weak_..."), DO NOT display it. Instead, AUTOMATICALLY use `explore_network` on that UID to find connected real companies, and display those.
		if current_uid:
			# Infer prior state from the user's chat history intents
			has_network = False
			has_history = False
			if chat_history:
				user_messages = [m.get("content", "").lower() for m in chat_history if m.get("role") == "user"]
				has_network = any(any(k in txt for k in ["network", "connect", "involv", "other compani", "relation", "tie"]) for txt in user_messages)
				has_history = any(any(k in txt for k in ["history", "event", "past", "chronolog"]) for txt in user_messages)
			
			if has_network and has_history:
				proactive_tool_rule = "immediately call `global_text_search` to find unstructured mentions in other texts."
			elif has_network:
				proactive_tool_rule = f"immediately call `get_node_history` with uid='{current_uid}'."
			elif has_history:
				proactive_tool_rule = f"immediately call `explore_network` with uid='{current_uid}'."
			else:
				proactive_tool_rule = f"immediately call `explore_network` with uid='{current_uid}' AND `get_node_history` with uid='{current_uid}' (one at a time, sequentially)."

			context_str = f"""
			PRIORITY CONTEXT RULE (OVERRIDE ALL DEFAULTS):
			The user is currently viewing a specific entity in the UI with UID='{current_uid}'.
			This means the entity is ALREADY IDENTIFIED. You MUST follow these rules immediately:
			1. DO NOT call `search_companies`. The entity is already known.
			2. If the user asks anything about "this entity", "this company", "this person", "him", "her", "it", or "what do you know" — {proactive_tool_rule}
			3. Jump directly to STATE 1 (Dossier Mode) in the investigation pipeline.
			4. SYSTEM DIRECTIVES: If you see a row in tool results containing '_SYSTEM_INSTRUCTION_', you MUST explicitly follow the instruction written there regarding table formatting and user notes!
			"""
		else:
			context_str = ""

		messages = [{"role": "system", "content": agent_prompt + context_str}]

		
		if chat_history:
			messages.extend(chat_history)
				
		messages.append({"role": "user", "content": user_question})
		
		start_of_new_messages_idx = len(messages)
		
		execution_trace = []
		final_data = []
		attempted_tools = set()
		exhausted_analytics_signatures = set()
		
		intent = self._route_intent(user_question)
		if current_uid:
			lowered_question = user_question.lower()
			if any(
				phrase in lowered_question
				for phrase in [
					"connected entities",
					"connected companies",
					"more companies",
					"find more companies",
					"more connected",
					"show network",
					"show connections",
					"connected entity",
				]
			):
				intent = "explore_network"
			elif any(
				phrase in lowered_question
				for phrase in ["history", "event history", "timeline", "past events", "show history"]
			):
				intent = "get_node_history"
		
		def add_trace(msg):
			execution_trace.append(msg)
			if trace_callback:
				trace_callback(msg)

		def tool_is_active(tool_name):
			return any(t["function"]["name"] == tool_name for t in active_tools)

		def build_empty_result_feedback(tool_name, args):
			suggestions = []
			target_uid = args.get("uid") or current_uid
			person_name = args.get("person_name")

			if tool_name == "explore_network":
				if target_uid and tool_is_active("get_node_history") and "get_node_history" not in attempted_tools:
					suggestions.append(f"Immediately try `get_node_history` with uid='{target_uid}'.")
				if person_name and tool_is_active("search_companies") and "search_companies" not in attempted_tools:
					suggestions.append(f"Resolve the entity explicitly with `search_companies(name='{person_name}')`.")
				if person_name and tool_is_active("global_text_search") and "global_text_search" not in attempted_tools:
					suggestions.append(f"If the structured path still fails, try `global_text_search(query='{person_name}')`.")
			elif tool_name == "get_node_history":
				if target_uid and tool_is_active("explore_network") and "explore_network" not in attempted_tools:
					suggestions.append(f"Immediately try `explore_network` with uid='{target_uid}'.")
				elif not target_uid and person_name and tool_is_active("search_companies") and "search_companies" not in attempted_tools:
					suggestions.append(f"Resolve the entity first with `search_companies(name='{person_name}')`.")
				if tool_is_active("global_text_search") and "global_text_search" not in attempted_tools:
					suggestions.append("If structured history is empty, try `global_text_search` before giving up.")
			elif tool_name == "search_companies":
				name = args.get("name")
				if name and tool_is_active("explore_network") and "explore_network" not in attempted_tools:
					suggestions.append(f"If the question is about relationships, try `explore_network(person_name='{name}')`.")
				if name and tool_is_active("global_text_search") and "global_text_search" not in attempted_tools:
					suggestions.append(f"If no structured match exists, immediately try `global_text_search(query='{name}')`.")
			elif tool_name == "global_text_search":
				remaining_structured = [
					name for name in ["search_companies", "explore_network", "get_node_history"]
					if tool_is_active(name) and name not in attempted_tools
				]
				if remaining_structured:
					suggestions.append(
						"Before stopping, try the remaining structured tools that still fit the question: "
						+ ", ".join(f"`{name}`" for name in remaining_structured)
						+ "."
					)
			elif tool_name in {"get_top_entities", "count_entities_by_event"}:
				locked_filters = []
				if args.get("location"):
					locked_filters.append(f"location='{args['location']}'")
				if args.get("keyword"):
					locked_filters.append(f"keyword='{args['keyword']}'")
				if args.get("metric"):
					locked_filters.append(f"metric='{args['metric']}'")
				if locked_filters:
					suggestions.append(
						"Do NOT silently broaden or drop the user's analytics filters. "
						"Keep these constraints unless the user explicitly asks to broaden the search: "
						+ ", ".join(locked_filters)
						+ "."
					)
				suggestions.append(
					"If pagination is exhausted, stop and explain that the structured analytics query returned no additional matches."
				)

			if not suggestions:
				remaining_actions = [
					name for name in ["search_companies", "explore_network", "get_node_history", "global_text_search"]
					if tool_is_active(name) and name not in attempted_tools
				]
				if remaining_actions:
					suggestions.append(
						"Before stopping, try the remaining relevant actions: "
						+ ", ".join(f"`{name}`" for name in remaining_actions)
						+ "."
					)
				else:
					suggestions.append("All relevant actions in this mode have already been tried. You may stop if you clearly explain that no structured or text results were found.")

			return "No results found with {tool}. {next_steps}".format(
				tool=tool_name,
				next_steps=" ".join(suggestions)
			)

		if intent == "all":
			active_tools = self.tools_schema
			add_trace("🧭 Router: Complex query. Loading ALL tools.")
		elif intent == "analytics":
			active_tools = [t for t in self.tools_schema if t["function"]["name"] in ["get_top_entities", "count_entities_by_event", "execute_custom_cypher"]]
			add_trace("🧭 Router: Analytics. Loaded aggregation tools.")
		else:
			# 1. Load the STANDARD INVESTIGATION SUITE (Search + Explore + History + Text)
			# We now load global_text_search by default to allow for automatic fallback.
			investigation_suite = ["search_companies", "explore_network", "get_node_history", "global_text_search"]
			active_tools = [t for t in self.tools_schema if t["function"]["name"] in investigation_suite]
			
			add_trace(f"🧭 Router: Intent '{intent}'. Loaded GRAPH tools + Text Fallback.")

		for _ in range(self.iterations):
			response_msg = self._call_llm(messages, tools=active_tools)

			content_str = response_msg.get("content")
			if content_str is None: content_str = ""
				
			if isinstance(response_msg, dict) and "Error calling LLM" in content_str:
				return {"answer": content_str, "cypher": "API Error", "data": []}
				
			messages.append(response_msg)
			
			tool_calls = response_msg.get("tool_calls")
			if not tool_calls:
				break 
				
			for tool_call in tool_calls:
				tool_name = tool_call["function"]["name"]
				args = json.loads(tool_call["function"]["arguments"])
				attempted_tools.add(tool_name)
				
				display_args = str(args) if tool_name != "execute_custom_cypher" else f"cypher_query='{args.get('cypher_query', '')}'"
				add_trace(f"🛠️ Called: {tool_name} | Args: {display_args}")
				
				try:
					# --- EXHAUSTION INTERCEPTOR ---
					intercepted = False
					if intent == "analytics" and tool_name in {"get_top_entities", "count_entities_by_event"}:
						analytics_signature = (
							tool_name,
							args.get("entity_type"),
							args.get("metric"),
							args.get("event_type"),
							args.get("location"),
							args.get("keyword"),
						)
						if analytics_signature in exhausted_analytics_signatures and int(args.get("offset", 0)) > 0:
							result_data = [{
								"error": (
									"ANALYTICS EXHAUSTED: This structured analytics query has no additional pages "
									"for the current filters. Do NOT broaden the search. Stop and explain that no "
									"additional structured matches were found unless the user explicitly changes "
									"the filters."
								)
							}]
							intercepted = True
							add_trace("⚠️ Exhaustion: Intercepted redundant analytics pagination with exhausted filters.")
					if current_uid and tool_name in ["explore_network", "get_node_history"]:
						user_messages = [m.get("content", "").lower() for m in messages if m.get("role") == "user"]
						current_user_message = user_messages[-1] if user_messages else ""
						prior_user_messages = user_messages[:-1] if len(user_messages) > 1 else []
						is_pagination_request = any(
							k in current_user_message
							for k in ["more", "next", "additional", "another", "other connected", "find more"]
						)
						if is_pagination_request and args.get("offset", 0) == 0:
							args["offset"] = int(args.get("limit", 15))
						
						has_network = any(any(k in txt for k in ["network", "connect", "involv", "other compani", "relation", "tie"]) for txt in prior_user_messages)
						has_history = any(any(k in txt for k in ["history", "event", "past", "chronolog"]) for txt in prior_user_messages)
						
						if tool_name == "explore_network" and has_network and not is_pagination_request:
							result_data = [{"error": "EXHAUSTION RULE: You have already shown the structured network connections for this entity. Do NOT return these. You MUST immediately use `global_text_search` to find deeper unstructured mentions in other texts."}]
							intercepted = True
							add_trace(f"⚠️ Exhaustion: Intercepted redundant {tool_name} call. Forcing Text Search.")
						elif tool_name == "get_node_history" and has_history and not is_pagination_request:
							result_data = [{"error": "EXHAUSTION RULE: You have already shown the structured event history for this entity. Do NOT return these. You MUST immediately use `global_text_search` to find deeper unstructured mentions."}]
							intercepted = True
							add_trace(f"⚠️ Exhaustion: Intercepted redundant {tool_name} call. Forcing Text Search.")
					# ------------------------------
					
					if not intercepted:
						if intent == "analytics" and tool_name in {"get_top_entities", "count_entities_by_event"}:
							last_same_tool_args = None
							for previous_message in reversed(messages):
								if previous_message.get("role") != "assistant":
									continue
								for previous_call in previous_message.get("tool_calls", []):
									if previous_call["function"]["name"] != tool_name:
										continue
									try:
										last_same_tool_args = json.loads(previous_call["function"]["arguments"])
									except Exception:
										last_same_tool_args = None
									if last_same_tool_args:
										break
								if last_same_tool_args:
									break

							if last_same_tool_args:
								for locked_key in ["entity_type", "metric", "location", "keyword"]:
									if locked_key in last_same_tool_args and locked_key not in args:
										args[locked_key] = last_same_tool_args[locked_key]
						if tool_name == "global_text_search":
							query_value = str(args.get("query", "")).strip()
							if current_uid and query_value in {str(current_uid), str(args.get("uid", ""))}:
								name_lookup = self.execute_query(
									"MATCH (n:BaseNode {uid: $uid}) RETURN coalesce(n.name, $uid) AS query LIMIT 1",
									{"uid": current_uid},
								)
								if name_lookup:
									args["query"] = name_lookup[0]["query"]
						func = getattr(self, tool_name)
						result_data = func(**args)
					
					if not result_data:
						feedback = build_empty_result_feedback(tool_name, args)
						add_trace(f"⚠️ Empty Result. Sending feedback to Agent.")
						
						messages.append({
							"role": "tool",
							"tool_call_id": tool_call["id"],
							"content": json.dumps({"error": feedback})
						})
					else:
						if "error" in result_data[0]:
							add_trace(f"🛑 Error: {result_data[0]['error']}")
						else:
							if tool_name == "search_companies":
								weak_results = [
									row for row in result_data
									if row.get("is_weak") is True or str(row.get("uid", "")).startswith("weak_")
								]
								if weak_results:
									weak_uid = weak_results[0].get("uid")
									result_data.append({
										"_SYSTEM_INSTRUCTION_": (
											"WEAK NODE RULE: Do NOT stop at this weak match. "
											f"Immediately call `explore_network` with uid='{weak_uid}' "
											"to recover connected real entities."
										)
									})
							if intent == "analytics" and tool_name in {"get_top_entities", "count_entities_by_event"}:
								has_more_rows = any(
									isinstance(row, dict) and row.get("SYSTEM_NOTE") == "🔴 MORE RESULTS AVAILABLE 🔴"
									for row in result_data
								)
								signature = (
									tool_name,
									args.get("entity_type"),
									args.get("metric"),
									args.get("event_type"),
									args.get("location"),
									args.get("keyword"),
								)
								if not has_more_rows:
									exhausted_analytics_signatures.add(signature)
							final_data.extend(result_data) 
							add_trace(f"✅ Result: Found {len(result_data)} records.")
							
							# Programmatic State Machine Nudge for Dossier Mode
							if current_uid and tool_name in ["explore_network", "get_node_history"]:
								has_network = tool_name == "explore_network"
								has_history = tool_name == "get_node_history"
								
								# Check current turn's tool calls
								for m in messages:
									if m.get("role") == "assistant" and m.get("tool_calls"):
										for tc in m["tool_calls"]:
											if tc["function"]["name"] == "explore_network": has_network = True
											elif tc["function"]["name"] == "get_node_history": has_history = True
								
								# Infer prior state from the user's chat history intents
								# If the user previously asked for network/history, we assume it was executed.
								user_messages = [m.get("content", "").lower() for m in messages if m.get("role") == "user"]
								# Exclude the very last message because that's the current query being handled
								prior_user_messages = user_messages[:-1] if len(user_messages) > 1 else []
								
								for txt in prior_user_messages:
									if any(k in txt for k in ["network", "connect", "involv", "other compani", "relation", "tie"]):
										has_network = True
									if any(k in txt for k in ["history", "event", "past", "chronolog"]):
										has_history = True
								
								if has_network and has_history:
									nudge = "We have explored the structured data. Would you like me to run a deep text search across all unstructured legal publications for any hidden mentions?"
								elif has_network:
									nudge = "Would you now like to see the event history for this entity?"
								elif has_history:
									nudge = "Would you now like to explore the network connections for this entity?"
									
								result_data.append({
									"_SYSTEM_INSTRUCTION_": f"STATE MACHINE OVERRIDE: You MUST end your text response to the user with EXACTLY this question: '{nudge}'"
								})
							
						messages.append({
							"role": "tool", 
							"tool_call_id": tool_call["id"], 
							"content": json.dumps(result_data, default=str)
						})
						
				except Exception as e:
					messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps({"error": str(e)})})

		# If we exhausted the tool loop right after a tool result, force one final
		# synthesis pass without tools so the user sees a formatted answer instead
		# of raw JSON.
		if messages and messages[-1].get("role") == "tool":
			synthesis_msg = self._call_llm(messages, tools=None)
			messages.append(synthesis_msg)

		formatted_trace = "\n".join(execution_trace) if execution_trace else "No tools were needed."
		final_answer = messages[-1].get("content") or "Search completed."

		new_messages_this_turn = messages[start_of_new_messages_idx:]

		return {
			"answer": final_answer,
			"cypher": formatted_trace, 
			"data": final_data,
			"new_messages": new_messages_this_turn,
			"messages": messages
		}
