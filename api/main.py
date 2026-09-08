"""FastAPI endpoints used by the human-in-the-loop dashboard."""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.database import db

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Agentic GraphRAG API")

ACTIVE_TRACES = {}

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
def shutdown_event():
    db.close()


@app.get("/health")
def health():
    """Return a lightweight service-health response."""
    return {"status": "ok"}

@app.get("/api/search")
def search(term: str):
    q_search = """
    MATCH (n:BaseNode) 
    WHERE toLower(n.name) CONTAINS toLower($term)
    WITH n, [l in labels(n) WHERE l <> 'BaseNode'][0] AS real_label
    
    // Grouping by name and city to deduplicate
    WITH n.name AS name, n.city AS city, real_label, collect(n.uid)[0] AS uid, max(n.risk_rank) AS risk
    
    // Scoring logic to put exact matches and Companies first
    WITH uid, name, city, real_label, risk,
         CASE WHEN toLower(name) = toLower($term) THEN 100 ELSE 1 END +
         CASE WHEN real_label = 'Company' THEN 50 ELSE 0 END AS score
         
    RETURN uid, name, real_label, risk, city
    ORDER BY score DESC LIMIT 20
    """
    results = db.run_query(q_search, {'term': term})
    return {"results": results}

@app.get("/api/entity/{uid}")
def get_entity_details(uid: str):
    q_details = """
    MATCH (n:BaseNode {uid: $uid})
    RETURN n.name AS name, n.city AS City, n.risk_rank AS Risk, n.community_id AS Community,
           n.address AS address, n.legal_form AS legal_form, n.deletion_date AS deletion_date, 
           n.purpose AS purpose, n.capital_nominal AS capital_nominal, 
           n.capital_paid AS capital_paid, n.is_weak AS is_weak, n.source AS source
    """
    results = db.run_query(q_details, {'uid': uid})
    if not results:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return {"data": results[0]}

@app.get("/api/entity/{uid}/network")
def get_entity_network(uid: str):
    q_net = """
    MATCH (n:BaseNode {uid: $uid})-[r]-(neighbor)
    RETURN 
        type(r) AS Relation,
        CASE 
            WHEN 'Event' IN labels(neighbor) THEN '📅 ' + coalesce(neighbor.date, '') + ' (' + coalesce(neighbor.rubric, 'Event') + ')'
            ELSE neighbor.name 
        END AS Neighbor,
        [l in labels(neighbor) WHERE l <> 'BaseNode'][0] AS Type
    LIMIT 50
    """
    neighbors = db.run_query(q_net, {'uid': uid})
    return {"network": neighbors}

@app.get("/api/entity/{uid}/people")
def get_entity_people(uid: str):
    q_people = """
    MATCH (c:BaseNode {uid: $uid})
    MATCH (c)-[]-(e:Event)-[]-(p:Person)
    RETURN DISTINCT 
        p.name as Name, 
        p.origin as Origin, 
        e.date as Since, 
        e.rubric as Type
    ORDER BY Since DESC
    """
    people_data = db.run_query(q_people, {'uid': uid})
    return {"people": people_data}

@app.get("/api/entity/{uid}/events")
def get_entity_events(uid: str):
    q_events = """
    MATCH (n:BaseNode {uid: $uid})
    OPTIONAL MATCH (n)-[]-(direct_e:Event)
    WITH n, collect(DISTINCT direct_e) AS direct_events
    OPTIONAL MATCH (n)-[:HAS_NAME]-(:NameHub)-[:HAS_NAME]-(alias:BaseNode)-[]-(indirect_e:Event)
    WITH direct_events + collect(DISTINCT indirect_e) AS events
    UNWIND events AS e
    WHERE e IS NOT NULL
    RETURN DISTINCT e.date AS Date, e.rubric AS Rubric, e.text AS Text, e.uid AS ID
    ORDER BY e.date DESC LIMIT 50
    """
    events = db.run_query(q_events, {'uid': uid})
    return {"events": events}

class ChatRequest(BaseModel):
    question: str
    uid: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
    trace_id: Optional[str] = None

@app.post("/api/chat")
def chat(payload: ChatRequest):
    history = payload.chat_history or []
    trace_id = payload.trace_id
    
    if trace_id:
        ACTIVE_TRACES[trace_id] = []
        
    def trace_callback(msg: str):
        if trace_id:
            ACTIVE_TRACES[trace_id].append(msg)
            
    try:
        result = db.investigator.ask(
            user_question=payload.question, 
            current_uid=payload.uid, 
            chat_history=history,
            trace_callback=trace_callback
        )
        # Clean up trace to prevent memory leak
        if trace_id in ACTIVE_TRACES:
            del ACTIVE_TRACES[trace_id]
        return {"result": result}
    except Exception as exc:
        if trace_id in ACTIVE_TRACES:
            del ACTIVE_TRACES[trace_id]
        LOGGER.exception("Agent request failed", exc_info=exc)
        raise HTTPException(status_code=500, detail="Agent request failed.") from exc

@app.get("/api/chat/trace/{trace_id}")
def get_trace(trace_id: str):
    return {"trace": ACTIVE_TRACES.get(trace_id, [])}

@app.get("/api/entity/{uid}/graph")
def get_entity_graph(uid: str):
    q_graph = """
    MATCH (startNode:BaseNode {uid: $uid})

    // STEP 1: Identity Resolution (Anchor -> NameHub -> Aliases)
    OPTIONAL MATCH identity_path = (startNode)-[:HAS_NAME*1..2]-(alias:BaseNode)

    WITH startNode, 
         collect(identity_path) AS identity_paths, 
         collect(DISTINCT alias) + startNode AS clusterNodes

    UNWIND clusterNodes AS actor
    WITH actor, identity_paths, startNode
    WHERE actor IS NOT NULL

    // STEP 2: Activity Traversal (Actor -> Event -> Target)
    OPTIONAL MATCH activity_path = (actor)-[:HAS_EVENT|ACTED_IN|INVOLVED_IN|HEAD_OFFICE_OF|DISSOLVED_IN]-(e:Event)-[]-(target:BaseNode)
    WHERE actor <> target 
      AND NOT target:NameHub

    WITH startNode, identity_paths, collect(activity_path) AS activity_paths
    WITH startNode, identity_paths + activity_paths AS all_paths
    
    RETURN 
      {
        id: startNode.uid, 
        name: coalesce(startNode.name, startNode.rubric, 'Unknown'), 
        labels: labels(startNode)
      } AS anchor,
      [p IN all_paths WHERE p IS NOT NULL | 
        [n IN nodes(p) | {id: n.uid, name: coalesce(n.name, n.rubric, 'Unknown'), labels: labels(n)}]
      ] AS paths_nodes,
      [p IN all_paths WHERE p IS NOT NULL | 
        [r IN relationships(p) | {source: startNode(r).uid, target: endNode(r).uid, type: type(r)}]
      ] AS paths_links
    """
    results = db.run_query(q_graph, {'uid': uid})
    
    if not results:
        return {"nodes": [], "links": []}

    row = results[0]
    
    nodes_map = {}
    links_list = []
    
    # Process the anchor node always
    anchor = row.get('anchor')
    if anchor and anchor.get('id'):
        nodes_map[anchor['id']] = anchor

    # Flatten nodes
    for path_nodes in row.get('paths_nodes', []):
        for node in path_nodes:
            if not node or not node.get('id'): continue
            nodes_map[node['id']] = dict(node)
            
    # Flatten links
    for path_links in row.get('paths_links', []):
        for link in path_links:
            links_list.append(dict(link))

    # Process labels into group and final real_label
    for node_id, node in nodes_map.items():
        labels = node.get('labels', [])
        real_label = "Event" if "Event" in labels else "BaseNode"
        if "Company" in labels: real_label = "Company"
        elif "Person" in labels: real_label = "Person"
        elif "NameHub" in labels: real_label = "NameHub"
            
        node['label'] = real_label
        node['group'] = 1 if real_label == "Company" else (2 if real_label == "Person" else (3 if real_label == "Event" else 4))
        # Keep id and name as they are
        if 'labels' in node:
            del node['labels']

    # Deduplicate links
    unique_links = []
    seen_links = set()
    for link in links_list:
        sig = (link['source'], link['target'], link['type'])
        if sig not in seen_links:
            seen_links.add(sig)
            unique_links.append(link)

    return {
        "nodes": list(nodes_map.values()),
        "links": unique_links
    }
