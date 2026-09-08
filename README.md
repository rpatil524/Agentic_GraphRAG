# Agentic Graph Retrieval-Augmented Generation for Commercial Registry Analysis

This repository contains the data-collection, graph-construction, agent,
dashboard, baseline, and evaluation code for the paper **Agentic Graph
Retrieval-Augmented Generation for Auditable Commercial Registry Analysis**.
The Swiss Official Gazette of Commerce (SHAB) is used as the case study.

The repository includes the three benchmark inputs and the runners needed to
execute the complete architecture and its main benchmark, baseline, and
ablation comparisons. It intentionally excludes raw SHAB downloads, graph
checkpoints, the Neo4j database, Chroma embeddings, generated answers,
evaluation scores, logs, credentials, and post-hoc manual-validation or
statistical-audit tooling.

## Data Source and Terms

The case-study data originate from SHAB/SOGC and the Official Gazettes Portal,
which is operated by the Swiss State Secretariat for Economic Affairs (SECO).
Before downloading or processing data, review and accept the current terms on
the [Official Gazettes Portal](https://www.shab.ch/). The terms reviewed during
this project require regular consultation and renewed consent every 90 days.

This repository and its outputs are not official publications. The
authoritative records are those published on the Official Gazettes Portal with
the SECO electronic signature or stamp. SECO does not guarantee complete or
correct API transfer, so important findings should be verified against the
authoritative records. Users are responsible for applicable data-protection,
retention, attribution, and redistribution requirements. See
[NOTICE.md](NOTICE.md) for the full project notice.

## Repository Layout

```text
.
|-- agenticGraphRAG/          Graph construction, Neo4j ingestion, and agent
|-- api/                      FastAPI dashboard backend
|-- data_collection/          Resumable SHAB API downloader
|-- docs/figures/             Architecture figures
|-- evaluation/
|   |-- baseline_rag/         all-MiniLM-L6-v2/Chroma dense baseline
|   |-- datasets/             The three published benchmark datasets
|   `-- experiments/          Sensitivity, baseline, and ablation runners
|-- frontend/                 Next.js dashboard
|-- scripts/                  Graph checkpoint and database entry points
|-- .env.example              Environment-variable template
|-- NOTICE.md                 SHAB source, terms, and data-use notice
|-- docker-compose.yml        Local Neo4j setup
`-- requirements.txt          Python dependencies
```

## Architecture

### Knowledge-Graph Construction

The graph is built in three phases. Phase 1 deterministically parses the
structured SHAB fields and creates strong `Company`, `Person`, and `Event`
nodes. Multiple publication columns are combined into one coherent event-text
field, and structured relationships connect registered entities to their
events. All downloaded records are considered during this phase, while only
the `HR`, `KK`, and `LS` commercial-registry rubrics are retained for the graph.

Phase 2 enriches selected high-value notices with entities and roles that occur
only in free legal text. The published configuration processes sub-rubrics
`HR01`, `KK02`, `KK03`, `KK06`, `LS01`, and `LS02`. Each selected event is sent
once to `gpt-4o-mini-2024-07-18`, using at most 2,000 characters and a
constrained JSON response. Extracted people and companies are marked as weak
nodes so their probabilistic origin remains explicit.

Phase 3 creates `NameHub` identity anchors from normalized, alphabetically
ordered name tokens. It merges token-order variants and removes an extracted
weak node when a strong node shares both its `NameHub` and its source `Event`.
This reproduces the weak-node resolution rule used to construct the evaluated
graph.

![Knowledge-graph construction pipeline](docs/figures/data_pipeline.png)

### Analytical Agent

The analytical agent separates routing, retrieval, and synthesis. First, one
LLM call classifies the request and exposes only the relevant graph tools.
Second, a bounded reflection loop performs up to four planning/tool iterations.
Each iteration can make one LLM planning call, execute a read-only semantic
tool, and feed the result or a deterministic error message back to the agent.
The available tools cover entity search, network traversal, event history,
server-side analytics, lexical full-text fallback, and guarded custom Cypher.

Finally, one synthesis call converts the collected evidence into the response.
A state machine records the active entity and completed exploration stage, then
constrains valid follow-up actions. The dashboard displays the final answer,
retrieved evidence, entity dossiers, graph neighborhoods, event histories, and
execution traces so that users can inspect how the answer was produced.

![Analytical-agent architecture](docs/figures/agent_architecture.png)

## Requirements

- Python 3.9 or later
- Node.js 20 or later
- Docker Desktop with Docker Compose, or an existing Neo4j 5 installation
- An OpenAI API key for weak-node extraction, agent execution, answer
  generation, benchmark generation, and LLM judging
- Sufficient local storage for raw publications, graph checkpoints, Neo4j, and
  the optional Chroma index

The retrieval-only dense sensitivity test and local Chroma embedding creation
do not require OpenAI. Neo4j Graph Data Science is needed only for the optional
PageRank and Louvain ingestion stage.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd frontend
npm ci
cd ..
```

Create a local configuration file and edit the empty secret values:

```bash
cp .env.example .env
```

Load it into every terminal used for Python commands:

```bash
set -a
source .env
set +a
```

At minimum, set:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-neo4j-password"
export NEO4J_DATABASE="shabdb"
```

Never commit `.env`, API keys, database passwords, or anonymization keys.

## Start Neo4j

For a fresh local database, set `NEO4J_PASSWORD` and start the supplied Compose
service:

```bash
docker compose up -d neo4j
docker compose logs -f neo4j
```

The container creates `shabdb` as its initial default database and exposes the
Neo4j Browser at <http://localhost:7474>. The initial-database setting applies
when the Docker volume is first created. If using Neo4j Desktop or an existing
server, create or select a database named `shabdb` and update `.env` to match.

Verify the connection without modifying the graph:

```bash
python - <<'PY'
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)
with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
    print(session.run("RETURN 1 AS ok").single()["ok"])
driver.close()
PY
```

## Collect SHAB Data

The downloader queries the public SHAB API week by week, saves resumable raw
JSON pages, and creates monthly CSV files. Review and accept the current SHAB
terms before collecting data. The environment confirmation below expires after
90 days. It does not replace acceptance required by the Official Gazettes
Portal.

```bash
export SHAB_TERMS_ACCEPTED="true"
export SHAB_TERMS_ACCEPTED_DATE="$(date +%F)"
export SHAB_START_DATE="2018-09-02"
export SHAB_END_DATE="2025-12-31"
export SHAB_OUTPUT_DIR="shab_data"
export SHAB_DELAY_SEC="1"
export SHAB_MAX_RETRIES="3"
python data_collection/download_data.py
```

Pagination continues until the API returns an empty page; it is not capped at a
fixed number of pages. Failed requests are retried with exponential backoff.

## Build and Ingest the Knowledge Graph

Convert each monthly CSV folder into a resumable graph checkpoint. This stage
runs both deterministic ingestion and OpenAI-based weak-node extraction.

```bash
export INPUT_GLOB="shab_data/*"
export OUTPUT_DIR="processed_data/processed_graph_data_archive"
export MAX_WORKERS="4"
python scripts/run_job_parallel.py
```

For a small pipeline check, set `ROW_LIMIT` before running the command. Remove
the variable for the complete build.

```bash
export ROW_LIMIT="100"
```

Ingest the checkpoints, resolve names, and run graph analytics:

```bash
export PROCESSED_GRAPH_DIR="processed_data/processed_graph_data_archive"
python scripts/ingest_db.py
```

Set `SKIP_ANALYTICS=true` if Graph Data Science is unavailable. Ingestion is
resumable because nodes and relationships are merged by stable identifiers, but
the identity-resolution stages can still be computationally expensive.

### Optional Anonymization

The ingestion entry point can replace entity names with deterministic aliases:

```bash
export ANONYMIZE_DATA="true"
export ANONYMIZATION_SECRET_KEY="a-long-random-secret"
export ANONYMIZATION_OUTPUT="processed_data/anonymization_mapping.json"
python scripts/ingest_db.py
```

The mapping file reverses the aliases and is therefore sensitive. It is ignored
by Git and must not be published.

## Run the Dashboard

Start the API from the repository root:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation is available at <http://localhost:8000/docs>. In another
terminal, start the frontend:

```bash
cd frontend
cp .env.example .env.local
npm run dev
```

Open <http://localhost:3000>. Use `CORS_ORIGINS` to allow additional trusted
frontend origins; do not use a wildcard origin with credentials in production.

## Evaluation Datasets

The repository includes the benchmark inputs but not generated results. Some
inputs contain source-derived entity names, facts, or notice excerpts required
to define and audit the questions; they are not official publications:

- `evaluation/datasets/automated_dataset.json`: 300 graph-seeded questions
  covering direct, multi-hop, temporal, and aggregation tasks.
- `evaluation/datasets/golden_dataset.json`: 60 manually curated single-turn
  questions with verified reference answers.
- `evaluation/datasets/conversational_dataset.json`: 14 manually curated source
  conversations. The published Tier 4 protocol selects 10 conversations
  containing 36 turns through the evaluator's documented default ID list.

To regenerate the automated benchmark from a populated graph:

```bash
python evaluation/generate_dataset.py
```

This overwrites the path selected by `AUTOMATED_DATASET_PATH`, which defaults to
`evaluation/datasets/automated_dataset.json`. Benchmark generation uses OpenAI
and read-only Neo4j queries.

## Build the Dense Baseline

The dense baseline embeds every Neo4j `Event.text` document with Chroma's local
`all-MiniLM-L6-v2` embedding function. Building this full-corpus index is a
large, long-running operation. It is resumable and does not require OpenAI.

```bash
python evaluation/baseline_rag/initialize_rag.py
```

Use `--reset` only when intentionally replacing an existing index:

```bash
python evaluation/baseline_rag/initialize_rag.py --reset
```

The index is written to `CHROMA_DB_PATH`, defaults to
`evaluation/baseline_rag/chroma_db`, and is excluded from Git.

## Reproduce the Four Evaluation Tiers

All generated answers, metrics, tables, and logs are written to
`EVAL_OUTPUT_DIR`, which defaults to `evaluation/results`. The directory is
excluded from Git so running an experiment does not modify the public source
tree.

### Tier 1: Graph Components

Evaluate the orthographic consistency of 1,000 sampled `NameHub` nodes using
the paper's 0.7 Levenshtein-ratio threshold or exact sorted-token agreement:

```bash
python evaluation/tier1_entity_resolution.py
```

### Tier 2: Agent Trajectories

Run the agent on the 300-question graph-seeded benchmark and record Search-First
Accuracy, Fallback Activation Rate, Average Reasoning Steps, Query Success Rate,
latency, retrieved context, and complete execution traces:

```bash
python evaluation/tier2_trajectory.py
```

### Tier 3: Single-Turn Answer Quality

Run the Agentic GraphRAG system and dense baseline on both single-turn
benchmarks, then judge them with the common Tier 3 protocol:

```bash
python evaluation/run_tier3.py --dataset both --concurrency 10
```

For development only, use `--dataset automated`, `--dataset human`, or
`--limit`. Do not use limited runs as final paper results.

### Tier 4: Multi-Turn Conversations

Evaluate the full graph system and dense baseline on the curated conversational
benchmark:

```bash
python evaluation/tier4_conversational_eval.py
```

The runner reuses an existing full-graph output in `EVAL_OUTPUT_DIR` when one is
available, then regenerates the dense conversational baseline and judge scores.
Pass `--rerun-graph` when the graph output must also be regenerated.

## Baseline Sensitivity and Ablation Studies

The following commands reproduce the additional comparisons reported in the
paper. They use the 60-question benchmark unless stated otherwise.

Dense retrieval-only top-k sensitivity, with no OpenAI calls:

```bash
python evaluation/experiments/phase2_vector_retrieval_only_topk.py \
  --dataset human --k 5 10 20
```

Full answer-quality top-k sensitivity and lexical/hybrid baselines:

```bash
python evaluation/experiments/phase2_vector_topk_answer_quality.py \
  --k 5 10 20 --dataset human --concurrency 8
python evaluation/experiments/phase2_lexical_answer_quality.py \
  --k 5 10 20 --dataset human --concurrency 8
python evaluation/experiments/phase2_hybrid_answer_quality.py \
  --k 5 10 20 --dataset human --concurrency 8
```

The lexical and hybrid runners require the Neo4j `global_search` full-text
index created by the ingestion setup. Hybrid retrieval combines Chroma dense
results and Neo4j lexical results with reciprocal rank fusion.

Run the three controlled graph ablations after the full Tier 3 human output is
available. The no-router and structured-only summaries also read the
no-reflection comparison file, so run them in this order:

```bash
python evaluation/experiments/phase2_ablation_no_reflection.py \
  --max-iterations 1 --concurrency 8
python evaluation/experiments/phase2_ablation_no_router.py \
  --disable-router --max-iterations 4 --concurrency 8
python evaluation/experiments/phase2_ablation_no_weak_nodes.py \
  --structured-only --max-iterations 4 --concurrency 8
```

Run the five additional architectures on the 300-question graph-seeded
benchmark. The command resumes completed JSONL rows unless `--force` is used:

```bash
python evaluation/experiments/phase2_graph_seeded_missing_architectures.py \
  --architectures lexical hybrid no_reflection no_router structured_only \
  --concurrency 3
```

Run the corresponding additional conversational architectures:

```bash
python evaluation/experiments/phase2_conversational_missing_architectures.py \
  --architectures lexical hybrid no_reflection no_router structured_only \
  --concurrency 8
```

These experiments make paid OpenAI calls and can run for a long time. Start
with each runner's `--help` and, where supported, `--limit` or selected IDs to
validate a local setup before launching a complete reproduction.

## Summarize Generated Results

After the four core tiers have completed:

```bash
python evaluation/compute_stats.py
```

This creates `evaluation/results/evaluation_summary.json` by default. Missing
tier files are reported as unavailable rather than replaced with fabricated
values.

## Reproducibility Notes

- Model snapshots default to those used in the paper and can be overridden by
  the `OPENAI_*_MODEL` variables in `.env.example`.
- LLM outputs may vary across repeated runs even when prompts and model names
  are unchanged. Preserve generated output files when auditing a reproduction.
- OpenAI availability, pricing, and model retention can change. Review current
  API documentation before starting a full run.
- The included evaluation datasets are research artifacts. Raw SHAB notices
  must be obtained from the original source under its current terms of use.
- Post-hoc manual-validation, provenance-audit, and statistical-analysis scripts
  are outside this public release. The included code covers the operational
  architecture and the main benchmark, baseline, and ablation runners.
- The code uses read-only database sessions for agent and dashboard queries.
  Graph mutation is confined to the explicit ingestion pipeline.

## Generated and Private Files

The `.gitignore` excludes:

- `.env` files and credentials;
- raw SHAB downloads and processed graph checkpoints;
- anonymization mappings;
- Neo4j and Chroma data;
- generated answers, evaluation metrics, tables, and logs;
- frontend build outputs and dependency directories;
- Python caches and local editor/OS files.

Only source code, configuration templates, documentation figures, and the three
benchmark input datasets should be committed to the public repository.
