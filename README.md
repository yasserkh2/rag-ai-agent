# Customer Care AI Agent

A LangGraph-based customer care chatbot starter project with a clean OOP structure, small focused nodes, and SOLID-friendly service boundaries.

## Current status

The project is set up as a runnable foundation for a customer support chatbot. It currently includes:

- Query ingestion
- Intent classification
- Graph-based routing
- Knowledge-base answer path with placeholder service output
- Appointment request response path
- Human escalation response path
- Shared chat state and conversation history handling
- Standalone processing layer for ingestion, chunking, and vectorization
- Standalone vector DB layer with Qdrant setup scaffolding
- CLI runner
- Graph PNG export
- OOP and SOLID documentation

This is a strong base for continuing the real implementation of retrieval, slot filling, confirmations, and external integrations.

## Architecture

The chatbot flow is:

1. `ingest_query`
2. `classify_intent`
3. Conditional route to:
   - `kb_answer`
   - `action_request`
   - `human_escalation`
4. `response`

The project separates responsibilities clearly:

- `app/graph/` contains workflow orchestration
- `app/graph/nodes/` contains thin graph node adapters
- `app/services/` contains business logic and reusable application services
- `processing/` contains ingestion, chunking, and vectorization workflows
- `vector_db/` contains vector-database abstractions and vendor-specific infrastructure
- `app/graph/dependencies.py` is the composition root

Because of that separation, you can change the graph structure in one place and change business behavior in another.

## Project structure

```text
app/
  graph/
    __init__.py
    builder.py
    dependencies.py
    router.py
    state.py
    nodes/
      __init__.py
      ingest_query.py
      classify_intent.py
      kb_answer.py
      action_request.py
      human_escalation.py
      response.py
  services/
    __init__.py
    contracts.py
    history.py
    intent.py
    models.py
    responses.py
    router.py
processing/
  __init__.py
  ingestion_pipeline/
    __init__.py
    contracts.py
    faqs.py
    models.py
  chunking/
    __init__.py
    contracts.py
    faqs.py
    models.py
  vectorization/
    __init__.py
    contracts.py
    faqs.py
    providers/
      __init__.py
      factory.py
      gemini.py
      local.py
      openai.py
    models.py
vector_db/
  __init__.py
  contracts.py
  models.py
  ARCHITECTURE.md
  qdrant/
    __init__.py
    setup.py
    docker-compose.yml
    README.md
    DECISION.md
    SETUP_SPECS.md
scripts/
  run_cli_chat.py
  setup_qdrant.py
  export_graph_png.py
README.md
DEVELOPMENT_DECISIONS.md
INTERFACE_DECISIONS.md
RAG_IMPLEMENTATION_PLAN.md
OOP_SOLID_PRINCIPLES.md
Customer Care AI Chatbot Agent Development Task-v2 (1) (1).md
pyproject.toml
graph.png
```

## Key concepts

### Nodes are thin

Each graph node should do as little as possible. Its job is to receive `ChatState`, call the correct dependency, and return the state update.

### Services hold business logic

Classification rules, routing rules, response generation, and history management live in services, not inside the graph builder.

### Vector DB code is separate from app services

Vector database interfaces and vendor-specific setup do not live in `app/services/`.

Instead:

- `app/services/` owns application behavior
- `vector_db/` owns vector-database contracts and infrastructure
- `vector_db/qdrant/` owns the current Qdrant implementation

This keeps the architecture more OOP-oriented and makes future replacement easier.

### Processing code is separate from vector storage

Ingestion, chunking, and vectorization code does not live inside `vector_db/`.

Instead:

- `processing/` owns ingestion, chunking, and vectorization workflows
- `vector_db/` owns vector persistence contracts and vector-layer models

This keeps parsing, chunking, and embedding preparation independent from the chosen vector backend.

### Dependencies are injected

`build_graph()` wires concrete implementations through `GraphDependencies`, which makes replacement and testing easier.

## Current implemented behavior

### Intent classification

The current classifier is keyword-based and supports:

- `kb_query`
- `action_request`
- `human_escalation`

### Routing

The router sends the conversation to:

- `kb_answer` for knowledge-base-style questions
- `action_request` for appointment-related requests
- `human_escalation` for human help requests or frustration signals

### Response generation

The knowledge-base path is still a placeholder service and does not yet perform true retrieval or generation from the mock KB.

The action request path currently returns a guided follow-up asking for service, date, and time.

The escalation path returns a human handoff message with a reason when available.

### Vector DB setup

The repository now includes a standalone vector DB layer and Qdrant setup scaffolding.

Current status:

- generic vector DB interface lives in `vector_db/contracts.py`
- setup result model lives in `vector_db/models.py`
- Qdrant setup implementation lives in `vector_db/qdrant/setup.py`
- setup script lives in `scripts/setup_qdrant.py`

This means the project is now prepared for a real retrieval implementation without mixing vector infrastructure into the app service layer.

### Processing layer

The repository now also includes a standalone processing layer for retrieval preparation code.

Current status:

- ingestion pipeline contract lives in `processing/ingestion_pipeline/contracts.py`
- chunking contract lives in `processing/chunking/contracts.py`
- vectorization contract lives in `processing/vectorization/contracts.py`
- FAQ-specific implementations now exist for ingestion, chunking, and vectorization
- processing contracts use `ABC` because these layers are expected to become explicit reusable workflows

This means FAQ, document, and structured-data workflows can be built without coupling parsing, chunking, or vector preparation logic to Qdrant or any future vector backend.

### Embedding providers

The embedding layer now supports provider-based implementations through `processing/vectorization/providers/`.

Current providers:

- `gemini` for Google AI Studio embeddings
- `openai` for OpenAI embeddings
- `local` for deterministic pipeline-only testing

The vectorization pipeline embeds stored FAQ chunks as documents, while retrieval embeds incoming user questions as queries. This matters for providers such as Gemini that support retrieval-specific task types.

## Setup

Use the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set the provider you want in `.env`:

```bash
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
# Latest stable Gemini embedding model
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
QDRANT_EMBEDDING_DIMENSION=1536
```

## Qdrant setup

You can start Qdrant in one of two ways.

### Embedded local mode

This mode uses the Python client with persisted local storage:

```bash
.venv/bin/python scripts/setup_qdrant.py
```

Default storage path:

```text
vector_db/qdrant/data/local
```

### Docker mode

If you prefer running Qdrant as a separate service:

```bash
docker compose -f vector_db/qdrant/docker-compose.yml up -d qdrant
export QDRANT_URL=http://localhost:6333
.venv/bin/python scripts/setup_qdrant.py
```

Standalone Qdrant docs live in:

- `vector_db/ARCHITECTURE.md`
- `vector_db/qdrant/README.md`
- `vector_db/qdrant/DECISION.md`
- `vector_db/qdrant/SETUP_SPECS.md`

## Run locally

Run the CLI chatbot:

```bash
.venv/bin/python scripts/run_cli_chat.py
```

## FAQ Pipeline Commands

Run the full FAQ processing pipeline with your `.env` settings:

```bash
.venv/bin/python scripts/run_faq_processing_pipeline.py
```

Run a small Gemini experiment with only 20 FAQ records:

```bash
EMBEDDING_PROVIDER=gemini FAQ_PIPELINE_LIMIT=20 QDRANT_PATH=vector_db/qdrant/data/experiment_gemini_20 .venv/bin/python scripts/run_faq_processing_pipeline.py
```

Run a small OpenAI experiment with only 20 FAQ records:

```bash
EMBEDDING_PROVIDER=openai FAQ_PIPELINE_LIMIT=20 QDRANT_PATH=vector_db/qdrant/data/experiment_openai_20 .venv/bin/python scripts/run_faq_processing_pipeline.py
```

Run a small local-provider experiment for pipeline-only testing:

```bash
EMBEDDING_PROVIDER=local FAQ_PIPELINE_LIMIT=20 QDRANT_PATH=vector_db/qdrant/data/experiment_faq_20 .venv/bin/python scripts/run_faq_processing_pipeline.py
```

Test retrieval against a stored experiment:

```bash
EMBEDDING_PROVIDER=gemini QDRANT_PATH=vector_db/qdrant/data/experiment_gemini_20 FAQ_RETRIEVAL_QUERY="What does credentialing include?" FAQ_RETRIEVAL_LIMIT=3 .venv/bin/python scripts/test_faq_retrieval.py
```

Inspect saved vector records from Qdrant:

```bash
QDRANT_PATH=vector_db/qdrant/data/experiment_faq_20 VECTOR_INSPECT_LIMIT=3 .venv/bin/python scripts/inspect_qdrant_vectors.py
```

Inspect saved records and include vector values:

```bash
QDRANT_PATH=vector_db/qdrant/data/experiment_faq_20 VECTOR_INSPECT_LIMIT=1 VECTOR_INSPECT_WITH_VECTORS=true .venv/bin/python scripts/inspect_qdrant_vectors.py
```

Export the graph PNG:

```bash
.venv/bin/python scripts/export_graph_png.py
```

Key environment variables:

- `EMBEDDING_PROVIDER`
- `FAQS_JSONL_PATH`
- `FAQ_PIPELINE_LIMIT`
- `FAQ_PIPELINE_BATCH_SIZE`
- `QDRANT_PATH`
- `QDRANT_COLLECTION`
- `QDRANT_EMBEDDING_DIMENSION`
- `GEMINI_API_KEY`
- `GEMINI_EMBEDDING_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`

## Example interaction

```text
You: I need to book an appointment
Bot: I can help with an appointment request. Please share the service you need, plus your preferred date and time.
```

## Documentation

- `README.md`: project overview and usage
- `DEVELOPMENT_DECISIONS.md`: architecture and implementation decisions
- `INTERFACE_DECISIONS.md`: contract-style and package-boundary decisions
- `RAG_IMPLEMENTATION_PLAN.md`: initial RAG starting point and rollout plan
- `OOP_SOLID_PRINCIPLES.md`: OOP and SOLID mapping to the codebase
- `Customer Care AI Chatbot Agent Development Task-v2 (1) (1).md`: original task brief with implementation tracking notes
- `vector_db/ARCHITECTURE.md`: vector DB layer placement and interface design
- `vector_db/qdrant/README.md`: Qdrant standalone setup guide
- `vector_db/qdrant/DECISION.md`: why Qdrant is the preferred vector DB for this task
- `vector_db/qdrant/SETUP_SPECS.md`: local and production setup expectations

## Next development steps

- Replace the placeholder KB service with real retrieval over the mock KB
- Add provider-specific retry and error handling for embedding failures
- Add retrieval-quality checks for Gemini/OpenAI experiment sets
- Expand ingestion beyond FAQs into documents and structured data
- Add entity extraction for appointment requests
- Add slot filling and confirmation flow
- Add a mock or real booking integration
- Add failure counting and clearer escalation policies
- Add automated tests for nodes, services, and routing
- Add an API or UI layer
