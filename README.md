# Customer Care AI Agent

A LangGraph-based customer care chatbot starter project with a clean OOP structure, small focused nodes, and SOLID-friendly service boundaries.

## Current status

The project is set up as a runnable foundation for a customer support chatbot. It currently includes:

- Query ingestion
- LLM-based intent classification with static safe fallback when the provider is unavailable
- Graph-based routing
- General conversation path for greetings, thanks, and vague helper turns
- Knowledge-base answer path with retrieval plus grounded RAG answering
- Multi-turn appointment action agent with mock external integration
- Mid-conversation human escalation routing with sticky handoff state
- YAML-based runtime config through `config.yml` with `.env` overrides
- Gemini and Azure OpenAI support for KB answers, action replies, action extraction, intent classification, and escalation replies
- OpenAI provider path exists in code but is not fully validated for end-to-end runtime use in this project
- Human escalation response path with LLM-first generation and safe template fallback
- Shared chat state and conversation history handling
- Session memory in the CLI chat loop
- Standalone processing layer for ingestion, chunking, and vectorization
- Standalone vector DB layer with Qdrant setup scaffolding
- CLI runner
- Graph PNG export
- OOP and SOLID documentation

This is a strong base for continuing retrieval, state-safe action flows, confirmations, and external integrations.

## Recent updates

- KB warmup runs once at graph bootstrap to reduce first-turn latency.
- KB retrieval ranking now uses cosine similarity scores only (no lexical bonus, contact boost, or reranker path in `RetrievalKnowledgeBaseService`).
- KB retrieval now embeds the raw user query directly (no retrieval-query rewrite in the active KB retrieval path).
- Document chunk text now carries stronger semantic metadata for embedding:
  - normalized `Keywords`
  - `Keyword Terms`, `Keyword Tokens`, and `Keyword Query Hints`
  - document identity fields (`Document ID`, `Document File`)
- Streamlit renders the assistant response as soon as `final_response` is produced during graph streaming.
- Mock booking integration now uses a JSON-backed calendar store at `data/booking_store.json` with slot states (`free` or `booked`) and booking persistence.
- Mock booking API now supports create, fetch, and delete booking operations, and availability is computed from stored slot state.

## Architecture

The chatbot flow is:

1. `ingest_query`
2. `classify_intent`
3. Conditional route to:
   - `general_conversation`
   - `kb_answer`
   - `action_request`
   - `human_escalation`
4. `evaluate_escalation` after unresolved `kb_answer`/`action_request` turns
5. `human_escalation` or `response` (resolved service turns can route directly to `response`)

The project separates responsibilities clearly:

- `app/graph/` contains workflow orchestration
- `app/graph/nodes/` contains thin graph node adapters
- `app/agents/` contains graph-executable agents that wrap conversational behaviors
- `app/services/` contains business logic and reusable application services
- `app/llm/` contains prompt building and model-provider generation logic
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
      general_conversation.py
      kb_answer.py
      action_request.py
      evaluate_escalation.py
      human_escalation.py
      response.py
  agents/
    __init__.py
    contracts.py
    models.py
    kb_agent.py
    action_agent.py
    general_conversation_agent.py
    escalation_agent.py
    factory.py
  llm/
    __init__.py
    action_extraction.py
    action_planning.py
    escalation_factory.py
    escalation_prompts.py
    intent_factory.py
    intent_prompts.py
    contracts.py
    factory.py
    http.py
    prompts.py
    providers/
      __init__.py
      azure_openai.py
      gemini.py
      openai.py
  services/
    __init__.py
    action_models.py
    action_request.py
    booking_api.py
    contracts.py
    history.py
    intent.py
    knowledge_base.py
    models.py
    responses.py
    router.py
  mock_api/
    __init__.py
    booking_api.py
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
data/
  booking_store.json
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

### Agents are reusable execution units

The `app/agents/` package provides graph-ready agents such as:

- `KnowledgeBaseAgent`
- `ActionRequestAgent`
- `GeneralConversationAgent`
- `HumanEscalationAgent`

This creates a cleaner expansion path for future agents while keeping graph nodes small.

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

The default classifier is now LLM-based and supports:

- `general_conversation`
- `kb_query`
- `action_request`
- `human_escalation`

Current behavior:

- the classifier uses a dedicated intent-classification prompt
- it considers the latest user message, recent conversation history, active action state, and current failure count
- it can route greetings, thanks, and capability/help turns into `general_conversation`
- it returns structured JSON for:
  - `intent`
  - `confidence`
  - `frustration_flag`
  - `escalation_reason`
- if the configured LLM provider is unavailable or rate-limited, the app falls back to a static safe decision (`kb_query` with low confidence) so the graph remains resilient

### Routing

The router sends the conversation to:

- `general_conversation` for greetings, thanks, broad capability questions, and vague conversational turns that should get a short guided reply
- `kb_answer` for knowledge-base-style questions
- `action_request` for appointment-related requests
- `human_escalation` for human help requests, frustration signals, or handoff-pending sessions

After `kb_answer` and `action_request`, the graph now runs a post-turn escalation evaluation step that can:

- increment repeated-failure state
- trigger escalation immediately when a turn already signals handoff
- route the current turn into `human_escalation` before the normal response path finishes

### Response generation

The general conversation path returns short conversational guidance for turns like greetings, thanks, or “what can you do?” without forcing a retrieval or action workflow. This keeps the experience natural while preserving a clean graph separation between routing and user-facing replies.

The knowledge-base path now retrieves from both FAQ and document collections (parallel search) and uses grounded generation to produce the final answer. If generation fails, the app returns an explicit unresolved message instead of extractive fallback content.

Retrieval ranking details:

- cosine-only ordering by vector score
- no active reranker in KB retrieval
- no active retrieval-query rewrite in KB retrieval
- `retrieval_query` currently mirrors the raw user query

The action request path is now a real multi-turn appointment agent. It collects booking fields across turns, validates service/date/time/name/email state in code, asks for one missing field at a time, lets the LLM phrase the reply naturally, asks for confirmation only when the booking is complete, and then submits a mock booking request.

The escalation path now uses an LLM-generated handoff reply when a provider is available. If generation fails or the provider is not configured, it falls back to a safe template message. It also clears active automated appointment state and keeps the session in handoff mode through `handoff_pending`.

### Human escalation flow

The escalation flow now works across the conversation, not only on a single turn.

Current behavior:

- explicit escalation requests can route directly to `human_escalation`
- KB and action turns report `turn_outcome` and optional failure metadata
- unresolved turns can increase `failure_count`
- repeated failure triggers escalation when the threshold is reached
- once handoff starts, the graph keeps routing the session to `human_escalation`

Key state fields used for this flow now include:

- `handoff_pending`
- `failure_count`
- `turn_outcome`
- `turn_failure_reason`
- `frustration_flag`
- `escalation_reason`

### Vector DB setup

The repository now includes a standalone vector DB layer and Qdrant setup scaffolding.

Current status:

- generic vector DB interface lives in `vector_db/contracts.py`
- setup result model lives in `vector_db/models.py`
- Qdrant setup implementation lives in `vector_db/qdrant/setup.py`
- setup script lives in `scripts/setup_qdrant.py`

This vector layer now supports the retrieval-backed `kb_answer` path without mixing vector infrastructure into the app service layer.

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

Current project default for retrieval is Gemini embeddings (`EMBEDDING_PROVIDER=gemini` with `GEMINI_EMBEDDING_MODEL=gemini-embedding-001`).

The vectorization pipeline embeds stored FAQ chunks as documents, while retrieval embeds incoming user questions as queries. This matters for providers such as Gemini that support retrieval-specific task types.

### Action agent

The appointment flow is now implemented as a real stateful action agent.

Current behavior:

- extracts booking fields with Gemini, OpenAI, or Azure OpenAI through `app/llm/action_extraction.py`
- stores slots in `ChatState`
- proactively fetches available dates when the service is known and the date is still missing
- asks for one missing field at a time across turns
- fetches available dates and time slots through a local mock HTTP API
- validates malformed slot values such as incomplete email addresses
- confirms the final booking details only after all required fields are complete and date/time are confirmed
- creates a mock booking with a confirmation id
- persists bookings and slot state in a local JSON calendar store

The action service is intentionally split this way:

- the LLM handles structured extraction and natural-language replies
- the service layer validates slot state and controls when the flow can advance
- confirmation is LLM-driven through `confirmation_intent`
- the action service does not use regex-based confirmation shortcuts

The external integration boundary is intentionally separated:

- `app/services/booking_api.py` is the client
- `app/mock_api/booking_api.py` is the local mock endpoint
- `data/booking_store.json` is the persisted local booking store

This keeps the graph and agent logic independent from the transport details and gives a clean replacement path for a future real calendar or booking API.

Mock booking API summary:

- `GET /available-dates`
- `GET /availability`
- `POST /bookings`
- `GET /bookings/{confirmation_id}`
- `DELETE /bookings/{confirmation_id}`

Default local seeded calendar window:

- start date: `2026-04-15`
- duration: `31` days
- slot frequency: every `30` minutes from `09:00 AM` to `05:00 PM`

### Current FAQ dataset

For current KB testing, the repo now includes a curated higher-quality FAQ set:

- `cob_mock_kb_large/high_quality_faqs/high_quality_faqs.jsonl`

## Full run instructions (start here)

Use this section if you want one complete path from setup to a working chatbot session.

### 1) Prerequisites

- Python `3.11+`
- `pip`
- Recommended: Azure OpenAI key (for chat-generation flows) + Gemini key (for embeddings)

### 2) Install dependencies

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 3) Configure runtime settings

Create local config files:

```bash
cp .env.example .env
cp config.yml.example config.yml
```

Set secrets in `.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
# Optional for hosted Qdrant:
QDRANT_API_KEY=
```

Set non-secret runtime settings in `config.yml`:

```yaml
embedding_provider: gemini
kb_answer_provider: azure_openai
action_agent_provider: azure_openai
action_extraction_provider: azure_openai
intent_classifier_provider: azure_openai
escalation_agent_provider: azure_openai

azure_openai:
  endpoint: https://your-resource-name.openai.azure.com
  chat_deployment: your_gpt_4_1_deployment_name
  api_version: 2024-02-01

gemini:
  embedding_model: gemini-embedding-001
  chat_model: gemini-2.5-flash
  retrieval_query_model: gemini-2.5-flash-lite
```

Preferred production-style setup:

- Gemini embeddings for ingestion + retrieval
- Azure OpenAI for KB answers, action replies/extraction, intent classification, and escalation replies
- Gemini and Azure OpenAI are the recommended model providers for this project
- OpenAI provider path is currently not fully validated end-to-end

Alternative fully-Gemini setup in `config.yml` (supported across the project):

```yaml
embedding_provider: gemini
kb_answer_provider: gemini
action_agent_provider: gemini
action_extraction_provider: gemini
intent_classifier_provider: gemini
escalation_agent_provider: gemini

gemini:
  embedding_model: gemini-embedding-001
  chat_model: gemini-2.5-flash
  retrieval_query_model: gemini-2.5-flash-lite
```

Config precedence at runtime:

- real shell environment variables
- `.env`
- `config.yml`

Recommended convention:

- `.env` contains secrets only
- `config.yml` contains non-secret settings

### 4) Initialize vector storage and index the demo KB

This step is required for grounded KB answers.

```bash
.venv/bin/python scripts/setup_qdrant.py
.venv/bin/python scripts/run_faq_processing_pipeline.py
.venv/bin/python scripts/run_document_processing_pipeline.py
```

Recommended defaults in `config.yml` for full local data indexing:

```yaml
faqs_jsonl_path: data/faqs/high_quality_faqs.jsonl
documents_manifest_path: data/documents/documents_manifest.json
```

You do not need to set ingestion paths in `.env` unless you want a temporary override.

### 5) Run the app

CLI chatbot:

```bash
.venv/bin/python scripts/run_cli_chat.py
```

Streamlit UI:

```bash
.venv/bin/python -m streamlit run ui/streamlit_app.py
```

No separate booking API server is needed; the local mock booking API starts automatically in-process.

### 6) Optional verification

Run tests:

```bash
.venv/bin/python -m pytest
```

Export the current graph image:

```bash
.venv/bin/python scripts/export_graph_png.py
```

## Documentation

Helpful repo docs:

- [KB Agent Walkthrough](KB_AGENT_WALKTHROUGH.md)
- [Action Agent Walkthrough](ACTION_AGENT_WALKTHROUGH.md)
- [Agent Interface Decision](AGENT_INTERFACE_DECISION.md)
- [OOP and SOLID Principles](OOP_SOLID_PRINCIPLES.md)
- [Document RAG Implementation](DOCUMENT_RAG_IMPLEMENTATION.md)
- [Retrieval Query Rewriting](RETRIEVAL_QUERY_REWRITING.md)

This set is intended to be a better retrieval baseline than the older large mixed synthetic FAQ source because it is:

- smaller and easier to inspect
- less repetitive
- more service-specific
- written more like direct customer questions
- better suited for early RAG quality checks

## Setup

Use the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Recommended secrets in `.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
# Optional for hosted Qdrant:
QDRANT_API_KEY=
```

Recommended non-secret provider setup in `config.yml`:

```yaml
embedding_provider: gemini
kb_answer_provider: azure_openai
action_agent_provider: azure_openai
action_extraction_provider: azure_openai
intent_classifier_provider: azure_openai
escalation_agent_provider: azure_openai

azure_openai:
  endpoint: https://your-resource-name.openai.azure.com
  chat_deployment: your_gpt_4_1_deployment_name
  api_version: 2024-02-01

gemini:
  embedding_model: gemini-embedding-001
  chat_model: gemini-2.5-flash
  retrieval_query_model: gemini-2.5-flash-lite
```

All-Gemini mode is also supported:

```yaml
embedding_provider: gemini
kb_answer_provider: gemini
action_agent_provider: gemini
action_extraction_provider: gemini
intent_classifier_provider: gemini
escalation_agent_provider: gemini

gemini:
  chat_model: gemini-2.5-flash
  retrieval_query_model: gemini-2.5-flash-lite
```

You can also use a structured `config.yml` instead of relying only on `.env`.
Copy [config.yml.example](/media/yasser/New%20Volume1/yasser/New_journey/customer-care-ai-agent/config.yml.example) to `config.yml` and edit the values you want.

Config loading precedence is:

- real shell environment variables win
- `.env` overrides `config.yml`
- `config.yml` provides the base defaults

## Qdrant setup

This project uses embedded local Qdrant mode.

### Embedded local mode

This mode uses the Python client with persisted local storage:

```bash
.venv/bin/python scripts/setup_qdrant.py
```

Default storage path:

```text
vector_db/qdrant/data/local
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

The CLI keeps chat history in memory for the current session, so follow-up questions can use prior turns as context.

Run the standalone Streamlit demo UI:

```bash
.venv/bin/python -m streamlit run ui/streamlit_app.py
```

If you run into module import issues, use:

```bash
PYTHONPATH=. .venv/bin/python -m streamlit run ui/streamlit_app.py
```

The UI is useful for interview demos because it shows:

- the assistant reply
- streamed reply rendering (token-style) for better perceived responsiveness
- live node progress while the graph runs (for example `Running: kb_answer`)
- per-turn latency caption (`backend` vs `ui_total`)
- the vector query used for retrieval
- the retrieved chunks for that specific turn
- backend trace logs for routing, retrieval, and escalation decisions

## FAQ Pipeline Commands

Build the interview-ready non-overlapping dataset first:

```bash
.venv/bin/python scripts/build_interview_demo_dataset.py
```

That script creates:

- `cob_mock_kb_large/interview_demo_kb/retrieval/documents`
- `cob_mock_kb_large/interview_demo_kb/retrieval/faqs`
- `cob_mock_kb_large/interview_demo_kb/operations`

Only `retrieval/` should be indexed for normal RAG. `operations/` is for appointment, case, and structured workflow data.

Run the full FAQ processing pipeline with your configured defaults (`config.yml` + optional `.env` overrides):

```bash
.venv/bin/python scripts/run_faq_processing_pipeline.py
```

Run the pipeline against the local data FAQ source explicitly (optional override):

```bash
FAQS_JSONL_PATH=data/faqs/high_quality_faqs.jsonl QDRANT_PATH=vector_db/qdrant/data/local .venv/bin/python scripts/run_faq_processing_pipeline.py
```

Run the document pipeline against the local data manifest explicitly (optional override):

```bash
DOCUMENTS_MANIFEST_PATH=data/documents/documents_manifest.json QDRANT_PATH=vector_db/qdrant/data/local .venv/bin/python scripts/run_document_processing_pipeline.py
```

Recommended clean rebuild flow using all local `data/` sources:

```bash
rm -rf vector_db/qdrant/data/local
mkdir -p vector_db/qdrant/data/local
.venv/bin/python scripts/setup_qdrant.py
FAQS_JSONL_PATH=data/faqs/high_quality_faqs.jsonl QDRANT_PATH=vector_db/qdrant/data/local .venv/bin/python scripts/run_faq_processing_pipeline.py
DOCUMENTS_MANIFEST_PATH=data/documents/documents_manifest.json QDRANT_PATH=vector_db/qdrant/data/local .venv/bin/python scripts/run_document_processing_pipeline.py
```

Why `FAQS_JSONL_PATH` and `DOCUMENTS_MANIFEST_PATH` are shown explicitly above:

- `config.yml` might point to a different dataset
- env vars are useful when you want one-off overrides without editing config files

If you want more visible progress during ingestion, lower the batch size:

```bash
FAQ_PIPELINE_BATCH_SIZE=3 .venv/bin/python scripts/run_faq_processing_pipeline.py
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
- `KB_ANSWER_PROVIDER`
- `INTENT_CLASSIFIER_PROVIDER`
- `FAQS_JSONL_PATH`
- `DOCUMENTS_MANIFEST_PATH`
- `DOCUMENTS_ROOT_PATH`
- `FAQ_PIPELINE_LIMIT`
- `FAQ_PIPELINE_BATCH_SIZE`
- `QDRANT_PATH`
- `QDRANT_COLLECTION`
- `QDRANT_EMBEDDING_DIMENSION`
- `GEMINI_API_KEY`
- `GEMINI_EMBEDDING_MODEL`
- `GEMINI_CHAT_MODEL`
- `GEMINI_MIN_REQUEST_INTERVAL_SECONDS`
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_CHAT_MODEL`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`
- `ACTION_AGENT_PROVIDER`
- `ACTION_EXTRACTION_PROVIDER`
- `INTENT_CLASSIFIER_SYSTEM_PROMPT`
- `ESCALATION_AGENT_PROVIDER`
- `ESCALATION_AGENT_SYSTEM_PROMPT`
- `ESCALATION_AGENT_MAX_OUTPUT_TOKENS`

Important:
- retrieval quality depends on using the same embedding provider for both ingestion and query time
- if you ingest with `EMBEDDING_PROVIDER=local`, you should also chat/query with `EMBEDDING_PROVIDER=local`
- if you ingest with `EMBEDDING_PROVIDER=gemini`, you should also chat/query with `EMBEDDING_PROVIDER=gemini`
- the interview demo defaults now target `cob_mock_kb_large/interview_demo_kb/retrieval/...`
- for the cleanest interview demo, do not index `cob_mock_kb_large/very_large_mixed_kb/documents` and `cob_mock_kb_large/very_large_mixed_kb/faqs` into the same retrieval store as the distilled high-quality corpus

## Example interaction

```text
You: What does credentialing include?
Bot: Credentialing includes gathering provider information, preparing payer enrollment paperwork, tracking submissions, following up with payers, and maintaining provider records when demographic or licensing details change.
```

Action-flow example:

```text
You: i need to book a meeting
Bot: Which service would you like to book?
You: website
Bot: For Digital Marketing and Website Services, available dates include 2026-04-15, 2026-04-16, and 2026-04-17. Which date works for you?
You: 2026-04-17
Bot: For Digital Marketing and Website Services on 2026-04-17, available times include 09:00 AM, 09:30 AM, 10:00 AM, and more. Which time would you like?
You: 10:00 AM
Bot: Great, I have your appointment for Digital Marketing and Website Services on 2026-04-17 at 10:00 AM. Could you please provide your name?
You: yasser khira
Bot: Thanks, Yasser Khira. Could you please provide your email address to complete the booking?
You: yasserkhira@gmail.com
Bot: Please confirm your appointment for Digital Marketing and Website Services on 2026-04-17 at 10:00 AM under the name Yasser Khira with the email yasserkhira@gmail.com. Should I proceed to book it?
You: yes
Bot: Your appointment is confirmed. Confirmation ID: apt_xxxxxxxxxx.
```

Escalation example:

```text
You: i need to escilate
Bot: I can absolutely connect you with a human teammate. Please share your name and either your phone number or email, and we will follow up shortly.
```

Notes:

- escalation replies are LLM-generated when the configured provider is available
- if escalation generation is unavailable, the app automatically falls back to a safe template handoff message
