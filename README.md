# Customer Care AI Agent

A LangGraph-based customer care chatbot starter project with a clean OOP structure, small focused nodes, and SOLID-friendly service boundaries.

## Current status

The project is set up as a runnable foundation for a customer support chatbot. It currently includes:

- Query ingestion
- LLM-based intent classification with deterministic fallback
- Graph-based routing
- General conversation path for greetings, thanks, and vague helper turns
- Knowledge-base answer path with retrieval plus grounded RAG answering
- Multi-turn appointment action agent with mock external integration
- Mid-conversation human escalation routing with sticky handoff state
- YAML-based runtime config through `config.yml` with `.env` overrides
- OpenAI, Gemini, and Azure OpenAI support for KB answers, action replies, action extraction, and intent classification
- Human escalation response path
- Shared chat state and conversation history handling
- Session memory in the CLI chat loop
- Standalone processing layer for ingestion, chunking, and vectorization
- Standalone vector DB layer with Qdrant setup scaffolding
- CLI runner
- Graph PNG export
- OOP and SOLID documentation

This is a strong base for continuing retrieval, state-safe action flows, confirmations, and external integrations.

## Architecture

The chatbot flow is:

1. `ingest_query`
2. `classify_intent`
3. Conditional route to:
   - `general_conversation`
   - `kb_answer`
   - `action_request`
   - `human_escalation`
4. `evaluate_escalation` after `kb_answer` and `action_request`
5. `human_escalation` or `response`

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
- if the configured LLM provider is unavailable or misconfigured, the app falls back to a deterministic keyword classifier so the chat still works

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

The knowledge-base path now retrieves FAQ context from Qdrant and uses a grounded generation prompt to produce the final answer. If generation is not configured or fails, it falls back to the best extractive FAQ answer.

The action request path is now a real multi-turn appointment agent. It collects booking fields across turns, validates service/date/time/name/email state in code, asks for one missing field at a time, lets the LLM phrase the reply naturally, asks for confirmation only when the booking is complete, and then submits a mock booking request.

The escalation path returns a human handoff message with a reason when available, clears active automated appointment state, and keeps the session in handoff mode through `handoff_pending`.

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

The vectorization pipeline embeds stored FAQ chunks as documents, while retrieval embeds incoming user questions as queries. This matters for providers such as Gemini that support retrieval-specific task types.

### Action agent

The appointment flow is now implemented as a real stateful action agent.

Current behavior:

- extracts booking fields with Gemini, OpenAI, or Azure OpenAI through `app/llm/action_extraction.py`
- stores slots in `ChatState`
- proactively fetches available dates when the service is known and the date is still missing
- asks for one missing field at a time across turns
- fetches available time slots through a local mock HTTP API
- validates malformed slot values such as incomplete email addresses
- confirms the final booking details only after all required fields are complete and date/time are confirmed
- creates a mock booking with a confirmation id

The action service is intentionally split this way:

- the LLM handles structured extraction and natural-language replies
- the service layer validates slot state and controls when the flow can advance
- confirmation is LLM-driven through `confirmation_intent`
- the action service does not use regex-based confirmation shortcuts

The external integration boundary is intentionally separated:

- `app/services/booking_api.py` is the client
- `app/mock_api/booking_api.py` is the local mock endpoint

This keeps the graph and agent logic independent from the transport details and gives a clean replacement path for a future real calendar or booking API.

### Current FAQ dataset

For current KB testing, the repo now includes a curated higher-quality FAQ set:

- `cob_mock_kb_large/high_quality_faqs/high_quality_faqs.jsonl`

## Documentation

Helpful repo docs:

- [KB Agent Walkthrough](/media/yasser/New Volume1/yasser/New_journey/customer-care-ai-agent/KB_AGENT_WALKTHROUGH.md)
- [Action Agent Walkthrough](/media/yasser/New Volume1/yasser/New_journey/customer-care-ai-agent/ACTION_AGENT_WALKTHROUGH.md)
- [Agent Interface Decision](/media/yasser/New Volume1/yasser/New_journey/customer-care-ai-agent/AGENT_INTERFACE_DECISION.md)
- [OOP and SOLID Principles](/media/yasser/New Volume1/yasser/New_journey/customer-care-ai-agent/OOP_SOLID_PRINCIPLES.md)
- [Document RAG Implementation](/media/yasser/New Volume1/yasser/New_journey/customer-care-ai-agent/DOCUMENT_RAG_IMPLEMENTATION.md)
- [Retrieval Query Rewriting](/media/yasser/New Volume1/yasser/New_journey/customer-care-ai-agent/RETRIEVAL_QUERY_REWRITING.md)

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
pip install -e .
```

Set the provider you want in `.env`:

```bash
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
# Latest stable Gemini embedding model
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_CHAT_MODEL=gemini-2.5-flash
KB_ANSWER_PROVIDER=gemini
INTENT_CLASSIFIER_PROVIDER=gemini
GEMINI_MIN_REQUEST_INTERVAL_SECONDS=1.0
QDRANT_EMBEDDING_DIMENSION=1536
```

Azure OpenAI is also supported for chat generation:

```bash
KB_ANSWER_PROVIDER=azure_openai
ACTION_AGENT_PROVIDER=azure_openai
ACTION_EXTRACTION_PROVIDER=azure_openai
INTENT_CLASSIFIER_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-01
```

You can also use a structured `config.yml` instead of relying only on `.env`.
Copy [config.yml.example](/media/yasser/New%20Volume1/yasser/New_journey/customer-care-ai-agent/config.yml.example) to `config.yml` and edit the values you want.

Config loading precedence is:

- real shell environment variables win
- `.env` overrides `config.yml`
- `config.yml` provides the base defaults

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

The CLI keeps chat history in memory for the current session, so follow-up questions can use prior turns as context.

Run the standalone Streamlit demo UI:

```bash
.venv/bin/python -m streamlit run ui/streamlit_app.py
```

The UI is useful for interview demos because it shows:

- the assistant reply
- the rewritten vector query used for retrieval
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

Run the full FAQ processing pipeline with your `.env` settings:

```bash
.venv/bin/python scripts/run_faq_processing_pipeline.py
```

Run the pipeline against the smaller high-quality FAQ test set:

```bash
FAQS_JSONL_PATH=cob_mock_kb_large/high_quality_faqs/high_quality_faqs.jsonl QDRANT_PATH=vector_db/qdrant/data/high_quality_faqs .venv/bin/python scripts/run_faq_processing_pipeline.py
```

Run the document pipeline against the interview-ready retrieval corpus:

```bash
.venv/bin/python scripts/run_document_processing_pipeline.py
```

Recommended clean rebuild flow for the interview demo:

```bash
rm -rf vector_db/qdrant/data/local
mkdir -p vector_db/qdrant/data/local
.venv/bin/python scripts/setup_qdrant.py
.venv/bin/python scripts/build_interview_demo_dataset.py
FAQS_JSONL_PATH=cob_mock_kb_large/interview_demo_kb/retrieval/faqs/faqs.jsonl QDRANT_PATH=vector_db/qdrant/data/local .venv/bin/python scripts/run_faq_processing_pipeline.py
DOCUMENTS_MANIFEST_PATH=cob_mock_kb_large/interview_demo_kb/retrieval/documents/documents_manifest.json QDRANT_PATH=vector_db/qdrant/data/local .venv/bin/python scripts/run_document_processing_pipeline.py
```

Why `DOCUMENTS_MANIFEST_PATH` is set explicitly:

- `config.yml` may still contain an older document manifest path
- setting the env var ensures ingestion uses the interview demo document corpus

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
Bot: Please provide the date you would like to schedule your appointment for Digital Marketing and Website Services.
You: thursday
Bot: For Digital Marketing and Website Services on Next Thursday, the available times are 09:00 AM, 10:30 AM, and 04:00 PM. Which time would you like?
You: 10 30
Bot: Great, I have your appointment for Digital Marketing and Website Services on Next Thursday at 10:30 AM. Could you please provide your name?
You: yasser khira
Bot: Thanks, Yasser Khira. Could you please provide your email address to complete the booking?
You: yasserkhira@gmail.com
Bot: Please confirm your appointment for Digital Marketing and Website Services on Next Thursday at 10:30 AM under the name Yasser Khira with the email yasserkhira@gmail.com. Should I proceed to book it?
```

Escalation example:

```text
You: i need to escilate
Bot: I need to transfer this conversation to a human agent. A human agent will follow up with you. Reason: User requested help from a human or showed frustration.
```

## Documentation

- `README.md`: project overview and usage
- `UI_DEBUG_GUIDE.md`: how to use the Streamlit demo, vector-query display, retrieved chunks, and backend trace panels
- `KB_AGENT_WALKTHROUGH.md`: end-to-end explanation of the KB agent flow, prompt, retrieval, memory, and fallback behavior
- `DOCUMENT_RAG_IMPLEMENTATION.md`: document ingestion, chunking, dual-collection retrieval, and verification notes
- `RETRIEVAL_QUERY_REWRITING.md`: follow-up query rewrite design and implementation details
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

- Improve retrieval filtering and reranking for weak matches
- Add provider-specific retry and error handling for embedding and generation failures
- Add retrieval-quality checks for Gemini/OpenAI experiment sets
- Expand ingestion beyond FAQs into documents and structured data
- Add entity extraction for appointment requests
- Expand the LLM intent classifier prompt and evaluation rules for richer escalation decisions
- Add automated tests for nodes, services, and routing
- Improve the Streamlit debug presentation and source metadata display
