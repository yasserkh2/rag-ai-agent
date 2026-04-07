# Interview Demo KB

This dataset is the interview-ready knowledge base layout for the demo chatbot.

## Design Goal

Keep the retrieval corpus clean:

- `retrieval/` is the only folder intended for RAG indexing
- `operations/` contains supporting operational data and should not be indexed for normal FAQ/document retrieval

That separation avoids the major overlap problem where rewritten high-quality retrieval files and the large mixed KB both describe the same services.

## Folder Map

- `retrieval/documents/`
  High-quality long-form documents for grounding.
- `retrieval/faqs/faqs.jsonl`
  High-quality concise FAQ set for direct answers.
- `operations/appointments/`
  Mock appointment records for booking flows.
- `operations/case_notes/`
  Mock case history for escalation flows.
- `operations/structured/`
  Canonical metadata for routing, normalization, and deterministic lookups.

## Recommended Usage

- Use `retrieval/` for vector indexing.
- Use `operations/structured/` for normalization and service metadata.
- Use `operations/appointments/` and `operations/case_notes/` only for workflow features, not as general RAG knowledge sources.
