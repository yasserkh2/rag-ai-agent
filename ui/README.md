# UI Demo

This directory contains a standalone Streamlit chat UI for the customer care agent.

## Run

From the project root:

```bash
.venv/bin/python -m streamlit run ui/streamlit_app.py
```

The app loads runtime settings from `config.yml` and `.env`, then reuses the existing LangGraph backend from `app/`.

## Debug Features

The UI is built for demo and interview visibility, not only end-user chatting.

It shows:

- `Vector DB Query`
  The rewritten query that is embedded and sent to retrieval
- `Retrieved Chunks`
  The chunks returned for the current turn and for each assistant message
- `Backend Trace`
  Per-turn logs for graph routing, node execution, retrieval, and escalation

## Notes

- The clean interview retrieval corpus lives under `cob_mock_kb_large/interview_demo_kb/retrieval/`
- If answers look stale, rebuild the local Qdrant store from that interview dataset before testing the UI
