# COB Solution Large Mock Knowledge Base

This package contains a large synthetic dataset for an interview chatbot task.

Files:
- cob_kb_master.json
- cob_documents.json
- cob_faqs.json
- cob_structured_data.json
- cob_case_notes.json
- cob_appointments.json

Counts:
- Documents: 45
- FAQs: 156
- Case notes: 60
- Appointment examples: 120

Design:
- Documents = long-form retrieval chunks
- FAQs = concise question-answer pairs
- Structured data = company/profile info, products, services, policies, contacts, packages, intents, entities, routing rules, slots, action flows, escalation playbook, mock API contracts
- Case notes = extra retrieval variety
- Appointments = action-flow mock records enriched with service IDs, slot IDs, phone numbers, and booking references

Important:
This is synthetic demo data and not official internal COB data.

Very large mixed-format dataset:
- See `very_large_mixed_kb/` for a high-volume collection with markdown documents, FAQ jsonl/csv, and structured json/csv tables.
