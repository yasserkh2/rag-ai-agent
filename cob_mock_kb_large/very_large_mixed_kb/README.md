# COB Very Large Mixed KB

This dataset is designed to satisfy the chatbot task requirement of a **collection of documents, FAQs, and structured data** with very large volume.

## Scale
- Markdown documents: 1200
- FAQ entries: 10000
- Case note rows: 4000
- Appointment rows: 20000
- KPI rows: 2000

## Mixed Formats
- Documents: `documents/*.md` + `structured/documents_index.json`
- FAQs: `faqs/faqs.jsonl`, `faqs/faqs.csv`
- Structured: `structured/structured_data.json`, `structured/services.csv`, `structured/service_kpis.csv`
- Cases: `case_notes/case_notes.jsonl`
- Appointments: `appointments/appointments.jsonl`, `appointments/appointments.csv`

## Notes
- All records are synthetic and for interview/demo use.
- Dates begin on 2026-04-06 and extend forward for large-volume simulation.
