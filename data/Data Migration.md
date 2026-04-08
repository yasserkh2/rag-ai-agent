# Data Migration Notes

## Overview
This directory was created as a new standalone data source for the knowledge base pipeline.

The goal was to move away from relying only on the older dataset under `cob_mock_kb_large` and start using a cleaner, more focused set of markdown documents under `data/documents`.

## What We Added
- Created a new top-level `data/` directory
- Created `data/documents/` for markdown knowledge-base documents
- Created `data/fandq/` for FAQ content
- Added a new set of COB Solution markdown documents in `data/documents`
- Added `data/documents/documents_manifest.json` so the current document ingestion pipeline can load the new files

## Why The Manifest Was Needed
The current document ingestion pipeline does not auto-scan a folder for markdown files.
It expects a `documents_manifest.json` file that lists every document and provides:

- `doc_id`
- `service_id`
- `service_name`
- `title`
- `file_path`
- `source_type`

Without this manifest, the files exist on disk but the pipeline does not know:
- which files to ingest
- what stable IDs to assign
- what metadata to attach
- how to create predictable chunk IDs

## Document Set Created
The new document set includes:

- `doc_0001_company_overview.md`
- `doc_0002_authorizations.md`
- `doc_0003_benefits_verification.md`
- `doc_0004_medical_billing_denial_management.md`
- `doc_0005_medical_auditing.md`
- `doc_0006_financial_management.md`
- `doc_0007_digital_marketing_website_services.md`
- `doc_0008_customer_care.md`
- `doc_0009_elevate_communication_services.md`
- `doc_0010_credentialing_provider_maintenance.md`
- `doc_0011_contact_information.md`

## Compatibility Adjustments
To make the new markdown files work with the existing chunking pipeline, the documents were written or normalized to use the section names that the chunker already expects:

- `## Service Overview`
- `## What This Service Usually Includes`
- `## Common Practice Scenarios`
- `## Information To Collect During Intake`
- `## How The Chatbot Should Respond`
- `## Escalation Guidance`
- `## Example Customer Questions`
- `## Keywords`

Two files needed extra normalization so their important content would be chunked correctly:

- `doc_0001_company_overview.md`
- `doc_0011_contact_information.md`

## Validation
The new manifest was tested with the local ingestion pipeline and successfully loaded 11 document records from `data/documents`.

## How To Use The New Documents
Point the document pipeline to the new dataset with one of these environment variables:

```bash
DOCUMENTS_ROOT_PATH=/media/yasser/New Volume1/yasser/New_journey/customer-care-ai-agent/data/documents
```

or

```bash
DOCUMENTS_MANIFEST_PATH=/media/yasser/New Volume1/yasser/New_journey/customer-care-ai-agent/data/documents/documents_manifest.json
```

## Next Step
`data/documents` is now ready for the current documents pipeline.
The next step, if needed, is to make `data/fandq` fully pipeline-ready in the same way.
