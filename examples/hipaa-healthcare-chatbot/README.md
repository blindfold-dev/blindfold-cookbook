# HIPAA-Compliant Healthcare Chatbot

Build a healthcare AI assistant that protects Protected Health Information (PHI) using Blindfold's US region and the `hipaa_us` policy.

## What this example shows

- **US region** — `Blindfold(region="us")` keeps PHI processing within the US
- **`hipaa_us` policy** — detects all 18 HIPAA identifiers: names, SSNs, DOB, MRNs, insurance IDs, addresses, phone numbers, emails, etc.
- **Single query** — tokenize a patient record, send to OpenAI, restore PHI in the response
- **Multi-turn conversation** — maintain PHI mapping across multiple chat turns
- **Batch redaction** — permanently remove PHI from multiple records for safe storage

## How it works

```
Clinical Data                   Blindfold US Region              OpenAI
"Patient Sarah Johnson,         "<Person_1>,                     AI sees only
 SSN 123-45-6789,        →      <Social Security Number_1>,  →  anonymized tokens
 MRN P-4532..."                  <Medical Record Number_1>..."

                                                                      ↓

"Patient Sarah Johnson    ←     Detokenize with mapping    ←    "Patient <Person_1>
 is a 47-year-old..."           (PHI stays in US)                is a 47-year-old..."
```

## HIPAA compliance highlights

- **PHI never reaches the LLM** — tokenization happens before data leaves your system
- **Minimum Necessary Rule** — only de-identified tokens are shared with OpenAI
- **Audit trail** — every API call is logged for compliance audits
- **US data residency** — Blindfold's US node processes PHI within the United States
- **BAA-ready** — contact us for a Business Associate Agreement

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
python main.py
```

## Example output

```
HIPAA-Compliant Patient Query

Original message:
  Patient Sarah Johnson (DOB 03/15/1978, SSN 123-45-6789, MRN P-4532)...

PHI identifiers detected: 7
  - Person: "Sarah Johnson" (confidence: 96%)
  - Date Of Birth: "03/15/1978" (confidence: 94%)
  - Social Security Number: "123-45-6789" (confidence: 99%)
  - Medical Record Number: "P-4532" (confidence: 91%)
  - Email Address: "sarah.johnson@email.com" (confidence: 99%)
  - Phone Number: "(555) 234-5678" (confidence: 97%)
  - Insurance Id: "BC-9876543" (confidence: 88%)

Tokenized (safe to send to AI):
  Patient <Person_1> (DOB <Date Of Birth_1>, SSN <Social Security Number_1>,
  MRN <Medical Record Number_1>) presented at Memorial Hospital...
```

## Three modes of PHI protection

| Mode | Method | Use case |
|---|---|---|
| **Tokenize** | `blindfold.tokenize()` | Reversible — for AI chat, summarization |
| **Redact** | `blindfold.redact()` | Permanent removal — for logs, storage |
| **Encrypt** | `blindfold.encrypt()` | Reversible with key — for secure archives |

## Resources

- [Blindfold HIPAA Guide](https://docs.blindfold.dev/compliance/hipaa)
- [US Region Documentation](https://docs.blindfold.dev/regions)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python)
