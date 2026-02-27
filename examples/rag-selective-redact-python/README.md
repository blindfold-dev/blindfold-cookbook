# RAG with Selective Redaction (Python)

Protect PII in a RAG pipeline by selectively redacting contact info at ingestion while keeping names searchable. The simplest strategy for PII-safe RAG.

## Strategy

**Ingestion**: Redact emails and phone numbers using `blindfold.redact()`. Names stay in plain text for direct vector search matching.

**Query**: Search with the original question (names match directly), tokenize the retrieved context and question together in a single call, send to the LLM, then detokenize the response.

## How it works

```
Ingestion:
  Support ticket
    --> blindfold.redact(entities=["email address", "phone number"])
    --> Names preserved, contact info removed
    --> Store in ChromaDB

Query:
  User question ---------> ChromaDB search (names match directly)
                                |
                          Retrieved context
                                |
                           context + question
                                |
                     blindfold.tokenize() (single call)
                                |
                          OpenAI LLM call
                                |
                      blindfold.detokenize()
                                |
                          Final answer
```

## Key benefits

- **Simple and stateless** -- no mapping storage, no registry, no reverse lookups
- **Direct name search** -- names in the vector store match user queries naturally
- **Single tokenize call** -- context and question get consistent token numbering
- **Contact info protected** -- emails and phones never reach the vector store or LLM

## Trade-offs

- Names remain in the vector store (acceptable for internal infrastructure, but not for zero-knowledge requirements)
- Redacted contact info is permanently removed (not reversible from the stored documents)

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
python main.py
```

## Example output

```
============================================================
INGESTION: Redact contact info, keep names
============================================================

  Ticket 1: 2 entities redacted
    Stored: "Ticket #1001: Customer Hans Mueller ([EMAIL_ADDRESS], [PHONE_NUMBER]) reported a bi..."

  Ticket 2: 2 entities redacted
    Stored: "Ticket #1002: Marie Dupont ([EMAIL_ADDRESS], [PHONE_NUMBER]) cannot access her dash..."

  ...

------------------------------------------------------------
  Question: "What was the issue reported by Hans Mueller?"

  Retrieved 2 chunks:
    "Ticket #1001: Customer Hans Mueller ([EMAIL_ADDRESS], [PHONE_NUMBER]) reported a bi..."

  Tokenized prompt (preview):
    "Context:\nTicket #1001: Customer <Person_1> ([EMAIL_ADDRESS], [PHONE_NUMBER]) reported a billing..."

  LLM response (tokenized):
    "<Person_1> reported a billing error on invoice INV-2024-0047..."

  Final answer:
    "Hans Mueller reported a billing error on invoice INV-2024-0047..."
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python-sdk)
- [Strategy Comparison Example](../rag-strategy-comparison-python/)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
