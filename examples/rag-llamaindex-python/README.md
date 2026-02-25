# RAG Pipeline with LlamaIndex

PII-safe RAG using LlamaIndex with Blindfold. Contact info is redacted at ingestion (names kept for searchability). At query time, retrieves with the original question, then tokenizes context + question in a single call before the LLM.

## What this example shows

- **Selective ingestion redaction** — redact emails and phone numbers, keep names searchable
- **Search-first flow** — retrieve with original question so names match in the index
- **Single tokenize call** — context + question tokenized together for consistent token numbering
- **LlamaIndex VectorStoreIndex** — in-memory vector search with OpenAI embeddings

## How it works

```
Ingestion:
  Documents → Redact contact info (keep names) → VectorStoreIndex

Query:
  Original question → Retrieve nodes → Tokenize(context + question) → LLM → Detokenize
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
python main.py
```

## Example output

```
=== Ingestion ===
  Redacted 2 entities from ticket
  Redacted 2 entities from ticket
  Redacted 2 entities from ticket
  Redacted 2 entities from ticket
Indexed 4 documents

=== Query ===
Original question: What happened with John Smith's billing issue?

Retrieved context:
  "Ticket #1001: Customer John Smith ([EMAIL_ADDRESS], [PHONE_NUMBER])..."

Answer: John Smith reported a billing discrepancy where their account
was charged $49.99 twice. A refund was issued within 24 hours.
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python-sdk)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
