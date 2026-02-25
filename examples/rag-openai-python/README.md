# RAG Pipeline with PII Protection (OpenAI + ChromaDB)

Build a PII-safe RAG pipeline using OpenAI, ChromaDB, and Blindfold. Contact info is redacted at ingestion (names kept for searchability), and at query time the retrieved context and question are tokenized in a single call before the LLM.

## What this example shows

- **Selective ingestion redaction** — redact emails and phone numbers, keep names for vector search
- **Search-first query flow** — search with original question so names match in the vector store
- **Single tokenize call** — context + question tokenized together for consistent token numbering
- **ChromaDB vector store** — in-memory vector database for document retrieval

## How it works

```
Ingestion:
  Support tickets → Split → Redact contact info (keep names) → Embed → ChromaDB

Query:
  User question           ChromaDB              Blindfold             OpenAI
  "What happened with     Search with           Tokenize context      AI generates answer
   Hans's ticket?"   →    original question  →  + question together → from tokenized prompt
                          (names match!)        (single API call)          ↓
                                                                     Detokenize response
                                                                     → real names restored
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
Redacting contact info from 4 support tickets...

  Ticket 1: 2 entities redacted
  Ticket 2: 2 entities redacted
  Ticket 3: 2 entities redacted
  Ticket 4: 2 entities redacted

Stored 4 chunks in ChromaDB

=== Query ===
User question: "What was the issue reported by Hans Mueller?"

Retrieved context:
  "Ticket #1001: Customer Hans Mueller ([EMAIL_ADDRESS], [PHONE_NUMBER])..."

AI response (with tokens): "Based on the records, <Person_1> reported..."
Final response: "Based on the records, Hans Mueller reported..."
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python-sdk)
- [Blindfold Cookbook](https://github.com/blindfold-dev/blindfold-cookbook)
