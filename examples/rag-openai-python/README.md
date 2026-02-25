# RAG Pipeline with PII Protection (OpenAI + ChromaDB)

Build a PII-safe RAG pipeline using OpenAI, ChromaDB, and Blindfold. Documents are redacted at ingestion time, and user queries are tokenized at query time — the LLM never sees real personal data.

## What this example shows

- **Two protection layers** — redact PII during ingestion, tokenize PII during queries
- **ChromaDB vector store** — in-memory vector database for document retrieval
- **Tokenize → Retrieve → Generate → Detokenize** — the full RAG flow with privacy protection
- **Realistic sample data** — support tickets with names, emails, and phone numbers

## How it works

```
Ingestion:
  Support tickets → Split → Redact PII → Embed → ChromaDB

Query:
  User question          Blindfold              OpenAI
  "What happened with    "What happened with    AI generates answer
   Hans's ticket?"  →    <Person_1>'s ticket?"  using redacted context
                              ↓                       ↓
                         Query ChromaDB          Detokenize response
                         (redacted docs)         → real names restored
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
Redacting PII from 4 support tickets...

  Ticket 1: 3 entities redacted
  Ticket 2: 2 entities redacted
  Ticket 3: 2 entities redacted
  Ticket 4: 2 entities redacted

Stored 4 chunks in ChromaDB

=== Query ===
User question: "What was the issue reported by Hans Mueller?"

Tokenized question: "What was the issue reported by <Person_1>?"

Retrieved context (redacted):
  "Ticket #1001: Customer [PERSON] ([EMAIL_ADDRESS], [PHONE_NUMBER])..."

AI response (with tokens): "Based on the records, <Person_1> reported..."
Final response: "Based on the records, Hans Mueller reported..."
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python-sdk)
- [Blindfold Cookbook](https://github.com/blindfold-dev/blindfold-cookbook)
