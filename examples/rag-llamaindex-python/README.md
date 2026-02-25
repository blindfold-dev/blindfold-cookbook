# RAG Pipeline with LlamaIndex

PII-safe RAG using LlamaIndex with a custom `BlindfoldNodePostprocessor`. Documents are redacted at ingestion, and retrieved nodes are tokenized before reaching the LLM.

## What this example shows

- **Custom `BlindfoldNodePostprocessor`** — tokenizes retrieved nodes before LLM processing
- **Ingestion-time redaction** — PII is stripped from documents before indexing
- **Query-time tokenization** — user questions and retrieved context are protected
- **LlamaIndex VectorStoreIndex** — in-memory vector search with OpenAI embeddings

## How it works

```
Ingestion:
  Documents → Redact PII → VectorStoreIndex

Query:
  Question → Tokenize → Query Engine → BlindfoldNodePostprocessor
                                        (tokenize retrieved nodes)
                                              ↓
                                         LLM (sees only tokens)
                                              ↓
                                         Detokenize response
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
  Redacted 3 entities from ticket
  Redacted 2 entities from ticket
  Redacted 3 entities from ticket
  Redacted 2 entities from ticket
Indexed 4 documents

=== Query ===
Original question: What happened with John Smith's billing issue?
Tokenized question: What happened with <Person_1>'s billing issue?

Answer: John Smith reported a billing discrepancy where their account
was charged $49.99 twice. A refund was issued within 24 hours.
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python-sdk)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
