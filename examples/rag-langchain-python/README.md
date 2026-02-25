# RAG Pipeline with LangChain + FAISS

PII-safe RAG using LangChain's native Blindfold integration. Documents are protected with `BlindfoldPIITransformer` at ingestion, and queries are wrapped with `blindfold_protect()` at runtime.

## What this example shows

- **`BlindfoldPIITransformer`** — redact PII in LangChain Documents before indexing
- **`blindfold_protect()`** — wrap the retrieval chain with automatic tokenization/detokenization
- **FAISS vector store** — lightweight in-memory vector search
- **End-to-end LangChain RAG** — loader → splitter → transformer → vectorstore → retrieval chain

## How it works

```
Ingestion:
  Documents → Split → BlindfoldPIITransformer(redact) → Embed → FAISS

Query:
  tokenize → { context: retriever, question } → prompt → LLM → detokenize
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
=== Document Ingestion ===
  Doc 1:
    Original: "Ticket #1001: Customer Sarah Chen (sarah.chen@acme.com, +1-555-234-567..."
    Protected: "Ticket #1001: Customer [PERSON] ([EMAIL_ADDRESS], [PHONE_NUMBER])..."
Stored 4 documents in FAISS

=== RAG Query ===
Question: What was Sarah Chen's issue?
Answer: Sarah Chen reported a billing discrepancy where her account was charged
$49.99 twice. A refund was issued within 24 hours.
```

## Resources

- [langchain-blindfold on PyPI](https://pypi.org/project/langchain-blindfold/)
- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [LangChain Integration Docs](https://docs.blindfold.dev/sdks/langchain)
