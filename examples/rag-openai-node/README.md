# RAG Pipeline with PII Protection (OpenAI + ChromaDB) — TypeScript

Build a PII-safe RAG pipeline using OpenAI, ChromaDB, and Blindfold in TypeScript. Documents are redacted at ingestion, and user queries are tokenized at query time.

## How it works

1. **Ingest** — support tickets are redacted with `blindfold.redact()` before indexing into ChromaDB
2. **Query** — user question is tokenized with `blindfold.tokenize()`
3. **Retrieve** — ChromaDB returns redacted documents (no PII in context)
4. **Generate** — OpenAI generates an answer from redacted context + tokenized question
5. **Detokenize** — `blindfold.detokenize()` restores real names in the response

## Quick start

```bash
cd examples/rag-openai-node
npm install
cp .env.example .env
# Add your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Node.js SDK Reference](https://docs.blindfold.dev/sdks/javascript-sdk)
- [Blindfold Cookbook](https://github.com/blindfold-dev/blindfold-cookbook)
