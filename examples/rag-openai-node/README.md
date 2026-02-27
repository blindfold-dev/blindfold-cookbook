# RAG Pipeline with PII Protection (OpenAI + ChromaDB) — TypeScript

Build a PII-safe RAG pipeline using OpenAI, ChromaDB, and Blindfold in TypeScript. Contact info is redacted at ingestion (names kept for searchability), and at query time the retrieved context and question are tokenized together before the LLM.

## How it works

1. **Ingest** — support tickets are redacted with `blindfold.redact()` targeting contact info only (names kept)
2. **Search** — user question is used as-is for retrieval (names match in the vector store)
3. **Tokenize** — retrieved context + question tokenized in a single call (consistent token numbering)
4. **Generate** — OpenAI generates an answer from the tokenized prompt (no PII)
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

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
