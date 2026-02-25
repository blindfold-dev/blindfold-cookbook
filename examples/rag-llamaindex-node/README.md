# LlamaIndex RAG Pipeline with PII Protection — TypeScript

PII-safe RAG using LlamaIndex.TS with Blindfold. Documents are redacted at ingestion, and user queries are tokenized before reaching the LLM.

## How it works

1. **Ingest** — support tickets are redacted with `blindfold.redact()` before indexing
2. **Query** — user question is tokenized with `blindfold.tokenize()`
3. **Retrieve + Generate** — LlamaIndex queries the index with tokenized question
4. **Detokenize** — `blindfold.detokenize()` restores real names in the response

## Quick start

```bash
cd examples/rag-llamaindex-node
npm install
cp .env.example .env
# Add your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [LlamaIndex.TS Documentation](https://ts.llamaindex.ai/)
- [Node.js SDK Reference](https://docs.blindfold.dev/sdks/javascript-sdk)
