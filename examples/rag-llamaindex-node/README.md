# LlamaIndex RAG Pipeline with PII Protection — TypeScript

PII-safe RAG using LlamaIndex.TS with Blindfold. Contact info is redacted at ingestion (names kept for searchability). At query time, retrieves with the original question, then tokenizes context + question in a single call before the LLM.

## How it works

1. **Ingest** — support tickets are redacted with `blindfold.redact()` targeting contact info only (names kept)
2. **Search** — original question used for retrieval (names match in the index)
3. **Tokenize** — retrieved context + question tokenized in a single call
4. **Generate** — LLM sees only tokenized prompt (no PII)
5. **Detokenize** — `blindfold.detokenize()` restores real names in the response

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

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
