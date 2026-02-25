# LangChain RAG Pipeline with PII Protection — TypeScript

PII-safe RAG using LangChain.js with inline Blindfold integration. Contact info is redacted before indexing (names kept for searchability), and queries use explicit retrieve-then-tokenize for consistent PII protection.

## What this example shows

- **Inline `transformDocuments()`** — selectively redacts contact info in LangChain Documents
- **Retrieve-then-tokenize** — search with original question, then tokenize context + question together
- **MemoryVectorStore** — lightweight in-memory vector search with OpenAI embeddings
- **No `langchain-blindfold` JS package needed** — all integration is done inline

## How it works

```
Ingestion:
  Documents → Split → transformDocuments(redact emails/phones) → Embed → MemoryVectorStore

Query:
  Original question → Retrieve → Tokenize(context + question) → LLM → Detokenize
```

## Quick start

```bash
cd examples/rag-langchain-node
npm install
cp .env.example .env
# Add your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [LangChain.js Documentation](https://js.langchain.com/)
- [Node.js SDK Reference](https://docs.blindfold.dev/sdks/javascript-sdk)
