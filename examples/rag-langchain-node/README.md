# LangChain RAG Pipeline with PII Protection — TypeScript

PII-safe RAG using LangChain.js with inline Blindfold integration. Documents are redacted before indexing, and the retrieval chain is wrapped with tokenize/detokenize.

## What this example shows

- **Inline `transformDocuments()`** — redacts PII in LangChain Documents before indexing
- **Inline `blindfoldProtect()`** — wraps the chain with tokenize/detokenize RunnableLambdas
- **MemoryVectorStore** — lightweight in-memory vector search with OpenAI embeddings
- **No `langchain-blindfold` JS package needed** — all integration is done inline

## How it works

```
Ingestion:
  Documents → Split → transformDocuments(redact) → Embed → MemoryVectorStore

Query:
  tokenize → { context: retriever, question } → prompt → LLM → detokenize
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
