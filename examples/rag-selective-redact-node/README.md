# RAG with Selective Redaction (Strategy A) -- TypeScript

The simplest PII protection strategy for RAG pipelines. Contact information is permanently redacted at ingestion while names are kept for searchability. At query time, context and question are tokenized before the LLM sees them.

## How it works

```
Ingestion:
  ticket --> blindfold.redact(entities: ["email address", "phone number"]) --> ChromaDB
  (emails/phones removed, names kept)

Query:
  question --> search (names match directly) --> retrieve context
  context + question --> blindfold.tokenize() --> LLM --> blindfold.detokenize() --> answer
```

### Ingestion phase

Each support ticket is passed through `blindfold.redact()` with specific entity types. Only emails and phone numbers are removed -- names stay in the text so vector search can match them directly.

**Before**: `"Customer Hans Mueller (hans.mueller@example.de, +49 151 12345678) reported..."`
**After**: `"Customer Hans Mueller (*****, *****) reported..."`

### Query phase

1. Search ChromaDB with the original question (names match naturally)
2. Combine retrieved context with the question
3. Tokenize everything in a single `blindfold.tokenize()` call
4. Send the tokenized prompt to the LLM (no real PII)
5. Detokenize the LLM response back to real values

## Quick start

```bash
npm install
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## Trade-offs

| Aspect | Detail |
|--------|--------|
| **Simplicity** | Lowest complexity -- no mapping storage, no registry |
| **Search quality** | Excellent -- names match directly in vector search |
| **PII in vector store** | Names are stored in plain text (internal risk) |
| **Contact info** | Permanently lost after redaction |
| **Statefulness** | Fully stateless -- each document processed independently |

## When to use this strategy

- You need a quick, low-complexity solution
- Names in the vector store are acceptable for your threat model
- Contact information does not need to be recoverable
- You want the simplest possible integration

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [TypeScript SDK Reference](https://docs.blindfold.dev/sdks/js-sdk)
