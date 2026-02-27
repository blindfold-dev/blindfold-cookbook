# RAG with Stored Mapping (Strategy B) -- TypeScript

Full tokenization strategy for RAG pipelines. Every ticket is fully tokenized at ingestion, and the token-to-value mapping is stored alongside each document. Zero PII in the vector store, but with the trade-off of inconsistent tokens across documents.

## How it works

```
Ingestion:
  ticket --> blindfold.tokenize() --> store tokenized text + mapping in ChromaDB

Query:
  question --> reverse lookup (names -> tokens) --> search ChromaDB
  retrieved docs --> merge mappings --> LLM --> blindfold.detokenize(merged) --> answer
```

### Ingestion phase

Each support ticket is passed through `blindfold.tokenize()`. The tokenized text goes into the vector store, and the token mapping is stored as metadata.

**Before**: `"Customer Hans Mueller (hans.mueller@example.de) reported..."`
**After**: `"Customer <Person_1> (<Email_Address_1>) reported..."` + mapping stored

### Query phase

1. Build a reverse lookup from all stored mappings (real value -> token)
2. Replace known names in the question with their tokens
3. Search ChromaDB with the tokenized query
4. Merge mappings from all retrieved documents
5. Send tokenized context + query to the LLM
6. Detokenize the LLM response using the merged mapping

### The inconsistency problem

Because Blindfold generates fresh tokens per `tokenize()` call, the same person gets different tokens in different documents:

- Ticket #2: Marie Dupont = `<Person_1>`
- Ticket #4: Marie Dupont = `<Person_2>`

This means vector search relies on the reverse lookup picking the right token. As the corpus grows, this becomes increasingly complex.

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
| **PII in vector store** | None -- all entities tokenized |
| **Reversibility** | Fully reversible via stored mappings |
| **Search quality** | Depends on reverse lookup accuracy |
| **Token consistency** | No -- same person gets different tokens per document |
| **Storage overhead** | Requires per-document mapping storage |
| **Complexity** | High -- reverse lookup, mapping merge, collision handling |

## When to use this strategy

- Zero PII in the vector store is a hard requirement
- Your corpus is relatively small (dozens to low hundreds of documents)
- You need full reversibility of all PII
- You can tolerate the complexity of per-document mapping management

## When to consider alternatives

- For simpler integration, use [Selective Redact (Strategy A)](../rag-selective-redact-node/)
- For better search quality with consistent tokens, use [Consistent Registry (Strategy C)](../rag-consistent-registry-node/)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [TypeScript SDK Reference](https://docs.blindfold.dev/sdks/js-sdk)
