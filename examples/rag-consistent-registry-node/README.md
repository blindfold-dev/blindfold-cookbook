# RAG with Consistent Token Registry (Strategy C) -- TypeScript

The most sophisticated PII protection strategy for RAG pipelines. Uses Blindfold to detect PII entities, then replaces them with consistent tokens from a global registry. The same entity always maps to the same token across all documents, giving perfect cross-document search quality with zero PII in the vector store.

## How it works

```
Ingestion:
  ticket --> blindfold.tokenize() (detect entities)
         --> registry.getOrCreate() (assign consistent tokens)
         --> registry.replaceInText(original) --> ChromaDB

Query:
  question --> registry.replaceInText() --> search ChromaDB
  retrieved context --> LLM --> registry.restoreText() --> answer
```

### The TokenRegistry class

The core of this strategy is a `TokenRegistry` that maintains a global mapping:

```typescript
class TokenRegistry {
  registry: Map<string, string>;  // "Hans Mueller" -> "<Person_1>"
  reverse: Map<string, string>;   // "<Person_1>" -> "Hans Mueller"

  getOrCreate(realValue, blindfoldToken): string  // get or assign consistent token
  replaceInText(text): string                     // replace real values with tokens
  restoreText(text): string                       // replace tokens with real values
}
```

### Ingestion phase

1. Each ticket is sent to `blindfold.tokenize()` to **detect** PII entities
2. Each detected entity is registered with `registry.getOrCreate()` -- if "Hans Mueller" was already seen, it returns the existing token
3. The **original** ticket text is transformed using `registry.replaceInText()` to apply consistent tokens
4. The consistent text is stored in ChromaDB

**Result**: "Hans Mueller" = `<Person_1>` in EVERY document, "Marie Dupont" = `<Person_2>` in EVERY document.

### Query phase

1. Replace known names in the question with `registry.replaceInText(question)`
2. Search ChromaDB -- consistent tokens match perfectly across all documents
3. Send tokenized context + query to the LLM
4. Restore real values with `registry.restoreText(response)`

## Quick start

```bash
npm install
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## Advantages over other strategies

| Aspect | Strategy A (Redact) | Strategy B (Stored Mapping) | Strategy C (Registry) |
|--------|-------|--------|---------|
| PII in vector store | Names visible | None | None |
| Token consistency | N/A | Different per doc | Same everywhere |
| Search quality | Direct name match | Depends on reverse lookup | Perfect token match |
| Reversibility | Contact info lost | Fully reversible | Fully reversible |
| Complexity | Low | High | Medium |

## Trade-offs

| Aspect | Detail |
|--------|--------|
| **PII in vector store** | None -- all entities replaced with consistent tokens |
| **Search quality** | Excellent -- same token matches across all documents |
| **Reversibility** | Fully reversible via the registry |
| **Token consistency** | Perfect -- "Hans Mueller" is always `<Person_1>` |
| **Persistence** | Registry must be persisted (database in production) |
| **Availability** | Registry required at both ingestion and query time |

## When to use this strategy

- You need zero PII in the vector store
- Cross-document search quality is important
- The same entities appear in multiple documents
- You can maintain a persistent registry (Redis, PostgreSQL, etc.)
- You need the best balance of security and search quality

## Production considerations

In production, back the `TokenRegistry` with a database:
- Use Redis for fast lookups with TTL
- Use PostgreSQL for durability and ACID compliance
- Add locking for concurrent ingestion
- Consider partitioning by tenant for multi-tenant systems

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [TypeScript SDK Reference](https://docs.blindfold.dev/sdks/js-sdk)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
