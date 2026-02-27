# RAG Strategy Comparison: Three PII Protection Approaches (TypeScript)

Side-by-side comparison of three PII protection strategies for RAG pipelines. Runs the same support tickets and queries through all three so you can see the difference.

## Strategies

### A: Selective Redact (simplest)
- **Ingestion**: Redact contact info (emails, phones) -- keep names for searchability
- **Query**: Search with original question, single `tokenize(context+question)`, LLM, `detokenize`
- **Complexity**: Low -- stateless, no mapping storage

### B: Tokenize with Stored Mapping (per-document)
- **Ingestion**: Tokenize everything, store mapping per document
- **Query**: Reverse-lookup names to tokens, search, LLM, `detokenize` with merged mappings
- **Complexity**: High -- per-document mappings, same person gets different tokens in each doc
- **Problem**: "Hans Mueller" = `<Person_1>` in doc A but `<Person_3>` in doc B

### C: Consistent Token Registry (best search quality)
- **Ingestion**: Blindfold detects PII, app assigns consistent tokens from a global registry
- **Query**: Lookup names in registry, search with consistent tokens, LLM, reverse replace
- **Complexity**: Medium -- requires persistent registry, but no per-document mapping headaches
- **Advantage**: "Hans Mueller" = `<Person_1>` in EVERY document

## Quick start

```bash
npm install
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## CLI usage

Run all three strategies (default):
```bash
npm start
```

Run only Strategy C:
```bash
npm start c
```

Run Strategy A and C:
```bash
npm start a c
```

## What to look for

1. **Consistent tokens (Strategy C)**: Marie Dupont appears in tickets #2 and #4. In Strategy B she gets different tokens per doc. In Strategy C she is `<Person_2>` everywhere.
2. **Search quality**: "What problems did Marie Dupont have?" -- Strategy C matches both tickets perfectly because tokens are consistent.
3. **Vector store content**: Strategy A has names in plain text, B and C have only tokens -- but C's tokens are consistent.
4. **Registry dump**: Strategy C prints its registry after ingestion so you can see the global token assignments.

## Trade-off summary

| | Names in vector store | Name search | Consistent tokens | Complexity |
|-|----------------------|-------------|-------------------|------------|
| **A** | Yes | Direct match | N/A | Low |
| **B** | No (tokens) | Via reverse lookup | No -- different per doc | High |
| **C** | No (tokens) | Via registry lookup | Yes -- same everywhere | Medium |

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [TypeScript SDK Reference](https://docs.blindfold.dev/sdks/js-sdk)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
