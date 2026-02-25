# RAG with Stored Token Mapping (Python)

Protect PII in a RAG pipeline by tokenizing everything at ingestion and storing per-document mappings. Zero PII in the vector store, but with the trade-off that the same person gets different tokens in each document.

## Strategy

**Ingestion**: Tokenize each document with `blindfold.tokenize()`, store the tokenized text in ChromaDB and the mapping (token-to-real-value) in a side store (also saved as metadata in ChromaDB).

**Query**: Build a reverse lookup from all stored mappings, replace real names in the question with their known tokens, search the vector store, merge mappings from retrieved documents, call the LLM, then detokenize the response.

## How it works

```
Ingestion:
  Support ticket
    --> blindfold.tokenize()
    --> Store tokenized text in ChromaDB
    --> Store mapping {<Person_1>: "Hans Mueller", ...} per document

Query:
  User question
    --> Build reverse lookup (real_value -> tokens)
    --> Replace names with known tokens
    --> ChromaDB search (tokens match tokenized docs)
            |
      Retrieved context (already tokenized)
            |
      Merge mappings from retrieved docs
            |
      OpenAI LLM call (tokenized prompt)
            |
      blindfold.detokenize(response, merged_mapping)
            |
      Final answer (real names restored)
```

## Key trade-off: inconsistent tokens

Each `blindfold.tokenize()` call generates independent token assignments. Marie Dupont appears in tickets #2 and #4, but she gets a **different token in each**:

```
ticket-1: <Person_1> = "Marie Dupont"     (Ticket #1002)
ticket-3: <Person_1> = "Marie Dupont"     (Ticket #1004, but different <Person_1>!)
```

This means:
- The reverse lookup maps "Marie Dupont" to multiple tokens across documents
- Searching for one token may not match the other document
- Merged mappings can have collisions (two different documents both use `<Person_1>` for different people)

## When to use this strategy

- You need **zero PII in the vector store** (full tokenization)
- Your documents are relatively independent (few cross-document name references)
- You can accept some search quality loss from inconsistent tokens
- You want Blindfold to handle both detection and replacement (no custom code)

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
python main.py
```

## Example output

```
============================================================
INGESTION: Tokenize everything, store per-document mappings
============================================================

  Ticket 1: 3 tokens created
    Stored: "Ticket #1001: Customer <Person_1> (<Email_Address_1>, <Phone_Number_1>) reported..."
    <Person_1> = "Hans Mueller"
    <Email_Address_1> = "hans.mueller@example.de"
    <Phone_Number_1> = "+49 151 12345678"

  ...

  NOTE: Same person gets DIFFERENT tokens per document:
    ticket-1: <Person_1> = "Marie Dupont"
    ticket-3: <Person_1> = "Marie Dupont"

------------------------------------------------------------
  Question: "What was the issue reported by Hans Mueller?"

  Reverse lookup: 9 unique real values
  Replacements in query:
    "Hans Mueller" -> <Person_1> (also known as: <Person_1> (ticket-0))
  Search query: "What was the issue reported by <Person_1>?"

  Retrieved 2 chunks:
    "Ticket #1001: Customer <Person_1> (<Email_Address_1>, <Phone_Number_1>) reported..."

  Merged mapping (6 entries from 2 docs)

  LLM response (tokenized):
    "<Person_1> reported a billing error on invoice INV-2024-0047..."

  Final answer:
    "Hans Mueller reported a billing error on invoice INV-2024-0047..."
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python-sdk)
- [Strategy Comparison Example](../rag-strategy-comparison-python/)
