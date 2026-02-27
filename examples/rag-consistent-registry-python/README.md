# RAG with Consistent Token Registry (Python)

Protect PII in a RAG pipeline using a global token registry that assigns the same token to the same person across all documents. Zero PII in the vector store with perfect cross-document search quality.

## Strategy

**Ingestion**: Use `blindfold.tokenize()` to detect PII entities, then register each with a `TokenRegistry` that assigns consistent tokens. The same person always gets the same token regardless of which document they appear in.

**Query**: Replace known names in the question with their registry tokens, search the vector store (consistent tokens match perfectly), call the LLM, then restore real values using the registry.

## How it works

```
Ingestion:
  Support ticket
    --> blindfold.tokenize() (detect PII entities)
    --> TokenRegistry.get_or_create() for each entity
    --> TokenRegistry.replace_in_text() on original text
    --> Store consistently-tokenized text in ChromaDB

  Registry state after all tickets:
    <Person_1>         = "Hans Mueller"
    <Person_2>         = "Marie Dupont"          <-- same in EVERY document
    <Person_3>         = "Lars Johansson"
    <Email_Address_1>  = "hans.mueller@example.de"
    ...

Query:
  User question
    --> TokenRegistry.replace_in_text() (names -> consistent tokens)
    --> ChromaDB search (tokens match exactly across all docs)
            |
      Retrieved context (consistently tokenized)
            |
      OpenAI LLM call (tokenized prompt)
            |
      TokenRegistry.restore_text() (tokens -> real names)
            |
      Final answer
```

## Key advantage: consistent tokens

Unlike per-document tokenization (where each `tokenize()` call creates independent tokens), the registry ensures Marie Dupont is `<Person_2>` in both ticket #2 and ticket #4:

```
Ticket #1002: <Person_2> (<Email_Address_2>, <Phone_Number_2>) cannot access...
Ticket #1004: <Person_2> (<Email_Address_2>, <Phone_Number_2>) reports a second issue...
```

This means:
- Searching for `<Person_2>` matches ALL of Marie's tickets
- No reverse lookup collisions
- The LLM sees coherent references across retrieved documents

## The TokenRegistry class

The `TokenRegistry` maintains three data structures:

- `registry`: maps real values to consistent tokens (`"Hans Mueller" -> "<Person_1>"`)
- `reverse`: maps consistent tokens back to real values (`"<Person_1>" -> "Hans Mueller"`)
- `_counters`: tracks the next ID per entity type (`"Person" -> 3`)

It uses Blindfold's token format to extract entity types (e.g., `<Person_1>` -> `"Person"`) and assigns sequential IDs within each type.

## Trade-offs

**Advantages:**
- Zero PII in the vector store -- tokens only
- Same person = same token everywhere -- perfect search quality
- Simple reverse lookup -- one token per real value, no collisions
- Fully reversible -- registry can restore all original values

**Considerations:**
- Requires a persistent token registry (database in production)
- Uses Blindfold as PII detector, does replacement itself (no `detokenize()` at query time)
- Registry must be available at both ingestion and query time
- New PII values not yet in the registry will not be replaced in queries

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
INGESTION: Detect PII, assign consistent tokens
============================================================

  Ticket 1: 3 entities detected
    Stored: "Ticket #1001: Customer <Person_1> (<Email_Address_1>, <Phone_Number_1>) reported..."

  Ticket 2: 3 entities detected
    Stored: "Ticket #1002: <Person_2> (<Email_Address_2>, <Phone_Number_2>) cannot access her..."

  Ticket 3: 3 entities detected
    Stored: "Ticket #1003: <Person_3> (<Email_Address_3>, <Phone_Number_3>) asked to export..."

  Ticket 4: 3 entities detected
    Stored: "Ticket #1004: <Person_2> (<Email_Address_2>, <Phone_Number_2>) reports a second..."

  --------------------------------------------------------
  REGISTRY DUMP (9 entries):
  --------------------------------------------------------
    <Person_1>                = "Hans Mueller"
    <Email_Address_1>         = "hans.mueller@example.de"
    <Phone_Number_1>          = "+49 151 12345678"
    <Person_2>                = "Marie Dupont"
    <Email_Address_2>         = "marie.dupont@example.fr"
    <Phone_Number_2>          = "+33 6 12 34 56 78"
    <Person_3>                = "Lars Johansson"
    <Email_Address_3>         = "lars.johansson@example.se"
    <Phone_Number_3>          = "+46 70 123 4567"

  Consistency check: Marie Dupont appears in tickets #2 and #4
    Both use: <Person_2>

------------------------------------------------------------
  Question: "What problems did Marie Dupont have?"

  Replacements in query:
    "Marie Dupont" -> <Person_2>
  Search query: "What problems did <Person_2> have?"

  Retrieved 2 chunks:
    "Ticket #1002: <Person_2> (<Email_Address_2>, <Phone_Number_2>) cannot access..."
    "Ticket #1004: <Person_2> (<Email_Address_2>, <Phone_Number_2>) reports a sec..."

  LLM response (tokenized):
    "<Person_2> had two issues: (1) locked out of dashboard after password reset..."

  Final answer:
    "Marie Dupont had two issues: (1) locked out of dashboard after password reset..."
```

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python-sdk)
- [Strategy Comparison Example](../rag-strategy-comparison-python/)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
