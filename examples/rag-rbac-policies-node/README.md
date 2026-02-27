# RAG with Role-Based PII Control (RBAC Policies) -- TypeScript

Demonstrates how different user roles see different levels of PII in a healthcare RAG pipeline. Each role has a Blindfold tokenization policy that controls which entity types are protected before the LLM sees them.

## How it works

```
Ingestion (shared across all roles):
  patient record --> blindfold.redact(entities: ["email address", "phone number"]) --> ChromaDB
  (contact info removed, clinical data kept for search)

Query (role-specific):
  question --> search --> retrieve context
  context + question --> blindfold.tokenize({ entities | policy }) --> LLM --> blindfold.detokenize() --> answer
```

### Roles

| Role | Sees | Tokenized |
|------|------|-----------|
| **Doctor** | Names, conditions, medications, DOB | Emails, phones, SSNs, financial IDs |
| **Nurse** | Names, conditions, medications | Emails, phones, SSNs, DOB, financial IDs |
| **Billing** | Names, insurance info | Emails, phones, SSNs, conditions, medications |
| **Researcher** | De-identified content only | All PII (policy="strict") |

### Ingestion phase

All records share the same ingestion pipeline. Contact information (emails, phone numbers) is permanently redacted so the vector store never holds raw contact data. Clinical content and names remain for searchability.

**Before**: `"Patient Record #PR-2024-001: Sarah Chen (sarah.chen@email.com, +1-555-0142, SSN 412-55-6789, DOB 1985-03-15) was diagnosed with Type 2 Diabetes..."`
**After**: `"Patient Record #PR-2024-001: Sarah Chen (*****, *****, SSN 412-55-6789, DOB 1985-03-15) was diagnosed with Type 2 Diabetes..."`

### Query phase

1. Search ChromaDB with the original question (names match naturally)
2. Combine retrieved context with the question
3. Tokenize using the role's entity list or policy:
   - `entities: [...]` for doctor, nurse, billing
   - `policy: "strict"` for researcher (removes all PII)
4. Send the tokenized prompt to the LLM (only role-appropriate data visible)
5. Detokenize the LLM response back to real values

## Quick start

```bash
npm install
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start                    # run all roles
npm start -- --role doctor   # run a single role
npm start -- --role nurse
npm start -- --role billing
npm start -- --role researcher
```

## Custom policies

In addition to the built-in `"strict"` policy, you can create custom policies in the [Blindfold dashboard](https://app.blindfold.dev) and reference them by name:

```typescript
// Use a custom policy created in the dashboard
const tokenized = await blindfold.tokenize(text, { policy: "hipaa_minimum_necessary" });
```

This lets you manage tokenization rules centrally without changing application code.

## When to use this pattern

- Your application has multiple user roles with different data access levels
- You need to enforce PII visibility rules at the LLM boundary
- Healthcare, finance, or HR systems with role-based access requirements
- You want to centrally manage PII policies without code changes

## Resources

- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [TypeScript SDK Reference](https://docs.blindfold.dev/sdks/js-sdk)
- [Policies Documentation](https://docs.blindfold.dev/policies)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
