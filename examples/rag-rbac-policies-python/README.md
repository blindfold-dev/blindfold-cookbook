# RAG with Role-Based PII Control

Same vector store, same retrieval, same LLM -- but different user roles see different levels of PII protection. The role determines which Blindfold policy is applied at query time before the prompt reaches the LLM.

## The concept

In many systems, multiple teams share a knowledge base but have different data access requirements. Instead of maintaining separate copies of the data for each role, you store documents once and apply role-specific PII tokenization at query time.

The privacy boundary is not in the vector store -- it is at the LLM boundary. Each role maps to a different set of Blindfold entities (or a built-in policy), and the tokenization happens after retrieval but before the LLM call.

## Healthcare use case

This example uses patient medical records with four roles:

| Role | Sees | Redacted |
|------|------|----------|
| **Doctor** | Names, conditions, medications, DOBs | Email, phone, SSN, financial data |
| **Nurse** | Names, conditions, medications | Email, phone, SSN, DOB, financial data |
| **Billing** | Names, insurance info, financial data | Email, phone, SSN, conditions, medications |
| **Researcher** | Nothing identifiable (fully de-identified) | All PII -- names, contact info, medical data, etc. |

## How it works

```
                          Patient Records
                                |
                     blindfold.redact(contact info)
                                |
                       ChromaDB vector store
                         (shared by all roles)
                                |
             +---------+---------+---------+
             |         |         |         |
          Doctor     Nurse    Billing  Researcher
             |         |         |         |
          tokenize  tokenize  tokenize  tokenize
          (emails,  (emails,  (emails,  policy=
           phones,   phones,   phones,  "strict"
           SSNs,     SSNs,     SSNs,    (all 60+
           finance)  DOBs,     medical  entities)
                     finance)  data)
             |         |         |         |
             +----+----+----+----+         |
                  |              |          |
              OpenAI LLM    OpenAI LLM  OpenAI LLM
                  |              |          |
           blindfold.detokenize()          |
                  |              |          |
             Final answer   Final answer  Final answer
             (PII restored  (PII restored (de-identified)
              for user)      for user)
```

## How policies map to roles

Each role has a Blindfold policy that controls what gets tokenized:

- **Doctor** (`role_doctor`): Tokenizes contact info and financial data. Clinical data passes through so the LLM can reason about conditions and medications.
- **Nurse** (`role_nurse`): Same as doctor plus DOB. Nurses need treatment info but not date of birth or financial details.
- **Billing** (`role_billing`): Tokenizes contact info, SSN, and clinical data (conditions, medications). Billing staff need insurance and financial info only.
- **Researcher** (`policy="strict"`): Uses the built-in strict policy which tokenizes all 60+ entity types. The LLM sees only de-identified content.

In this example, the doctor, nurse, and billing roles use `entities=[...]` so no dashboard setup is needed. In production, you would create custom policies via the Blindfold dashboard and use `policy="role_doctor"` etc.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
python main.py
```

Run all roles (default):

```bash
python main.py
```

Run a single role:

```bash
python main.py --role doctor
python main.py --role researcher
```

## What to look for in the output

1. **Same question, different answers**: The query "What conditions has Sarah Chen been treated for?" produces different output depending on the role. A doctor sees the full clinical answer. A researcher sees `<Person_1> was treated for <Medical_Condition_1>`.

2. **Tokenized prompts**: For the first query, the output shows what each role sends to the LLM. Compare how much PII each role strips out.

3. **Billing vs. clinical split**: Ask about insurance -- billing sees it, but medical conditions are redacted. Ask about conditions -- doctors see it, but billing does not.

4. **Researcher de-identification**: The `policy="strict"` role strips everything. Names become `<Person_N>`, conditions become `<Medical_Condition_N>`, etc. The LLM can still answer content-based questions using the structure of the text.

## Example output

```
Question: "What conditions has Sarah Chen been treated for?"

    [doctor]       Final answer: "Sarah Chen was diagnosed with Type 2 Diabetes..."
    [nurse]        Final answer: "Sarah Chen was diagnosed with Type 2 Diabetes..."
    [billing]      Final answer: "Sarah Chen had records related to..."
    [researcher]   Final answer: "<Person_1> was treated for <Medical_Condition_1>..."
```

## Resources

- [Blindfold Policies Documentation](https://docs.blindfold.dev/policies)
- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python-sdk)
- [Strategy Comparison Example](../rag-strategy-comparison-python/)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
