# Blindfold + E2B: PII-Safe AI Data Analyst

An AI data analyst that writes analysis code **without ever seeing real personal data**. Blindfold tokenizes PII before OpenAI sees the dataset, then E2B executes the AI-generated code on the original data in an isolated sandbox.

## How it works

```
Original CSV (with PII)          Tokenized CSV (safe)              AI-generated code
┌─────────────────────┐    ┌─────────────────────────┐    ┌──────────────────────────┐
│ John Smith, 45,      │    │ <Person_1>, 45,          │    │ import pandas as pd      │
│ john@email.com,      │ →  │ <Email Address_1>,       │ →  │ df = pd.read_csv(...)    │
│ Type 2 Diabetes      │    │ Type 2 Diabetes          │    │ df.groupby('diagnosis')  │
└─────────────────────┘    └─────────────────────────┘    └──────────────────────────┘
         │                    Blindfold tokenizes            OpenAI writes code
         │                                                   (only sees tokens)
         │
         ▼
┌──────────────────────────┐
│  E2B Sandbox             │
│  Runs AI code on the     │    →    Analysis results
│  ORIGINAL data           │         (accurate, real values)
└──────────────────────────┘
```

1. **Tokenize** — Blindfold replaces PII (names, emails, SSNs) with safe tokens
2. **Generate code** — OpenAI writes pandas analysis code based on the tokenized CSV. It sees `<Person_1>` and `<Email Address_1>`, never real data
3. **Execute** — E2B runs the AI-generated code in an isolated sandbox with the original data, producing accurate results

## Why this matters

- **OpenAI never sees PII** — it only needs column structure to write correct pandas code
- **E2B isolates execution** — AI-generated code runs in a sandboxed environment
- **Results are accurate** — analysis runs on real data, not tokens
- **Double protection** — PII tokenization + sandbox isolation

## What this example shows

- Tokenizing CSV data with the `strict` policy (catches all entity types)
- Sending tokenized data to OpenAI for code generation
- Executing AI-generated code in E2B with original data
- Multi-step analysis: initial exploration → follow-up questions

## Quick start

```bash
cd examples/e2b-data-analyst
pip install -r requirements.txt
cp .env.example .env
# Add your BLINDFOLD_API_KEY, E2B_API_KEY, and OPENAI_API_KEY to .env
python main.py
```

Get API keys:
- **Blindfold**: [app.blindfold.dev](https://app.blindfold.dev) (free tier available)
- **E2B**: [e2b.dev/dashboard](https://e2b.dev/dashboard?tab=keys)
- **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys)

## Example output

```
============================================================
Example 1: PII-Safe Data Analysis
============================================================

[Step 1] Tokenizing CSV with Blindfold...
  Detected 24 PII entities:
    - Person: John Smith (confidence: 85%)
    - Email Address: john.smith@email.com (confidence: 95%)
    - Social Security Number: 123-45-6789 (confidence: 90%)
    ...

  Tokenized CSV (this is what OpenAI sees):
    name,email,ssn,age,diagnosis,medication,doctor,city
    <Person_1>,<Email Address_1>,<Social Security Number_1>,45,Type 2 Diabetes,...
    <Person_2>,<Email Address_2>,<Social Security Number_2>,32,Hypertension,...
    ...

[Step 2] Asking OpenAI to write analysis code...
  (OpenAI only sees tokens — never real names, emails, or SSNs)

[Step 3] Executing code in E2B sandbox with real data...
  (Sandbox runs on original CSV — results are accurate)

  Analysis results:
    Patients per diagnosis:
      Type 2 Diabetes    3
      Hypertension       2
      Asthma             2
      Migraine           1
    ...
```

## Resources

- [Blindfold Documentation](https://docs.blindfold.dev)
- [E2B Documentation](https://e2b.dev/docs)
- [E2B Code Interpreter](https://github.com/e2b-dev/code-interpreter)
- [Blindfold Python SDK](https://pypi.org/project/blindfold-sdk/)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
