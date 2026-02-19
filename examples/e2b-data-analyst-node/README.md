# Blindfold + E2B: PII-Safe AI Data Analyst (TypeScript)

An AI data analyst that writes analysis code **without ever seeing real personal data**. Blindfold tokenizes PII before OpenAI sees the dataset, then E2B executes the AI-generated code on the original data in an isolated sandbox.

## How it works

1. **Tokenize** — Blindfold replaces PII with safe tokens in the CSV
2. **Generate code** — OpenAI writes pandas code based on tokenized data (only sees `<Person_1>`, never real names)
3. **Execute** — E2B runs the code in a sandbox with the original data for accurate results

## Quick start

```bash
cd examples/e2b-data-analyst-node
npm install
cp .env.example .env
# Add your BLINDFOLD_API_KEY, E2B_API_KEY, and OPENAI_API_KEY
npm start
```

## Resources

- [Blindfold Documentation](https://docs.blindfold.dev)
- [E2B Documentation](https://e2b.dev/docs)
- [Blindfold Node.js SDK](https://www.npmjs.com/package/@blindfold/sdk)
