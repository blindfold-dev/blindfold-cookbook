# HIPAA-Compliant Healthcare Chatbot (TypeScript)

A healthcare chatbot that tokenizes PHI before it reaches OpenAI using Blindfold's `hipaa_us` policy and US region. Supports multi-turn conversations with accumulated PHI mapping.

## How it works

1. **Tokenize** — `hipaa_us` policy detects all 18 HIPAA Safe Harbor identifiers
2. **Send to OpenAI** — only tokens like `<Person_1>`, `<Ssn_1>` reach the model
3. **Detokenize** — restore real PHI for authorized clinicians

## Quick start

```bash
cd examples/hipaa-healthcare-chatbot-node
npm install
cp .env.example .env
# Add your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## Resources

- [HIPAA Compliance Guide](https://docs.blindfold.dev/compliance/hipaa)
- [US Region Documentation](https://docs.blindfold.dev/essentials/regions)
- [Blindfold Node.js SDK](https://www.npmjs.com/package/@blindfold/sdk)
