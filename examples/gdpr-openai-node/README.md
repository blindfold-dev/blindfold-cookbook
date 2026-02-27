# GDPR-Compliant OpenAI Integration (TypeScript)

Tokenize EU user data before it reaches OpenAI using Blindfold's `gdpr_eu` policy and EU region. The LLM never sees real names, emails, IBANs, or addresses.

## How it works

1. **Tokenize** — `gdpr_eu` policy detects EU-relevant PII (names, emails, IBANs, addresses, DOB)
2. **Send to OpenAI** — only tokenized text crosses the border
3. **Detokenize** — restore original values in the AI response

## Quick start

```bash
cd examples/gdpr-openai-node
npm install
cp .env.example .env
# Add your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## Resources

- [GDPR Compliance Guide](https://docs.blindfold.dev/compliance/gdpr)
- [EU Region Documentation](https://docs.blindfold.dev/essentials/regions)
- [Blindfold Node.js SDK](https://www.npmjs.com/package/@blindfold/sdk)

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
