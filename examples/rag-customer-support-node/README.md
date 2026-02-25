# GDPR Customer Support RAG Pipeline — TypeScript

Multi-turn GDPR-compliant customer support chatbot with EU sample tickets in German, French, Spanish, and English. Uses the `gdpr_eu` policy and EU region.

## What this example shows

- **`gdpr_eu` policy** — detects EU-relevant PII: names, emails, phones, IBANs, national IDs
- **EU region** — `Blindfold({ region: "eu" })` ensures PII processing stays in Europe
- **Multi-turn conversation** — mapping accumulates across turns for consistent detokenization
- **Multilingual PII** — German, French, Spanish, and English support tickets
- **`CustomerSupportRAG` class** — production-ready pattern with ingestion and query methods

## Quick start

```bash
cd examples/rag-customer-support-node
npm install
cp .env.example .env
# Add your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## GDPR compliance highlights

- **No PII leaves the EU** — tokenization happens in the EU region
- **Data minimization** (Art. 5) — only anonymized tokens sent to the LLM
- **Multilingual detection** — PII detected across German, French, Spanish, and English

## Resources

- [Blindfold GDPR Guide](https://docs.blindfold.dev/compliance/gdpr)
- [EU Region Documentation](https://docs.blindfold.dev/essentials/regions)
- [RAG Pipeline Protection Guide](https://docs.blindfold.dev/rag)
