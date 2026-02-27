# GDPR Customer Support RAG Pipeline — TypeScript

Multi-turn GDPR-compliant customer support chatbot with EU sample tickets in German, French, Spanish, and English. Uses the `gdpr_eu` policy and EU region.

At ingestion, contact info (emails, phones, IBANs, national IDs, addresses) is redacted while names are kept for searchability. At query time, the original question is used for retrieval, then context + question are tokenized together before the LLM.

## What this example shows

- **`gdpr_eu` policy** — detects EU-relevant PII: emails, phones, IBANs, national IDs
- **Selective redaction** — contact info redacted, names kept for vector search
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

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
