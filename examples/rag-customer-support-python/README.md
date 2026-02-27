# GDPR Customer Support RAG Pipeline

Real-world GDPR-compliant customer support chatbot with multi-turn conversation. Processes EU support tickets in German, French, Spanish, and English using the `gdpr_eu` policy and EU region.

At ingestion, contact info (emails, phones, IBANs, national IDs, addresses) is redacted while names are kept for searchability. At query time, the original question is used for retrieval, then context + question are tokenized together before the LLM.

## What this example shows

- **`gdpr_eu` policy** — detects EU-relevant PII: emails, phones, IBANs, national IDs, addresses
- **Selective redaction** — contact info redacted, names kept for vector search
- **EU region** — `Blindfold(region="eu")` ensures PII processing stays in Europe
- **Multi-turn conversation** — mapping accumulates across turns for consistent detokenization
- **Multilingual PII** — German, French, Spanish, and English support tickets
- **`CustomerSupportRAG` class** — production-ready pattern with ingestion and query methods

## How it works

```
Ingestion (EU region):
  EU tickets (DE/FR/ES/EN) → Redact contact info with gdpr_eu (keep names) → ChromaDB

Multi-turn query:
  Turn 1: "Hans Mueller's issue?" → search (names match!) → tokenize(context+question) → LLM → detokenize
  Turn 2: "Marie Dupont's request?" → search (names match!) → tokenize(context+question) → LLM → detokenize
          (mapping accumulates across turns)
```

## GDPR compliance highlights

- **No PII leaves the EU** — tokenization happens in the EU region before data goes to OpenAI
- **Data minimization** (Art. 5) — only anonymized tokens are sent to the LLM
- **Right to access** (Art. 15) — data export requests handled in sample tickets
- **Multilingual detection** — PII detected across German, French, Spanish, and English

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
python main.py
```

## Example output

```
=== Ingesting EU Support Tickets ===
  Ticket 1: 3 entities redacted ['Email Address', 'Phone Number', 'Iban Code']
  Ticket 2: 3 entities redacted ['Email Address', 'Phone Number', 'Address']
  Ticket 3: 3 entities redacted ['Email Address', 'Phone Number', 'National Id']
  Ticket 4: 2 entities redacted ['Email Address', 'Phone Number']
Stored 4 chunks in ChromaDB

=== Multi-Turn Customer Support Chat ===

Customer: What happened with Hans Mueller's billing issue?
Agent: Hans Mueller reported a billing error where their account was
charged twice on 2026-01-20. A refund has been initiated.

Customer: Did Marie Dupont's data export request get resolved?
Agent: Yes, Marie Dupont's GDPR Art. 15 data export request was
completed and sent to her email address.
```

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
