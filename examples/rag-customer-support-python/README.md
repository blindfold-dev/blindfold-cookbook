# GDPR Customer Support RAG Pipeline

Real-world GDPR-compliant customer support chatbot with multi-turn conversation. Processes EU support tickets in German, French, Spanish, and English using the `gdpr_eu` policy and EU region.

## What this example shows

- **`gdpr_eu` policy** — detects EU-relevant PII: names, emails, phones, IBANs, national IDs, addresses
- **EU region** — `Blindfold(region="eu")` ensures PII processing stays in Europe
- **Multi-turn conversation** — mapping accumulates across turns for consistent detokenization
- **Multilingual PII** — German, French, Spanish, and English support tickets
- **`CustomerSupportRAG` class** — production-ready pattern with ingestion and query methods

## How it works

```
Ingestion (EU region):
  EU tickets (DE/FR/ES/EN) → Redact with gdpr_eu → ChromaDB

Multi-turn query:
  Turn 1: "Hans Mueller's issue?"  → tokenize → retrieve → LLM → detokenize
  Turn 2: "Marie Dupont's request?" → tokenize → retrieve → LLM → detokenize
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
  Ticket 1: 4 entities redacted ['Person', 'Email Address', 'Phone Number', 'Iban Code']
  Ticket 2: 3 entities redacted ['Person', 'Email Address', 'Phone Number']
  Ticket 3: 4 entities redacted ['Person', 'Email Address', 'Phone Number', 'National Id']
  Ticket 4: 3 entities redacted ['Person', 'Email Address', 'Phone Number']
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
