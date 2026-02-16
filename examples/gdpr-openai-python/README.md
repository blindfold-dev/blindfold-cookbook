# GDPR-Compliant OpenAI Integration

Process EU user data through OpenAI while staying GDPR-compliant. PII is tokenized in Blindfold's EU region before it reaches OpenAI — the LLM never sees real personal data.

## What this example shows

- **EU region** — `Blindfold(region="eu")` ensures PII processing stays in the EU
- **`gdpr_eu` policy** — detects EU-relevant entities: names, emails, phones, IBANs, addresses, dates of birth
- **Tokenize → LLM → Detokenize** — the standard Blindfold flow for safe AI interactions
- **Batch processing** — tokenize multiple support tickets in a single API call

## How it works

```
EU User Data                    Blindfold EU Region              OpenAI (US)
"Hans Mueller,                  "<Person_1>,                     AI processes only
 hans@example.de,        →      <Email Address_1>,        →     anonymized tokens
 IBAN DE89..."                   <Iban Code_1>..."

                                                                      ↓

"Dear Hans Mueller,       ←     Detokenize with mapping    ←    "Dear <Person_1>,
 we've updated your              (PII never left EU)              we've updated your
 account..."                                                      account..."
```

## GDPR compliance highlights

- **No PII leaves the EU** — tokenization happens in the EU region before data goes to OpenAI
- **Data minimization** (Art. 5) — only anonymized tokens are sent to the LLM
- **Audit trail** — every tokenization is logged for DPA compliance
- **Data residency** — Blindfold's EU node processes and stores data in Europe

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your BLINDFOLD_API_KEY and OPENAI_API_KEY
python main.py
```

## Example output

```
User message:
  "Hi, my name is Hans Mueller and I need help with my subscription..."

  Detected 5 PII entities
    - Person: Hans Mueller (confidence: 95%)
    - Email Address: hans.mueller@example.de (confidence: 99%)
    - Phone Number: +49 170 1234567 (confidence: 97%)
    - Iban Code: DE89 3704 0044 0532 0130 00 (confidence: 98%)
    - Date Of Birth: 15/03/1985 (confidence: 92%)
  Tokenized text sent to OpenAI:
    "Hi, my name is <Person_1> and I need help with my subscription.
     My email is <Email Address_1>, phone <Phone Number_1>..."

  Final response (PII restored):
    "Dear Hans Mueller, I've located your account..."
```

## Resources

- [Blindfold GDPR Guide](https://docs.blindfold.dev/compliance/gdpr)
- [EU Region Documentation](https://docs.blindfold.dev/regions)
- [Python SDK Reference](https://docs.blindfold.dev/sdks/python)
