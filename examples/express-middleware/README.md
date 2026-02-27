# Express.js Middleware

Express middleware that automatically tokenizes PII in incoming request bodies and provides a `detokenize` helper for responses.

## How it works

1. Middleware intercepts `req.body.message` (configurable field)
2. Tokenizes PII and replaces the body field with the safe version
3. Attaches `req.blindfold.detokenize()` for restoring values in responses

## Setup

```bash
npm install
cp ../../.env.example .env
# Edit .env with your API keys
```

## Run

```bash
npm start
```

## Test

```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Email john@example.com about his order"}'
```

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
