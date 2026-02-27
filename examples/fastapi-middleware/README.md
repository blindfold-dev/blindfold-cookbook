# FastAPI Middleware

FastAPI middleware that automatically tokenizes PII in incoming request bodies and stores the mapping on `request.state` for detokenization.

## How it works

1. Middleware intercepts POST request body
2. Tokenizes PII in the configured text field
3. Stores mapping on `request.state.blindfold` for downstream route handlers

## Setup

```bash
pip install -r requirements.txt
cp ../../.env.example .env
# Edit .env with your API keys
```

## Run

```bash
uvicorn main:app --reload
```

## Test

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Email john@example.com about his order"}'
```

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.

Local mode limitations:
- NLP-only entities (Person, Organization, Medical Condition) are not detected
- Custom dashboard policies are not available (built-in policies work)
