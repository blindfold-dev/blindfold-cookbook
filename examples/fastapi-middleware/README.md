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
