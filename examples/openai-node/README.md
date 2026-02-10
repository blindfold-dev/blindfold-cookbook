# OpenAI + Blindfold (TypeScript)

Protect PII in OpenAI chat conversations. User messages are tokenized before reaching OpenAI, and responses are detokenized to restore original values.

## How it works

1. **Tokenize** user message — `John Smith` becomes `<Person_1>`
2. **Send** tokenized text to OpenAI — no real PII leaves your system
3. **Detokenize** AI response — `<Person_1>` becomes `John Smith` again

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
