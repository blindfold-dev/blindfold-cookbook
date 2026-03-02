# OpenAI + Blindfold (.NET)

Protect PII in OpenAI chat conversations. User messages are tokenized before reaching OpenAI, and responses are detokenized to restore original values.

## How it works

1. **Tokenize** user message — `John Smith` becomes `<Person_1>`
2. **Send** tokenized text to OpenAI — no real PII leaves your system
3. **Detokenize** AI response — `<Person_1>` becomes `John Smith` again

## Prerequisites

- .NET 6.0+ SDK

## Setup

```bash
dotnet restore
cp ../../../.env.example .env
# Edit .env with your API keys
```

## Run

```bash
dotnet run
```

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.
