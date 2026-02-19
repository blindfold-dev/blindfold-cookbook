# LangChain.js + Blindfold: PII-Safe Chains (TypeScript)

Protect PII in LangChain.js chains using Blindfold. The `blindfoldProtect()` helper returns a tokenizer and detokenizer that compose with the pipe operator.

## How it works

```
tokenize → prompt → LLM → detokenize
```

1. **Tokenize** — Blindfold replaces PII with tokens before the prompt
2. **LLM processes** — OpenAI only sees `<Person_1>`, `<Email Address_1>`
3. **Detokenize** — original values restored in the final response

## Quick start

```bash
cd examples/langchain-node
npm install
cp .env.example .env
# Add your BLINDFOLD_API_KEY and OPENAI_API_KEY
npm start
```

## Resources

- [Blindfold Documentation](https://docs.blindfold.dev)
- [LangChain.js Documentation](https://js.langchain.com)
- [Blindfold Node.js SDK](https://www.npmjs.com/package/@blindfold/sdk)
