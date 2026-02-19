# Blindfold Cookbook

Example code and guides for [Blindfold](https://blindfold.dev) — the privacy API that protects PII in AI applications.

Each example is a self-contained project you can clone and run. They show how to integrate Blindfold with popular LLM providers and frameworks.

**Resources:** [Documentation](https://docs.blindfold.dev) | [Python SDK](https://pypi.org/project/blindfold-sdk/) | [Node.js SDK](https://www.npmjs.com/package/@blindfold/sdk) | [CLI](https://www.npmjs.com/package/@blindfold/cli) | [MCP Server](https://www.npmjs.com/package/@blindfold/mcp-server)

## Prerequisites

1. Get a free API key at [app.blindfold.dev](https://app.blindfold.dev)
2. Copy `.env.example` to `.env` in the example directory
3. Add your `BLINDFOLD_API_KEY` (and `OPENAI_API_KEY` where needed)

## LLM Providers

Examples showing how to protect PII when calling LLM APIs.

<table>
<thead>
<tr>
  <th>Provider</th>
  <th>Description</th>
  <th>Python</th>
  <th>TypeScript</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>OpenAI</b></td>
  <td>Tokenize user messages before GPT, detokenize responses</td>
  <td><a href="examples/openai-python">openai-python</a></td>
  <td><a href="examples/openai-node">openai-node</a></td>
</tr>
<tr>
  <td><b>LangChain</b></td>
  <td>PII-safe chains with RunnableLambda</td>
  <td><a href="examples/langchain-python">langchain-python</a></td>
  <td>—</td>
</tr>
</tbody>
</table>

## Compliance

End-to-end examples for building regulation-compliant AI applications.

<table>
<thead>
<tr>
  <th>Regulation</th>
  <th>Description</th>
  <th>Region</th>
  <th>Example</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>GDPR</b></td>
  <td>EU-compliant OpenAI integration with <code>gdpr_eu</code> policy, batch processing</td>
  <td>EU</td>
  <td><a href="examples/gdpr-openai-python">gdpr-openai-python</a></td>
</tr>
<tr>
  <td><b>HIPAA</b></td>
  <td>Healthcare chatbot protecting PHI with <code>hipaa_us</code> policy, multi-turn chat</td>
  <td>US</td>
  <td><a href="examples/hipaa-healthcare-chatbot">hipaa-healthcare-chatbot</a></td>
</tr>
</tbody>
</table>

## Framework Integrations

Middleware that automatically protects PII in web applications.

<table>
<thead>
<tr>
  <th>Framework</th>
  <th>Description</th>
  <th>Python</th>
  <th>TypeScript</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>Express.js</b></td>
  <td>Middleware that tokenizes request bodies, detokenizes responses</td>
  <td>—</td>
  <td><a href="examples/express-middleware">express-middleware</a></td>
</tr>
<tr>
  <td><b>FastAPI</b></td>
  <td>Middleware that tokenizes request bodies, stores mapping on request.state</td>
  <td><a href="examples/fastapi-middleware">fastapi-middleware</a></td>
  <td>—</td>
</tr>
</tbody>
</table>

## AI Agents

Examples showing how to safely give AI agents access to sensitive data.

<table>
<thead>
<tr>
  <th>Platform</th>
  <th>Description</th>
  <th>Example</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>E2B</b></td>
  <td>AI data analyst — OpenAI writes code from tokenized CSV, E2B executes on real data</td>
  <td><a href="examples/e2b-data-analyst">e2b-data-analyst</a></td>
</tr>
</tbody>
</table>

## How Blindfold works

```
User message                    Tokenized message               AI response
"Email john@example.com"  →  "Email <Email Address_1>"  →  "I'll email <Email Address_1>"
                                                              ↓
                                                         "I'll email john@example.com"
```

1. **Tokenize** — PII is replaced with safe tokens before data leaves your system
2. **Process** — The AI model only sees anonymized text
3. **Detokenize** — Original values are restored in the response

## Contributing

Contributions are welcome! If you have an integration example to share, open a pull request.

## License

MIT
