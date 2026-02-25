/**
 * LlamaIndex RAG Pipeline with Blindfold PII Protection — TypeScript
 *
 * Documents are redacted at ingestion, and user queries are tokenized
 * before reaching the LLM.
 *
 * Usage:
 *   npm install
 *   cp .env.example .env  # add your API keys
 *   npm start
 */

import "dotenv/config";
import { Blindfold } from "@blindfold/sdk";
import {
  Document,
  VectorStoreIndex,
  Settings,
  OpenAI,
  OpenAIEmbedding,
} from "llamaindex";

const blindfold = new Blindfold({
  apiKey: process.env.BLINDFOLD_API_KEY!,
});

const SUPPORT_TICKETS = [
  "Ticket #1001: Customer John Smith (john.smith@example.com, +1-555-0123) reported a billing discrepancy. Account was charged $49.99 twice on 2026-01-15. Resolution: refund issued within 24 hours.",
  "Ticket #1002: Maria Garcia (maria.garcia@example.es, +34 612 345 678) requested a data export under GDPR Art. 15. Export was generated and sent to her email within the required 30-day period.",
  "Ticket #1003: Customer Li Wei (li.wei@example.cn, +86 138 0013 8000) reported intermittent API timeouts since 2026-02-01. Root cause: regional DNS misconfiguration. Fixed within 4 hours.",
  "Ticket #1004: Emma Johnson (emma.johnson@example.co.uk, +44 20 7946 0958) could not access her dashboard after a password reset. Issue traced to a browser cache conflict. Resolved by clearing session cookies.",
];

async function main() {
  // Configure LlamaIndex
  Settings.llm = new OpenAI({
    model: "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY!,
  });
  Settings.embedModel = new OpenAIEmbedding({
    apiKey: process.env.OPENAI_API_KEY!,
  });

  // === Ingestion: redact PII before indexing ===
  console.log("=== Ingestion ===");
  const documents: Document[] = [];

  for (let i = 0; i < SUPPORT_TICKETS.length; i++) {
    const result = await blindfold.redact(SUPPORT_TICKETS[i]);
    console.log(`  Ticket ${i + 1}: ${result.entities_count} entities redacted`);
    documents.push(new Document({ text: result.text }));
  }

  const index = await VectorStoreIndex.fromDocuments(documents);
  console.log(`Indexed ${documents.length} documents\n`);

  // === Query: tokenize question, query, detokenize response ===
  console.log("=== Query ===");
  const question = "What happened with John Smith's billing issue?";

  const tokenized = await blindfold.tokenize(question);
  console.log(`Original: ${question}`);
  console.log(`Tokenized: ${tokenized.text}\n`);

  const queryEngine = index.asQueryEngine();
  const response = await queryEngine.query({ query: tokenized.text });

  const final = blindfold.detokenize(
    response.message.content as string,
    tokenized.mapping
  );
  console.log(`Answer: ${final.text}`);
}

main();
