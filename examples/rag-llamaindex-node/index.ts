/**
 * LlamaIndex RAG Pipeline with Blindfold PII Protection — TypeScript
 *
 * Contact info is redacted at ingestion (names kept for searchability).
 * At query time, retrieves with the original question, then tokenizes
 * context + question in a single call before the LLM.
 *
 * Works in two modes:
 *   - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
 *   - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations
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

// API key is optional — omit it to run in local mode (regex-based, offline)
const blindfold = new Blindfold({
  apiKey: process.env.BLINDFOLD_API_KEY,
});

const SUPPORT_TICKETS = [
  "Ticket #1001: Customer John Smith (john.smith@example.com, +1-555-0123) reported a billing discrepancy. Account was charged $49.99 twice on 2026-01-15. Resolution: refund issued within 24 hours.",
  "Ticket #1002: Maria Garcia (maria.garcia@example.es, +34 612 345 678) requested a data export under GDPR Art. 15. Export was generated and sent to her email within the required 30-day period.",
  "Ticket #1003: Customer Li Wei (li.wei@example.cn, +86 138 0013 8000) reported intermittent API timeouts since 2026-02-01. Root cause: regional DNS misconfiguration. Fixed within 4 hours.",
  "Ticket #1004: Emma Johnson (emma.johnson@example.co.uk, +44 20 7946 0958) could not access her dashboard after a password reset. Issue traced to a browser cache conflict. Resolved by clearing session cookies.",
];

async function main() {
  // Configure LlamaIndex
  const llm = new OpenAI({
    model: "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY!,
  });
  Settings.llm = llm;
  Settings.embedModel = new OpenAIEmbedding({
    apiKey: process.env.OPENAI_API_KEY!,
  });

  // === Ingestion: redact contact info, keep names searchable ===
  console.log("=== Ingestion ===");
  const documents: Document[] = [];

  for (let i = 0; i < SUPPORT_TICKETS.length; i++) {
    const result = await blindfold.redact(SUPPORT_TICKETS[i], {
      entities: ["email address", "phone number"],
    });
    console.log(`  Ticket ${i + 1}: ${result.entities_count} entities redacted`);
    documents.push(new Document({ text: result.text }));
  }

  const index = await VectorStoreIndex.fromDocuments(documents);
  console.log(`Indexed ${documents.length} documents\n`);

  // === Query: retrieve first, single tokenize call, then LLM ===
  console.log("=== Query ===");
  const question = "What happened with John Smith's billing issue?";
  console.log(`Original: ${question}\n`);

  // Step 1: Retrieve with original question — names match in vector store
  const retriever = index.asRetriever({ similarityTopK: 3 });
  const nodes = await retriever.retrieve(question);
  const context = nodes.map((n) => n.node.getText()).join("\n\n");

  // Step 2: Single tokenize call — consistent token numbering
  const promptText = `Context:\n${context}\n\nQuestion: ${question}`;
  const tokenized = await blindfold.tokenize(promptText);

  // Step 3: LLM call with tokenized prompt — no PII
  const response = await llm.complete({
    prompt:
      "You are a helpful support assistant. Answer the user's question " +
      "based only on the provided context. Keep your answer concise.\n\n" +
      tokenized.text,
  });
  const aiResponse = response.text;
  console.log(`AI response (with tokens): ${aiResponse}\n`);

  // Step 4: Detokenize to restore real names
  const final = blindfold.detokenize(aiResponse, tokenized.mapping);
  console.log(`Answer: ${final.text}`);
}

main();
