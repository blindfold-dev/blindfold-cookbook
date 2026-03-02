/**
 * RAG Pipeline with PII Protection (OpenAI + ChromaDB) — TypeScript
 *
 * Redacts contact info at ingestion time (keeps names for searchability).
 * At query time, searches with the original question, then tokenizes
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
import OpenAI from "openai";
import { ChromaClient } from "chromadb";

// API key is optional — omit it to run in local mode (regex-based, offline)
const blindfold = new Blindfold({
  apiKey: process.env.BLINDFOLD_API_KEY,
});
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

const SUPPORT_TICKETS = [
  "Ticket #1001: Customer John Smith (john.smith@example.com, +1-555-0123) reported a billing discrepancy. Account was charged $49.99 twice on 2026-01-15. Resolution: refund issued within 24 hours.",
  "Ticket #1002: Maria Garcia (maria.garcia@example.es, +34 612 345 678) requested a data export under GDPR Art. 15. Export was generated and sent to her email within the required 30-day period.",
  "Ticket #1003: Customer Li Wei (li.wei@example.cn, +86 138 0013 8000) reported intermittent API timeouts since 2026-02-01. Root cause: regional DNS misconfiguration. Fixed within 4 hours.",
  "Ticket #1004: Emma Johnson (emma.johnson@example.co.uk, +44 20 7946 0958) could not access her dashboard after a password reset. Issue traced to a browser cache conflict. Resolved by clearing session cookies.",
];

async function ingest(collection: any) {
  console.log("=== Ingestion ===");
  const safeDocuments: string[] = [];
  const ids: string[] = [];

  for (let i = 0; i < SUPPORT_TICKETS.length; i++) {
    // Redact contact info only — keep names searchable
    const result = await blindfold.redact(SUPPORT_TICKETS[i], {
      entities: ["email address", "phone number"],
    });
    console.log(`  Ticket ${i + 1}: ${result.entities_count} entities redacted`);
    safeDocuments.push(result.text);
    ids.push(`ticket-${i}`);
  }

  await collection.add({ documents: safeDocuments, ids });
  console.log(`Stored ${safeDocuments.length} documents in ChromaDB\n`);
}

async function query(collection: any, question: string) {
  console.log("=== Query ===");
  console.log(`Question: ${question}`);

  // Step 1: Search with original question — names match in vector store
  const results = await collection.query({
    queryTexts: [question],
    nResults: 3,
  });
  const context = results.documents[0].join("\n\n");

  // Step 2: Single tokenize call — consistent token numbering
  const promptText = `Context:\n${context}\n\nQuestion: ${question}`;
  const tokenized = await blindfold.tokenize(promptText);
  console.log(`Tokenized prompt (preview): ${tokenized.text.slice(0, 120)}...\n`);

  // Step 3: Send to LLM — no PII in the prompt
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content:
          "You are a helpful support assistant. Answer the user's question based only on the provided context. Keep your answer concise.",
      },
      { role: "user", content: tokenized.text },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;

  // Step 4: Detokenize to restore real names
  const final = blindfold.detokenize(aiResponse, tokenized.mapping);
  console.log(`Answer: ${final.text}`);
}

async function main() {
  const chroma = new ChromaClient();
  const collection = await chroma.getOrCreateCollection({
    name: "support_tickets",
  });

  await ingest(collection);
  await query(collection, "What happened with John Smith's billing issue?");
}

main();
