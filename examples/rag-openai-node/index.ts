/**
 * RAG Pipeline with PII Protection (OpenAI + ChromaDB) — TypeScript
 *
 * Redacts PII at ingestion time and tokenizes user queries at
 * query time. The LLM never sees real personal data.
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

const blindfold = new Blindfold({
  apiKey: process.env.BLINDFOLD_API_KEY!,
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
    const result = await blindfold.redact(SUPPORT_TICKETS[i]);
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

  // Tokenize the question
  const tokenized = await blindfold.tokenize(question);
  console.log(`Tokenized: ${tokenized.text}\n`);

  // Retrieve relevant chunks
  const results = await collection.query({
    queryTexts: [tokenized.text],
    nResults: 3,
  });
  const context = results.documents[0].join("\n\n");

  // Send to LLM with redacted context + tokenized question
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are a helpful support assistant. Answer using only this context:\n\n${context}`,
      },
      { role: "user", content: tokenized.text },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;

  // Detokenize to restore real names
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
