/**
 * RAG with Selective Redaction (Strategy A) -- TypeScript
 *
 * The simplest PII protection strategy for RAG pipelines.
 *
 * At ingestion, contact information (emails, phone numbers) is permanently
 * redacted while names are kept for searchability. At query time, the
 * retrieved context and question are tokenized in a single call before
 * being sent to the LLM, then the response is detokenized for the user.
 *
 * Flow:
 *   1. Ingest: blindfold.redact(ticket, { entities: ["email address", "phone number"] })
 *   2. Query:  search with original question (names match directly)
 *   3. Protect: blindfold.tokenize(context + question) before the LLM
 *   4. LLM:    generate answer from tokenized prompt
 *   5. Restore: blindfold.detokenize(response, mapping)
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

const blindfold = new Blindfold({ apiKey: process.env.BLINDFOLD_API_KEY! });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

const SUPPORT_TICKETS = [
  "Ticket #1001: Customer Hans Mueller (hans.mueller@example.de, +49 151 12345678) reported a billing error on invoice INV-2024-0047. He was charged twice for the Pro plan in January. Refund requested.",
  "Ticket #1002: Marie Dupont (marie.dupont@example.fr, +33 6 12 34 56 78) cannot access her dashboard after a password reset. She tried three times and is now locked out. Needs urgent unlock.",
  "Ticket #1003: Lars Johansson (lars.johansson@example.se, +46 70 123 4567) asked to export all his personal data under GDPR. He wants a full copy within 30 days as required by regulation.",
  "Ticket #1004: Marie Dupont (marie.dupont@example.fr, +33 6 12 34 56 78) reports a second issue — her subscription was downgraded without notice. She expected Pro features but only has Basic.",
];

const QUERIES = [
  "What was the issue reported by Hans Mueller?",
  "What problems did Marie Dupont have?",
  "Which tickets involved billing issues?",
];

const SYSTEM_PROMPT =
  "You are a helpful support assistant. Answer the user's question based only on the provided context. Keep your answer concise.";

// ---------------------------------------------------------------------------
// Ingestion: Redact contact info, keep names
// ---------------------------------------------------------------------------

async function ingestTickets(chroma: ChromaClient) {
  const collection = await chroma.getOrCreateCollection({
    name: "selective_redact",
  });

  console.log("=== Ingesting Support Tickets (Selective Redact) ===\n");
  console.log("Redacting emails and phone numbers, keeping names for search...\n");

  for (let idx = 0; idx < SUPPORT_TICKETS.length; idx++) {
    const result = await blindfold.redact(SUPPORT_TICKETS[idx], {
      entities: ["email address", "phone number"],
    });

    await collection.add({
      documents: [result.text],
      ids: [`ticket-${idx}`],
    });

    console.log(`  Ticket ${idx + 1}: ${result.entities_count} entities redacted`);
    console.log(`    Stored: "${result.text.slice(0, 90)}..."`);
  }

  console.log(`\nStored ${SUPPORT_TICKETS.length} tickets in ChromaDB\n`);
  return collection;
}

// ---------------------------------------------------------------------------
// Query: Search, tokenize, LLM, detokenize
// ---------------------------------------------------------------------------

async function queryRAG(
  collection: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>,
  question: string
): Promise<string> {
  // Step 1: Search with original question -- names match directly
  console.log(`  Searching for relevant tickets...`);
  const results = await collection.query({
    queryTexts: [question],
    nResults: 2,
  });
  const docs = results.documents[0] as string[];
  const context = docs.join("\n\n");

  console.log(`  Retrieved ${docs.length} chunks:`);
  for (const doc of docs) {
    console.log(`    "${doc.slice(0, 80)}..."`);
  }

  // Step 2: Tokenize the combined context + question
  const promptText = `Context:\n${context}\n\nQuestion: ${question}`;
  const tokenized = await blindfold.tokenize(promptText);

  console.log(`\n  Tokenized ${tokenized.entities_count} entities before LLM call`);
  console.log(`  Tokenized prompt preview: "${tokenized.text.slice(0, 120)}..."`);

  // Step 3: LLM call -- no PII in the prompt
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: tokenized.text },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;
  console.log(`\n  LLM response (tokenized): "${aiResponse}"`);

  // Step 4: Detokenize -- restore real values for the user
  const final = blindfold.detokenize(aiResponse, tokenized.mapping);
  return final.text;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const chroma = new ChromaClient();
  const collection = await ingestTickets(chroma);

  console.log("=== Running Queries ===\n");

  for (const question of QUERIES) {
    console.log(`Question: "${question}"\n`);
    const answer = await queryRAG(collection, question);
    console.log(`\n  Final answer: "${answer}"`);
    console.log("-".repeat(60) + "\n");
  }

  console.log("=== How Selective Redact Works ===\n");
  console.log("  1. Emails and phone numbers are permanently redacted at ingestion");
  console.log("  2. Names stay in the vector store for direct search matching");
  console.log("  3. At query time, context+question is tokenized before the LLM");
  console.log("  4. The LLM never sees real PII -- only tokens");
  console.log("  5. The response is detokenized before showing to the user\n");
  console.log("  Pros: Simple, stateless, no mapping storage needed");
  console.log("  Cons: Names visible in vector store (internal infra risk)");
  console.log("        Contact info permanently lost\n");
}

main();
