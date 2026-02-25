/**
 * RAG with Stored Mapping (Strategy B) -- TypeScript
 *
 * Every ticket is fully tokenized at ingestion, and the token-to-real-value
 * mapping is stored alongside each document. At query time, a reverse lookup
 * replaces known names in the question with their tokens, the tokenized
 * query is used to search, and mappings from retrieved documents are merged
 * for detokenization.
 *
 * Key trade-off: the same person gets DIFFERENT tokens in each document
 * because Blindfold generates fresh tokens per call. "Hans Mueller" might be
 * <Person_1> in ticket A but <Person_3> in ticket B. This makes cross-document
 * search harder but keeps zero PII in the vector store.
 *
 * Flow:
 *   1. Ingest: blindfold.tokenize(ticket) -> store tokenized text + mapping per doc
 *   2. Query:  build reverse lookup, replace names with tokens, search
 *   3. Merge:  combine mappings from all retrieved documents
 *   4. LLM:    generate answer from tokenized context
 *   5. Restore: blindfold.detokenize(response, mergedMapping)
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

// In-memory mapping store. In production, use a database.
type MappingStore = Record<string, Record<string, string>>;

// ---------------------------------------------------------------------------
// Ingestion: Tokenize everything, store mappings per document
// ---------------------------------------------------------------------------

async function ingestTickets(chroma: ChromaClient) {
  const collection = await chroma.getOrCreateCollection({
    name: "stored_mapping",
  });
  const mappingStore: MappingStore = {};

  console.log("=== Ingesting Support Tickets (Stored Mapping) ===\n");
  console.log("Tokenizing all PII, storing mapping per document...\n");

  for (let idx = 0; idx < SUPPORT_TICKETS.length; idx++) {
    const result = await blindfold.tokenize(SUPPORT_TICKETS[idx]);
    const docId = `ticket-${idx}`;

    await collection.add({
      documents: [result.text],
      ids: [docId],
      metadatas: [{ mapping: JSON.stringify(result.mapping) }],
    });
    mappingStore[docId] = result.mapping;

    const tokenCount = Object.keys(result.mapping).length;
    console.log(`  Ticket ${idx + 1}: ${tokenCount} tokens created`);
    console.log(`    Stored: "${result.text.slice(0, 90)}..."`);

    // Show the mapping for this document
    for (const [token, realValue] of Object.entries(result.mapping)) {
      console.log(`      ${token} = "${realValue}"`);
    }
  }

  console.log(`\nStored ${SUPPORT_TICKETS.length} tickets in ChromaDB`);

  // Highlight the inconsistency problem
  console.log("\n  NOTE: Same person may have DIFFERENT tokens across documents.");
  console.log("  Check the mappings above -- Marie Dupont appears in tickets 2 and 4");
  console.log("  but likely has different token IDs in each.\n");

  return { collection, mappingStore };
}

// ---------------------------------------------------------------------------
// Query: Reverse lookup, search, merge mappings, LLM, detokenize
// ---------------------------------------------------------------------------

async function queryRAG(
  collection: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>,
  mappingStore: MappingStore,
  question: string
): Promise<string> {
  // Step 1: Build reverse lookup (real value -> list of [docId, token] pairs)
  const reverseLookup: Record<string, Array<[string, string]>> = {};
  for (const [docId, mapping] of Object.entries(mappingStore)) {
    for (const [token, realValue] of Object.entries(mapping)) {
      if (!reverseLookup[realValue]) reverseLookup[realValue] = [];
      reverseLookup[realValue].push([docId, token]);
    }
  }

  console.log(
    `  Reverse lookup has ${Object.keys(reverseLookup).length} unique real values`
  );

  // Step 2: Replace real names in query with their tokens
  let tokenizedQuery = question;
  const matchedTokens: string[] = [];
  for (const [realValue, entries] of Object.entries(reverseLookup)) {
    if (tokenizedQuery.includes(realValue)) {
      // Use the first token found for this real value
      const token = entries[0][1];
      tokenizedQuery = tokenizedQuery.split(realValue).join(token);
      matchedTokens.push(`${realValue} -> ${token}`);
    }
  }

  if (matchedTokens.length > 0) {
    console.log(`  Replaced in query: ${matchedTokens.join(", ")}`);
  } else {
    console.log("  No known values found in query (content-based search only)");
  }
  console.log(`  Search query: "${tokenizedQuery}"`);

  // Step 3: Search -- tokens match tokens in vector store
  const results = await collection.query({
    queryTexts: [tokenizedQuery],
    nResults: 2,
  });
  const docs = results.documents[0] as string[];
  const context = docs.join("\n\n");

  console.log(`\n  Retrieved ${docs.length} chunks:`);
  for (const doc of docs) {
    console.log(`    "${doc.slice(0, 80)}..."`);
  }

  // Step 4: Merge mappings from all retrieved documents
  const mergedMapping: Record<string, string> = {};
  for (const docId of results.ids[0]) {
    if (mappingStore[docId]) {
      Object.assign(mergedMapping, mappingStore[docId]);
    }
  }
  console.log(
    `\n  Merged ${Object.keys(mergedMapping).length} tokens from ${results.ids[0].length} documents`
  );

  // Step 5: LLM call -- context and query are already tokenized
  const promptText = `Context:\n${context}\n\nQuestion: ${tokenizedQuery}`;
  console.log(`  Prompt to LLM preview: "${promptText.slice(0, 120)}..."`);

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: promptText },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;
  console.log(`\n  LLM response (tokenized): "${aiResponse}"`);

  // Step 6: Detokenize using merged mappings
  const final = blindfold.detokenize(aiResponse, mergedMapping);
  return final.text;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const chroma = new ChromaClient();
  const { collection, mappingStore } = await ingestTickets(chroma);

  console.log("=== Running Queries ===\n");

  for (const question of QUERIES) {
    console.log(`Question: "${question}"\n`);
    const answer = await queryRAG(collection, mappingStore, question);
    console.log(`\n  Final answer: "${answer}"`);
    console.log("-".repeat(60) + "\n");
  }

  console.log("=== How Stored Mapping Works ===\n");
  console.log("  1. Every ticket is fully tokenized -- zero PII in the vector store");
  console.log("  2. Token-to-value mapping is stored alongside each document");
  console.log("  3. At query time, a reverse lookup replaces names with tokens");
  console.log("  4. Mappings from retrieved documents are merged for detokenization");
  console.log("  5. The LLM never sees real PII\n");
  console.log("  Pros: Zero PII in vector store, fully reversible");
  console.log("  Cons: Same person gets different tokens per document");
  console.log("        Reverse lookup has collisions as corpus grows");
  console.log("        Requires per-document mapping storage\n");
}

main();
