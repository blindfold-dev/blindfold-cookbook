/**
 * RAG with Consistent Token Registry (Strategy C) -- TypeScript
 *
 * The most sophisticated PII protection strategy for RAG pipelines.
 * Uses Blindfold to detect PII entities, then replaces them with consistent
 * tokens from a global registry. "Hans Mueller" is always <Person_1>,
 * regardless of which document or when it was ingested.
 *
 * This gives zero PII in the vector store AND perfect cross-document
 * search because the same entity always maps to the same token.
 *
 * Flow:
 *   1. Ingest: blindfold.tokenize(ticket) to detect entities
 *   2. Register: registry.getOrCreate(realValue, blindfoldToken) for each entity
 *   3. Replace: registry.replaceInText(originalTicket) with consistent tokens
 *   4. Store: add consistent text to ChromaDB
 *   5. Query: registry.replaceInText(question), search, LLM, registry.restoreText()
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

const blindfold = new Blindfold({ apiKey: process.env.BLINDFOLD_API_KEY });
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
// Token Registry: Assigns consistent tokens across all documents
// ---------------------------------------------------------------------------

class TokenRegistry {
  registry: Map<string, string> = new Map(); // real_value -> token
  reverse: Map<string, string> = new Map(); // token -> real_value
  private counters: Map<string, number> = new Map();

  private extractEntityType(token: string): string {
    const inner = token.replace(/^<|>$/g, "");
    const lastUnderscore = inner.lastIndexOf("_");
    if (lastUnderscore > 0 && /^\d+$/.test(inner.slice(lastUnderscore + 1))) {
      return inner.slice(0, lastUnderscore);
    }
    return inner;
  }

  /**
   * Get the consistent token for a real value, or create a new one.
   * If "Hans Mueller" was already registered as <Person_1>, returns <Person_1>.
   * Otherwise, creates the next available token for that entity type.
   */
  getOrCreate(realValue: string, blindfoldToken: string): string {
    if (this.registry.has(realValue)) return this.registry.get(realValue)!;

    const entityType = this.extractEntityType(blindfoldToken);
    const count = (this.counters.get(entityType) || 0) + 1;
    this.counters.set(entityType, count);
    const consistentToken = `<${entityType}_${count}>`;

    this.registry.set(realValue, consistentToken);
    this.reverse.set(consistentToken, realValue);
    return consistentToken;
  }

  /**
   * Replace all known real values in text with their consistent tokens.
   * Sorts by length descending so "Marie Dupont" is replaced before "Marie".
   */
  replaceInText(text: string): string {
    let result = text;
    const sorted = [...this.registry.entries()].sort(
      (a, b) => b[0].length - a[0].length
    );
    for (const [realValue, token] of sorted) {
      result = result.split(realValue).join(token);
    }
    return result;
  }

  /**
   * Replace all consistent tokens with real values (detokenize).
   * Sorts by length descending to avoid partial replacements.
   */
  restoreText(text: string): string {
    let result = text;
    const sorted = [...this.reverse.entries()].sort(
      (a, b) => b[0].length - a[0].length
    );
    for (const [token, realValue] of sorted) {
      result = result.split(token).join(realValue);
    }
    return result;
  }
}

// ---------------------------------------------------------------------------
// Ingestion: Detect PII with Blindfold, apply consistent tokens
// ---------------------------------------------------------------------------

async function ingestTickets(chroma: ChromaClient, registry: TokenRegistry) {
  const collection = await chroma.getOrCreateCollection({
    name: "consistent_registry",
  });

  console.log("=== Ingesting Support Tickets (Consistent Registry) ===\n");
  console.log("Detecting PII with Blindfold, replacing with consistent tokens...\n");

  for (let idx = 0; idx < SUPPORT_TICKETS.length; idx++) {
    // Step 1: Use Blindfold to detect entities
    const result = await blindfold.tokenize(SUPPORT_TICKETS[idx]);

    // Step 2: Register each entity with a consistent token
    for (const [bfToken, realValue] of Object.entries(result.mapping)) {
      registry.getOrCreate(realValue, bfToken);
    }

    // Step 3: Replace PII in the ORIGINAL text with consistent tokens
    const consistentText = registry.replaceInText(SUPPORT_TICKETS[idx]);

    await collection.add({
      documents: [consistentText],
      ids: [`ticket-${idx}`],
    });

    const entityCount = Object.keys(result.mapping).length;
    console.log(`  Ticket ${idx + 1}: ${entityCount} entities detected`);
    console.log(`    Stored: "${consistentText.slice(0, 90)}..."`);
  }

  // Print the global registry
  console.log(`\n  Registry (${registry.registry.size} entries):`);
  for (const [realValue, token] of registry.registry) {
    console.log(`    ${token} = "${realValue}"`);
  }

  console.log(`\nStored ${SUPPORT_TICKETS.length} tickets in ChromaDB\n`);
  return collection;
}

// ---------------------------------------------------------------------------
// Query: Replace names with consistent tokens, search, LLM, restore
// ---------------------------------------------------------------------------

async function queryRAG(
  collection: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>,
  registry: TokenRegistry,
  question: string
): Promise<string> {
  // Step 1: Replace any known real values in the question with their tokens
  const tokenizedQuery = registry.replaceInText(question);

  const replacements: string[] = [];
  for (const [realValue, token] of registry.registry) {
    if (question.includes(realValue)) {
      replacements.push(`${realValue} -> ${token}`);
    }
  }

  if (replacements.length > 0) {
    console.log(`  Replaced in query: ${replacements.join(", ")}`);
  } else {
    console.log("  No known values found in query (content-based search only)");
  }
  console.log(`  Search query: "${tokenizedQuery}"`);

  // Step 2: Search -- consistent tokens match consistently in vector store
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

  // Step 3: LLM call -- all PII replaced with consistent tokens
  const promptText = `Context:\n${context}\n\nQuestion: ${tokenizedQuery}`;
  console.log(`\n  Prompt to LLM preview: "${promptText.slice(0, 120)}..."`);

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: promptText },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;
  console.log(`\n  LLM response (tokenized): "${aiResponse}"`);

  // Step 4: Restore real values using the registry
  const final = registry.restoreText(aiResponse);
  return final;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const chroma = new ChromaClient();
  const registry = new TokenRegistry();
  const collection = await ingestTickets(chroma, registry);

  console.log("=== Running Queries ===\n");

  for (const question of QUERIES) {
    console.log(`Question: "${question}"\n`);
    const answer = await queryRAG(collection, registry, question);
    console.log(`\n  Final answer: "${answer}"`);
    console.log("-".repeat(60) + "\n");
  }

  console.log("=== How Consistent Registry Works ===\n");
  console.log("  1. Blindfold detects PII entities in each ticket");
  console.log("  2. Each unique entity gets a consistent token from the registry");
  console.log('     "Hans Mueller" = <Person_1> in EVERY document');
  console.log('     "Marie Dupont" = <Person_2> in EVERY document');
  console.log("  3. The original text is replaced with consistent tokens");
  console.log("  4. At query time, names in the question are replaced via the registry");
  console.log("  5. Vector search matches perfectly across all documents");
  console.log("  6. The LLM response is restored using the registry\n");
  console.log("  Pros: Zero PII in vector store, perfect cross-document search,");
  console.log("        fully reversible, simple reverse lookup");
  console.log("  Cons: Requires a persistent registry (database in production),");
  console.log("        registry must be available at both ingestion and query time\n");
}

main();
