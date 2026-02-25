/**
 * RAG Strategy Comparison: Three PII Protection Approaches (TypeScript)
 *
 * Runs the SAME support tickets and queries through three strategies
 * side by side so you can see how each handles ingestion, search, and
 * LLM interaction.
 *
 * Strategy A - Selective Redact (recommended):
 *   Ingestion: redact contact info (emails, phones), keep names
 *   Query: search with original question, single tokenize(context+question), LLM, detokenize
 *
 * Strategy B - Tokenize with Stored Mapping:
 *   Ingestion: tokenize everything, store mapping per document
 *   Query: reverse-lookup names->tokens, search with tokenized query, LLM, detokenize
 *   Problem: same person gets different tokens in each document
 *
 * Strategy C - Consistent Token Registry:
 *   Ingestion: use Blindfold to detect PII, then replace with consistent
 *     tokens from a global registry ("Hans Mueller" = <Person_1> everywhere)
 *   Query: lookup names in registry, search with consistent tokens, LLM, reverse replace
 *   Best searchability + zero PII in vector store, but more complex
 *
 * Usage:
 *   npm install
 *   cp .env.example .env  # add your API keys
 *   npm start             # run all three strategies
 *   npm start c           # run only Strategy C
 *   npm start a c         # run Strategy A and C
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
// Strategy A: Selective Redact
// ---------------------------------------------------------------------------

async function strategyAIngest(chroma: ChromaClient) {
  const collection = await chroma.getOrCreateCollection({
    name: "strategy_a",
  });

  console.log("  Ingesting tickets (redact emails + phones, keep names)...");
  for (let idx = 0; idx < SUPPORT_TICKETS.length; idx++) {
    const result = await blindfold.redact(SUPPORT_TICKETS[idx], {
      entities: ["email address", "phone number"],
    });
    await collection.add({
      documents: [result.text],
      ids: [`ticket-${idx}`],
    });
    const preview = result.text.slice(0, 80) + "...";
    console.log(`    Ticket ${idx + 1}: ${result.entities_count} entities redacted`);
    console.log(`      Stored: "${preview}"`);
  }

  return collection;
}

async function strategyAQuery(
  collection: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>,
  question: string
): Promise<string> {
  // Search -- names match because they are in the vector store
  const results = await collection.query({
    queryTexts: [question],
    nResults: 2,
  });
  const context = (results.documents[0] as string[]).join("\n\n");

  console.log(`  Retrieved ${results.documents[0].length} chunks`);
  for (const doc of results.documents[0] as string[]) {
    console.log(`    "${doc.slice(0, 80)}..."`);
  }

  // Single tokenize call on context + question
  const promptText = `Context:\n${context}\n\nQuestion: ${question}`;
  const tokenized = await blindfold.tokenize(promptText);
  console.log(`\n  Tokenized prompt (first 120 chars):`);
  console.log(`    "${tokenized.text.slice(0, 120)}..."`);

  // LLM call
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: tokenized.text },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;
  console.log(`\n  LLM response (tokenized): "${aiResponse}"`);

  // Detokenize
  const final = blindfold.detokenize(aiResponse, tokenized.mapping);
  return final.text;
}

// ---------------------------------------------------------------------------
// Strategy B: Tokenize with Stored Mapping (per-document)
// ---------------------------------------------------------------------------

async function strategyBIngest(chroma: ChromaClient) {
  const collection = await chroma.getOrCreateCollection({
    name: "strategy_b",
  });
  const mappingStore: Record<string, Record<string, string>> = {};

  console.log("  Ingesting tickets (tokenize everything, store mappings)...");
  for (let idx = 0; idx < SUPPORT_TICKETS.length; idx++) {
    const result = await blindfold.tokenize(SUPPORT_TICKETS[idx]);
    const docId = `ticket-${idx}`;
    await collection.add({
      documents: [result.text],
      ids: [docId],
      metadatas: [{ mapping: JSON.stringify(result.mapping) }],
    });
    mappingStore[docId] = result.mapping;
    const preview = result.text.slice(0, 80) + "...";
    console.log(`    Ticket ${idx + 1}: ${Object.keys(result.mapping).length} tokens created`);
    console.log(`      Stored: "${preview}"`);
  }

  return { collection, mappingStore };
}

async function strategyBQuery(
  collection: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>,
  mappingStore: Record<string, Record<string, string>>,
  question: string
): Promise<string> {
  // Build reverse lookup: real value -> list of (docId, token) pairs
  const reverseLookup: Record<string, Array<[string, string]>> = {};
  for (const [docId, mapping] of Object.entries(mappingStore)) {
    for (const [token, realValue] of Object.entries(mapping)) {
      if (!reverseLookup[realValue]) reverseLookup[realValue] = [];
      reverseLookup[realValue].push([docId, token]);
    }
  }

  console.log(`  Reverse lookup has ${Object.keys(reverseLookup).length} unique real values`);

  // Replace real names in query with their tokens
  let tokenizedQuery = question;
  const matchedTokens: string[] = [];
  for (const [realValue, entries] of Object.entries(reverseLookup)) {
    if (tokenizedQuery.includes(realValue)) {
      // Use the first token we find for this real value
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

  // Search -- tokens match tokens in vector store
  const results = await collection.query({
    queryTexts: [tokenizedQuery],
    nResults: 2,
  });
  const context = (results.documents[0] as string[]).join("\n\n");

  console.log(`\n  Retrieved ${results.documents[0].length} chunks`);
  for (const doc of results.documents[0] as string[]) {
    console.log(`    "${doc.slice(0, 80)}..."`);
  }

  // Merge mappings from all retrieved documents
  const mergedMapping: Record<string, string> = {};
  for (const docId of results.ids[0]) {
    if (mappingStore[docId]) {
      Object.assign(mergedMapping, mappingStore[docId]);
    }
  }

  // LLM call -- already tokenized context and query
  const promptText = `Context:\n${context}\n\nQuestion: ${tokenizedQuery}`;
  console.log(`\n  Prompt to LLM (first 120 chars):`);
  console.log(`    "${promptText.slice(0, 120)}..."`);

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: promptText },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;
  console.log(`\n  LLM response (tokenized): "${aiResponse}"`);

  // Detokenize using merged mappings
  const final = blindfold.detokenize(aiResponse, mergedMapping);
  return final.text;
}

// ---------------------------------------------------------------------------
// Strategy C: Consistent Token Registry
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

  replaceInText(text: string): string {
    let result = text;
    // Sort by length descending so "Marie Dupont" is replaced before "Marie"
    const sorted = [...this.registry.entries()].sort(
      (a, b) => b[0].length - a[0].length
    );
    for (const [realValue, token] of sorted) {
      result = result.split(realValue).join(token);
    }
    return result;
  }

  restoreText(text: string): string {
    let result = text;
    // Sort by length descending to avoid partial replacements
    const sorted = [...this.reverse.entries()].sort(
      (a, b) => b[0].length - a[0].length
    );
    for (const [token, realValue] of sorted) {
      result = result.split(token).join(realValue);
    }
    return result;
  }
}

async function strategyCIngest(chroma: ChromaClient) {
  const collection = await chroma.getOrCreateCollection({
    name: "strategy_c",
  });
  const registry = new TokenRegistry();

  console.log("  Ingesting tickets (detect PII, apply consistent tokens)...");
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
    const preview = consistentText.slice(0, 80) + "...";
    console.log(
      `    Ticket ${idx + 1}: ${Object.keys(result.mapping).length} entities detected`
    );
    console.log(`      Stored: "${preview}"`);
  }

  console.log(`\n  Registry (${registry.registry.size} entries):`);
  for (const [realValue, token] of registry.registry) {
    console.log(`    ${token} = "${realValue}"`);
  }

  return { collection, registry };
}

async function strategyCQuery(
  collection: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>,
  registry: TokenRegistry,
  question: string
): Promise<string> {
  // Replace any known real values in the question with their tokens
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

  // Search -- consistent tokens match consistently in vector store
  const results = await collection.query({
    queryTexts: [tokenizedQuery],
    nResults: 2,
  });
  const context = (results.documents[0] as string[]).join("\n\n");

  console.log(`\n  Retrieved ${results.documents[0].length} chunks`);
  for (const doc of results.documents[0] as string[]) {
    console.log(`    "${doc.slice(0, 80)}..."`);
  }

  // LLM call -- all PII replaced with consistent tokens
  const promptText = `Context:\n${context}\n\nQuestion: ${tokenizedQuery}`;
  console.log(`\n  Prompt to LLM (first 120 chars):`);
  console.log(`    "${promptText.slice(0, 120)}..."`);

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: promptText },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;
  console.log(`\n  LLM response (tokenized): "${aiResponse}"`);

  // Restore real values using the registry
  const final = registry.restoreText(aiResponse);
  return final;
}

// ---------------------------------------------------------------------------
// Runner functions
// ---------------------------------------------------------------------------

async function runStrategyA(chroma: ChromaClient) {
  console.log("=".repeat(70));
  console.log("STRATEGY A: Selective Redact");
  console.log("  Contact info redacted, names kept in vector store");
  console.log("=".repeat(70));
  console.log();

  const collection = await strategyAIngest(chroma);
  console.log();

  for (const question of QUERIES) {
    console.log(`  Question: "${question}"`);
    const answer = await strategyAQuery(collection, question);
    console.log(`\n  Final answer: "${answer}"`);
    console.log("-".repeat(50));
  }
}

async function runStrategyB(chroma: ChromaClient) {
  console.log("=".repeat(70));
  console.log("STRATEGY B: Tokenize with Stored Mapping (per-document)");
  console.log("  Everything tokenized, mappings stored per document");
  console.log("  NOTE: same person gets DIFFERENT tokens in each document");
  console.log("=".repeat(70));
  console.log();

  const { collection, mappingStore } = await strategyBIngest(chroma);
  console.log();

  for (const question of QUERIES) {
    console.log(`  Question: "${question}"`);
    const answer = await strategyBQuery(collection, mappingStore, question);
    console.log(`\n  Final answer: "${answer}"`);
    console.log("-".repeat(50));
  }
}

async function runStrategyC(chroma: ChromaClient) {
  console.log("=".repeat(70));
  console.log("STRATEGY C: Consistent Token Registry");
  console.log("  Blindfold detects PII, app assigns consistent tokens");
  console.log("  Same person = same token in EVERY document");
  console.log("=".repeat(70));
  console.log();

  const { collection, registry } = await strategyCIngest(chroma);
  console.log();

  for (const question of QUERIES) {
    console.log(`  Question: "${question}"`);
    const answer = await strategyCQuery(collection, registry, question);
    console.log(`\n  Final answer: "${answer}"`);
    console.log("-".repeat(50));
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const STRATEGY_MAP: Record<string, (chroma: ChromaClient) => Promise<void>> = {
  a: runStrategyA,
  b: runStrategyB,
  c: runStrategyC,
};

async function main() {
  const args = process.argv.slice(2);
  const strategies =
    args.length > 0 ? args.map((a) => a.toLowerCase()) : ["a", "b", "c"];

  // Validate strategy arguments
  for (const s of strategies) {
    if (!STRATEGY_MAP[s]) {
      console.error(
        `Unknown strategy "${s}". Valid options: a, b, c`
      );
      process.exit(1);
    }
  }

  const chroma = new ChromaClient();

  for (let i = 0; i < strategies.length; i++) {
    if (i > 0) console.log();
    await STRATEGY_MAP[strategies[i]](chroma);
  }

  if (strategies.length > 1) {
    console.log();
    console.log("=".repeat(70));
    console.log("SUMMARY");
    console.log("=".repeat(70));
    console.log(`
  Strategy A (Selective Redact):
    + Simple, stateless -- no mapping storage needed
    + Each document ingested independently
    + Names match directly in vector search
    - Names are stored in the vector store (internal infra risk)
    - Contact info permanently lost

  Strategy B (Tokenize with Stored Mapping):
    + Zero PII in vector store -- tokens only
    + Fully reversible -- can restore all original data
    - Same person gets DIFFERENT tokens per document (<Person_1> in doc A,
      <Person_3> in doc B) -- reverse lookup has collisions
    - Requires per-document mapping storage
    - Grows in complexity as documents increase

  Strategy C (Consistent Token Registry):
    + Zero PII in vector store -- tokens only
    + Same person = same token EVERYWHERE (<Person_1> always = Hans Mueller)
    + Perfect vector search -- tokens match exactly across all documents
    + Simple reverse lookup -- one token per real value
    + Fully reversible
    - Requires a persistent token registry (DB)
    - Uses Blindfold as PII detector, does replacement itself
    - detokenize() not used -- app does its own string replacement
    - Registry must be available at both ingestion and query time

  All three strategies protect PII at the LLM boundary -- the external
  third party (OpenAI, Anthropic, etc.) never sees real personal data.
`);
  }
}

main();
