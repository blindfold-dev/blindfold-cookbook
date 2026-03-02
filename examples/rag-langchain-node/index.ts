/**
 * LangChain RAG Pipeline with Blindfold PII Protection — TypeScript
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
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda } from "@langchain/core/runnables";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// API key is optional — omit it to run in local mode (regex-based, offline)
const blindfold = new Blindfold({
  apiKey: process.env.BLINDFOLD_API_KEY,
});

// Inline document transformer — redacts contact info, keeps names
async function transformDocuments(docs: Document[]): Promise<Document[]> {
  const safeDocs: Document[] = [];
  for (const doc of docs) {
    const result = await blindfold.redact(doc.pageContent, {
      entities: ["email address", "phone number"],
    });
    safeDocs.push(
      new Document({
        pageContent: result.text,
        metadata: { ...doc.metadata, entities_redacted: result.entities_count },
      })
    );
  }
  return safeDocs;
}

const SUPPORT_TICKETS = [
  new Document({
    pageContent:
      "Ticket #1001: Customer Sarah Chen (sarah.chen@acme.com, +1-555-234-5678) reported a billing discrepancy. Account was charged $49.99 twice on 2026-01-15. Resolution: refund issued within 24 hours.",
    metadata: { source: "tickets.csv", row: 1 },
  }),
  new Document({
    pageContent:
      "Ticket #1002: James Rivera (james.rivera@example.com) reported intermittent API timeouts since 2026-02-01. Root cause: regional DNS misconfiguration. Fixed within 4 hours, SLA credit applied.",
    metadata: { source: "tickets.csv", row: 2 },
  }),
  new Document({
    pageContent:
      "Ticket #1003: Maria Garcia (maria.garcia@example.es, +34 612 345 678) requested a GDPR data export under Art. 15. Export generated and sent to her email within 30 days.",
    metadata: { source: "tickets.csv", row: 3 },
  }),
  new Document({
    pageContent:
      "Ticket #1004: Emma Johnson (emma.johnson@example.co.uk, +44 20 7946 0958) could not access dashboard after password reset. Issue traced to browser cache conflict. Resolved by clearing session cookies.",
    metadata: { source: "tickets.csv", row: 4 },
  }),
];

async function main() {
  // === Ingestion ===
  console.log("=== Document Ingestion ===");

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const chunks = await splitter.splitDocuments(SUPPORT_TICKETS);

  const safeDocs = await transformDocuments(chunks);
  for (const doc of safeDocs) {
    console.log(`  Redacted ${doc.metadata.entities_redacted} entities`);
  }

  const vectorStore = await MemoryVectorStore.fromDocuments(
    safeDocs,
    new OpenAIEmbeddings()
  );
  const retriever = vectorStore.asRetriever({ k: 3 });
  console.log(`Stored ${safeDocs.length} documents\n`);

  // === Query ===
  console.log("=== RAG Query ===");

  const llm = new ChatOpenAI({ modelName: "gpt-4o-mini" });

  const formatDocs = (docs: Document[]) =>
    docs.map((d) => d.pageContent).join("\n\n");

  // Retrieve-then-tokenize chain
  const chain = new RunnableLambda({
    func: async (question: string) => {
      // Step 1: Retrieve with original question — names match
      const docs = await retriever.invoke(question);
      const context = formatDocs(docs);

      // Step 2: Single tokenize call — consistent token numbering
      const promptText = `Context:\n${context}\n\nQuestion: ${question}`;
      const tokenized = await blindfold.tokenize(promptText);

      // Step 3: Split tokenized prompt for the LLM template
      const parts = tokenized.text.split("\n\nQuestion: ");
      const tokenizedContext = parts[0].replace("Context:\n", "");
      const tokenizedQuestion = parts[1] || tokenized.text;

      const messages = [
        {
          role: "system" as const,
          content: `Answer the question using only the context below.\n\nContext:\n${tokenizedContext}`,
        },
        { role: "user" as const, content: tokenizedQuestion },
      ];

      const response = await llm.invoke(messages);
      const aiText = await new StringOutputParser().invoke(response);

      // Step 4: Detokenize to restore real names
      const final = blindfold.detokenize(aiText, tokenized.mapping);
      return final.text;
    },
  });

  const question = "What was Sarah Chen's issue?";
  console.log(`Question: ${question}`);

  const answer = await chain.invoke(question);
  console.log(`Answer: ${answer}`);
}

main();
