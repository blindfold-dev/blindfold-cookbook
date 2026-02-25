/**
 * LangChain RAG Pipeline with Blindfold PII Protection — TypeScript
 *
 * Implements inline blindfoldProtect() and document transformer since
 * there is no langchain-blindfold JS package. Documents are redacted
 * at ingestion, queries are tokenized at runtime.
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
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableLambda,
} from "@langchain/core/runnables";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const blindfold = new Blindfold({
  apiKey: process.env.BLINDFOLD_API_KEY!,
});

// Inline document transformer (equivalent to BlindfoldPIITransformer in Python)
async function transformDocuments(docs: Document[]): Promise<Document[]> {
  const safeDocs: Document[] = [];
  for (const doc of docs) {
    const result = await blindfold.redact(doc.pageContent);
    safeDocs.push(
      new Document({
        pageContent: result.text,
        metadata: { ...doc.metadata, entities_redacted: result.entities_count },
      })
    );
  }
  return safeDocs;
}

// Inline blindfold_protect (equivalent to Python langchain-blindfold)
function blindfoldProtect(policy: string = "basic") {
  const mappingStore: Record<string, string> = {};

  const tokenize = new RunnableLambda({
    func: async (text: string) => {
      const result = await blindfold.tokenize(text, { policy });
      Object.assign(mappingStore, result.mapping);
      return result.text;
    },
  });

  const detokenize = new RunnableLambda({
    func: async (text: string) => {
      const result = blindfold.detokenize(text, mappingStore);
      for (const key of Object.keys(mappingStore)) {
        delete mappingStore[key];
      }
      return result.text;
    },
  });

  return { tokenize, detokenize };
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

  const { tokenize, detokenize } = blindfoldProtect("basic");

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the question using only the context below.\n\nContext:\n{context}",
    ],
    ["human", "{question}"],
  ]);

  const llm = new ChatOpenAI({ modelName: "gpt-4o-mini" });

  const formatDocs = (docs: Document[]) =>
    docs.map((d) => d.pageContent).join("\n\n");

  const chain = tokenize
    .pipe(
      RunnablePassthrough.assign({
        context: new RunnableLambda({
          func: async (question: string) => {
            const docs = await retriever.invoke(question);
            return formatDocs(docs);
          },
        }),
        question: new RunnableLambda({ func: async (q: string) => q }),
      })
    )
    .pipe(prompt)
    .pipe(llm)
    .pipe(new StringOutputParser())
    .pipe(detokenize);

  const question = "What was Sarah Chen's issue?";
  console.log(`Question: ${question}`);

  const answer = await chain.invoke(question);
  console.log(`Answer: ${answer}`);
}

main();
