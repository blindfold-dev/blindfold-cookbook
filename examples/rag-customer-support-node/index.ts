/**
 * GDPR-Compliant Customer Support RAG Pipeline — TypeScript
 *
 * Multi-turn customer support chatbot with EU sample tickets.
 * Uses the gdpr_eu policy and EU region for GDPR compliance.
 *
 * At ingestion, contact info is redacted (names kept for searchability).
 * At query time, retrieves with the original question, then tokenizes
 * context + question in a single call before the LLM. Mapping
 * accumulates across turns for consistent detokenization.
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

// EU support tickets with German, French, Spanish PII
const EU_SUPPORT_TICKETS = [
  "Ticket #EU-2001: Kunde Hans Mueller (hans.mueller@example.de, +49 170 9876543) meldet einen Abrechnungsfehler. Konto wurde am 2026-01-20 doppelt belastet. IBAN: DE89 3704 0044 0532 0130 00. Status: Ruckerstattung eingeleitet.",
  "Ticket #EU-2002: Cliente Marie Dupont (marie.dupont@example.fr, +33 6 12 34 56 78) demande un export de donnees conformement a l'article 15 du RGPD. Adresse: 42 Rue de Rivoli, 75001 Paris. Status: Export envoye par email.",
  "Ticket #EU-2003: Cliente Sofia Garcia (sofia.garcia@example.es, +34 612 345 678) reporta que no puede acceder a su panel de control desde el 2026-02-01. DNI: 12345678Z. Status: Problema resuelto, conflicto de cache.",
  "Ticket #EU-2004: Customer Emma Wilson (emma.wilson@example.co.uk, +44 20 7946 0958) reports subscription downgrade not reflected in billing. Previous charge: EUR 29.99, expected: EUR 14.99. Card ending 4821. Status: Billing adjusted, partial refund issued.",
];

class CustomerSupportRAG {
  private blindfold: Blindfold;
  private openai: OpenAI;
  private collection: any;
  private conversationHistory: { role: string; content: string }[] = [];
  private accumulatedMapping: Record<string, string> = {};

  constructor() {
    this.blindfold = new Blindfold({
      apiKey: process.env.BLINDFOLD_API_KEY,
      region: "eu",
    });
    this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
  }

  async initialize() {
    const chroma = new ChromaClient();
    this.collection = await chroma.getOrCreateCollection({
      name: "eu_support_tickets",
    });
  }

  async ingestTickets(tickets: string[]) {
    console.log("=== Ingesting EU Support Tickets ===");
    const safeDocuments: string[] = [];
    const ids: string[] = [];

    for (let i = 0; i < tickets.length; i++) {
      // Redact contact info — keep names searchable
      const result = await this.blindfold.redact(tickets[i], {
        policy: "gdpr_eu",
        entities: [
          "email address",
          "phone number",
          "iban",
          "credit card number",
          "address",
          "date of birth",
          "national id number",
        ],
      });
      const entityTypes = result.detected_entities.map((e: any) => e.type);
      console.log(
        `  Ticket ${i + 1}: ${result.entities_count} entities redacted [${entityTypes.join(", ")}]`
      );
      safeDocuments.push(result.text);
      ids.push(`eu-ticket-${i}`);
    }

    await this.collection.add({ documents: safeDocuments, ids });
    console.log(`Stored ${safeDocuments.length} chunks in ChromaDB\n`);
  }

  async query(question: string): Promise<string> {
    // Step 1: Search with original question — names match in vector store
    const results = await this.collection.query({
      queryTexts: [question],
      nResults: 3,
    });
    const context = results.documents[0].join("\n\n");

    // Step 2: Single tokenize call — consistent token numbering
    const promptText = `Context:\n${context}\n\nQuestion: ${question}`;
    const tokenized = await this.blindfold.tokenize(promptText, {
      policy: "gdpr_eu",
    });
    Object.assign(this.accumulatedMapping, tokenized.mapping);

    // Step 3: Build conversation
    const messages: any[] = [
      {
        role: "system",
        content:
          "You are a GDPR-aware customer support assistant. Answer questions using only the provided context. Be concise and helpful.",
      },
      ...this.conversationHistory,
      { role: "user", content: tokenized.text },
    ];

    // Step 4: Get AI response — no PII in the prompt
    const completion = await this.openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
    });
    const aiResponse = completion.choices[0].message.content!;

    // Store in conversation history (tokenized)
    this.conversationHistory.push(
      { role: "user", content: tokenized.text },
      { role: "assistant", content: aiResponse }
    );

    // Step 5: Detokenize for the user
    const restored = this.blindfold.detokenize(
      aiResponse,
      this.accumulatedMapping
    );
    return restored.text;
  }
}

async function main() {
  const rag = new CustomerSupportRAG();
  await rag.initialize();
  await rag.ingestTickets(EU_SUPPORT_TICKETS);

  const questions = [
    "What happened with Hans Mueller's billing issue?",
    "Did Marie Dupont's data export request get resolved?",
    "Which tickets had payment-related issues?",
  ];

  console.log("=== Multi-Turn Customer Support Chat ===\n");
  for (const question of questions) {
    console.log(`Customer: ${question}`);
    const response = await rag.query(question);
    console.log(`Agent: ${response}\n`);
  }
}

main();
