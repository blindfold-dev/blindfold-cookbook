/**
 * RAG with Role-Based PII Control (RBAC Policies) -- TypeScript
 *
 * Demonstrates how different user roles see different levels of PII in a
 * healthcare RAG pipeline. Each role has a Blindfold policy that controls
 * which entity types are tokenized before the LLM sees them.
 *
 * Works in two modes:
 *   - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
 *   - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations
 *
 * Roles:
 *   - doctor:     Sees names, conditions, medications. Contact info redacted.
 *   - nurse:      Sees names and conditions. No SSN, contact, or DOB.
 *   - billing:    Sees names and insurance. Clinical details redacted.
 *   - researcher: Fully de-identified. All PII removed (policy="strict").
 *
 * Flow:
 *   1. Ingest:  blindfold.redact(record, { entities: [...] }) to remove
 *               contact info from stored documents (all roles share this).
 *   2. Query:   blindfold.tokenize(context + question, { entities | policy })
 *               with role-specific settings before the LLM call.
 *   3. Restore: blindfold.detokenize(response, mapping) for the user.
 *
 * Usage:
 *   npm install
 *   cp .env.example .env  # add your API keys
 *   npm start             # run all roles
 *   npm start -- --role doctor   # run a single role
 */

import "dotenv/config";
import { Blindfold } from "@blindfold/sdk";
import OpenAI from "openai";
import { ChromaClient } from "chromadb";

// API key is optional — omit it to run in local mode (regex-based, offline)
const blindfold = new Blindfold({ apiKey: process.env.BLINDFOLD_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

// ---------------------------------------------------------------------------
// Sample healthcare patient records
// ---------------------------------------------------------------------------

const PATIENT_RECORDS = [
  "Patient Record #PR-2024-001: Sarah Chen (sarah.chen@email.com, +1-555-0142, SSN 412-55-6789, DOB 1985-03-15) was diagnosed with Type 2 Diabetes on 2024-01-15. Prescribed Metformin 500mg twice daily. Insurance: BlueCross policy BC-2847193. Primary care physician: Dr. James Wilson.",
  "Patient Record #PR-2024-002: Marcus Johnson (marcus.j@email.com, +1-555-0198, SSN 331-78-4521, DOB 1992-07-22) presented with acute bronchitis on 2024-02-03. Prescribed Azithromycin 250mg for 5 days. Insurance: Aetna policy AE-9182736. Referred by Dr. Lisa Park.",
  "Patient Record #PR-2024-003: Elena Rodriguez (elena.r@email.com, +1-555-0267, SSN 528-91-3456, DOB 1978-11-08) underwent knee replacement surgery on 2024-03-10. Post-op recovery normal. Insurance: UnitedHealth policy UH-5529384. Surgeon: Dr. Robert Kim. Billing: $47,500 pre-insurance.",
  "Patient Record #PR-2024-004: Sarah Chen (sarah.chen@email.com, +1-555-0142) follow-up visit on 2024-04-20. HbA1c improved from 8.2% to 7.1%. Continuing Metformin, added Lisinopril 10mg for blood pressure management. Next appointment in 3 months.",
  "Patient Record #PR-2024-005: Marcus Johnson (marcus.j@email.com, +1-555-0198) emergency visit on 2024-05-15. Severe allergic reaction to shellfish. Administered epinephrine. Prescribed EpiPen for future emergencies. Insurance claim filed with Aetna policy AE-9182736.",
];

// ---------------------------------------------------------------------------
// Role definitions -- each role tokenizes different entity types
// ---------------------------------------------------------------------------

const ROLES: Record<string, { description: string; entities: string[] | null }> = {
  doctor: {
    description: "Full clinical access -- sees patient names, conditions, medications. Contact info and financial data redacted.",
    entities: ["email address", "phone number", "social security number", "credit card number", "iban"],
  },
  nurse: {
    description: "Clinical care access -- sees names and conditions. No SSN, contact, or financial data.",
    entities: ["email address", "phone number", "social security number", "credit card number", "iban", "date of birth"],
  },
  billing: {
    description: "Financial access -- sees names and insurance info. Clinical details and contact info redacted.",
    entities: ["email address", "phone number", "social security number", "medical condition", "medication"],
  },
  researcher: {
    description: "De-identified access -- all PII removed. Content-based search only.",
    entities: null, // uses policy="strict" to remove all PII
  },
};

const QUERIES = [
  "What medications is Sarah Chen taking?",
  "What was Marcus Johnson's emergency about?",
  "Which patients had surgery?",
];

const SYSTEM_PROMPT =
  "You are a healthcare assistant. Answer the question based only on the provided patient records. Keep your answer concise and clinical.";

// ---------------------------------------------------------------------------
// Ingestion: Redact contact info from all records (shared across roles)
// ---------------------------------------------------------------------------

async function ingestRecords(chroma: ChromaClient) {
  const collection = await chroma.getOrCreateCollection({
    name: "rbac_patient_records",
  });

  console.log("=== Ingesting Patient Records ===\n");
  console.log("Redacting contact info (emails, phones) for storage...\n");

  for (let idx = 0; idx < PATIENT_RECORDS.length; idx++) {
    const result = await blindfold.redact(PATIENT_RECORDS[idx], {
      entities: ["email address", "phone number"],
    });

    await collection.add({
      documents: [result.text],
      ids: [`record-${idx}`],
    });

    console.log(`  Record ${idx + 1}: ${result.entities_count} entities redacted`);
    console.log(`    Stored: "${result.text.slice(0, 90)}..."`);
  }

  console.log(`\nStored ${PATIENT_RECORDS.length} records in ChromaDB\n`);
  return collection;
}

// ---------------------------------------------------------------------------
// Query: Role-specific tokenization before the LLM
// ---------------------------------------------------------------------------

async function queryWithRole(
  collection: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>,
  question: string,
  role: string
): Promise<string> {
  const roleConfig = ROLES[role];

  // Step 1: Search with original question -- names match in the vector store
  console.log(`  Searching for relevant records...`);
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

  // Step 2: Tokenize with role-specific entity list (or strict policy)
  const promptText = `Context:\n${context}\n\nQuestion: ${question}`;

  let tokenized;
  if (roleConfig.entities === null) {
    // Researcher role: use the "strict" policy to remove ALL PII.
    // You can also create custom policies in the Blindfold dashboard
    // and reference them by name here (e.g., policy: "my_custom_policy").
    tokenized = await blindfold.tokenize(promptText, { policy: "strict" });
    console.log(`\n  Tokenized with policy="strict" (${tokenized.entities_count} entities)`);
  } else {
    tokenized = await blindfold.tokenize(promptText, { entities: roleConfig.entities });
    console.log(`\n  Tokenized ${tokenized.entities_count} entities for role "${role}"`);
  }

  console.log(`  Tokenized prompt preview: "${tokenized.text.slice(0, 120)}..."`);

  // Step 3: LLM call -- only role-appropriate data visible
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: tokenized.text },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;
  console.log(`\n  LLM response (tokenized): "${aiResponse}"`);

  // Step 4: Detokenize -- restore values the role is allowed to see
  const final = blindfold.detokenize(aiResponse, tokenized.mapping);
  return final.text;
}

// ---------------------------------------------------------------------------
// Run queries for a single role
// ---------------------------------------------------------------------------

async function runRole(
  collection: Awaited<ReturnType<ChromaClient["getOrCreateCollection"]>>,
  role: string
) {
  const roleConfig = ROLES[role];
  console.log("=".repeat(70));
  console.log(`ROLE: ${role.toUpperCase()}`);
  console.log(`  ${roleConfig.description}`);
  if (roleConfig.entities) {
    console.log(`  Tokenized entities: ${roleConfig.entities.join(", ")}`);
  } else {
    console.log(`  Policy: strict (all PII removed)`);
  }
  console.log("=".repeat(70));
  console.log();

  for (const question of QUERIES) {
    console.log(`  Question: "${question}"\n`);
    const answer = await queryWithRole(collection, question, role);
    console.log(`\n  Final answer: "${answer}"`);
    console.log("-".repeat(60) + "\n");
  }
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

function parseArgs(): string[] {
  const args = process.argv.slice(2);
  let selectedRole: string | null = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--role" && i + 1 < args.length) {
      selectedRole = args[i + 1].toLowerCase();
      i++;
    }
  }

  if (selectedRole) {
    if (!ROLES[selectedRole]) {
      console.error(
        `Unknown role "${selectedRole}". Valid roles: ${Object.keys(ROLES).join(", ")}`
      );
      process.exit(1);
    }
    return [selectedRole];
  }

  return Object.keys(ROLES);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const roles = parseArgs();

  const chroma = new ChromaClient();
  const collection = await ingestRecords(chroma);

  for (let i = 0; i < roles.length; i++) {
    if (i > 0) console.log();
    await runRole(collection, roles[i]);
  }

  console.log("=== How Role-Based PII Control Works ===\n");
  console.log("  1. Patient records are ingested with contact info redacted (shared)");
  console.log("  2. At query time, each role has different tokenization settings:");
  console.log('     - Doctor:     emails, phones, SSNs tokenized (sees clinical data)');
  console.log('     - Nurse:      emails, phones, SSNs, DOB tokenized');
  console.log('     - Billing:    emails, phones, SSNs, conditions, medications tokenized');
  console.log('     - Researcher: policy="strict" removes ALL PII');
  console.log("  3. The LLM only sees what each role is allowed to see");
  console.log("  4. Detokenization restores the role-appropriate values\n");
  console.log("  You can also create custom policies in the Blindfold dashboard");
  console.log("  and reference them by name (e.g., policy: 'hipaa_minimum_necessary').\n");
}

main();
