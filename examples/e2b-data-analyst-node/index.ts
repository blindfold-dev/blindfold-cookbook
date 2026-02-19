/**
 * Blindfold + E2B: PII-Safe AI Data Analyst (TypeScript)
 *
 * OpenAI writes analysis code from tokenized data (never sees real PII),
 * then E2B executes the code on the original data for accurate results.
 *
 * Usage:
 *   npm install
 *   cp .env.example .env  # add your API keys
 *   npm start
 */

import "dotenv/config";
import { Blindfold } from "@blindfold/sdk";
import { Sandbox } from "@e2b/code-interpreter";
import OpenAI from "openai";

const blindfold = new Blindfold({ apiKey: process.env.BLINDFOLD_API_KEY! });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

const SAMPLE_CSV = `name,email,ssn,age,diagnosis,medication,doctor,city
John Smith,john.smith@email.com,123-45-6789,45,Type 2 Diabetes,Metformin 500mg,Dr. Sarah Johnson,Boston
Maria Garcia,maria.garcia@company.org,987-65-4321,32,Hypertension,Lisinopril 10mg,Dr. Michael Chen,New York
Robert Wilson,rwilson@hospital.net,456-78-9012,58,Type 2 Diabetes,Metformin 1000mg,Dr. Sarah Johnson,Boston
Emily Davis,emily.d@example.com,234-56-7890,27,Asthma,Albuterol Inhaler,Dr. Lisa Park,Chicago
James Brown,jbrown@mail.com,345-67-8901,51,Hypertension,Amlodipine 5mg,Dr. Michael Chen,New York
Anna Martinez,anna.m@health.org,567-89-0123,39,Asthma,Fluticasone Inhaler,Dr. Lisa Park,San Francisco
David Lee,david.lee@work.com,678-90-1234,63,Type 2 Diabetes,Insulin Glargine,Dr. Sarah Johnson,Boston
Sarah Thompson,s.thompson@email.com,789-01-2345,44,Migraine,Sumatriptan 50mg,Dr. Amy Wilson,Chicago`;

async function main() {
  // Step 1: Tokenize — replace PII with safe tokens
  console.log("[Step 1] Tokenizing CSV with Blindfold...\n");
  const tokenized = await blindfold.tokenize(SAMPLE_CSV, { policy: "strict" });

  console.log(`Detected ${tokenized.entities_count} PII entities:`);
  for (const entity of tokenized.detected_entities) {
    console.log(
      `  - ${entity.type}: ${entity.text} (${Math.round(entity.score * 100)}%)`
    );
  }

  console.log(`\nTokenized CSV (what OpenAI sees):`);
  for (const line of tokenized.text.split("\n").slice(0, 4)) {
    console.log(`  ${line}`);
  }
  console.log("  ...");

  // Step 2: Ask OpenAI to write analysis code (it only sees tokens)
  console.log("\n[Step 2] Asking OpenAI to write analysis code...\n");

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content:
          "You are a data analyst. Write Python code using pandas to analyze CSV data. " +
          "Read the CSV from '/tmp/data.csv'. Print all results with labels. " +
          "Return ONLY Python code, no markdown, no backticks.",
      },
      {
        role: "user",
        content:
          `Here is the dataset:\n\n${tokenized.text}\n\n` +
          "Analyze it: count patients per diagnosis, average age per diagnosis, " +
          "which doctor treats the most patients, and patients per city.",
      },
    ],
  });
  const code = completion.choices[0].message.content!;

  console.log("AI-generated code:");
  for (const line of code.split("\n")) {
    console.log(`  ${line}`);
  }

  // Step 3: Execute in E2B sandbox with the ORIGINAL data
  console.log("\n[Step 3] Running code in E2B sandbox (with real data)...\n");

  const sandbox = await Sandbox.create();
  try {
    await sandbox.files.write("/tmp/data.csv", SAMPLE_CSV);
    const execution = await sandbox.runCode(code);

    if (execution.error) {
      console.log(`Error: ${execution.error.name}: ${execution.error.value}`);
    } else {
      console.log("Results:");
      console.log(execution.logs.stdout.join(""));
    }
  } finally {
    await sandbox.kill();
  }
}

main();
