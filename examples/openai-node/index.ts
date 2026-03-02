/**
 * OpenAI + Blindfold: Protect PII in LLM conversations.
 *
 * Tokenizes user messages before sending to OpenAI, then detokenizes
 * the response so real names/emails appear in the final output.
 *
 * Works in two modes:
 *   - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
 *   - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations
 */

import "dotenv/config";
import { Blindfold } from "@blindfold/sdk";
import OpenAI from "openai";

// API key is optional — omit it to run in local mode (regex-based, offline)
const blindfold = new Blindfold({ apiKey: process.env.BLINDFOLD_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

async function protectedChat(
  userMessage: string,
  policy: string = "basic",
  model: string = "gpt-4o-mini"
): Promise<string> {
  // 1. Tokenize — replace PII with safe tokens
  const tokenized = await blindfold.tokenize(userMessage, { policy });
  console.log("Tokenized:", tokenized.text);

  // 2. Send tokenized text to OpenAI — no real PII leaves your system
  const completion = await openai.chat.completions.create({
    model,
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: tokenized.text },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;

  // 3. Detokenize — restore original values in the AI response
  const restored = blindfold.detokenize(aiResponse, tokenized.mapping);
  return restored.text;
}

async function main() {
  const message =
    "Please summarize the account for John Smith (john.smith@acme.com), customer ID 4532-7562-9102-3456.";
  console.log(`\nUser: ${message}\n`);

  const response = await protectedChat(message, "basic");
  console.log(`\nAssistant: ${response}`);
}

main();
