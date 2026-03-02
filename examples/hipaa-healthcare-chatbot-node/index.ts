/**
 * HIPAA-Compliant Healthcare Chatbot with Blindfold (TypeScript)
 *
 * Tokenizes PHI with the hipaa_us policy and US region before sending
 * to OpenAI. Supports multi-turn conversation with accumulated mapping.
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
import OpenAI from "openai";

// API key is optional — omit it to run in local mode (regex-based, offline)
const blindfold = new Blindfold({
  apiKey: process.env.BLINDFOLD_API_KEY,
  region: "us",
});
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

class HealthcareChatbot {
  private conversation: { role: string; content: string }[] = [
    {
      role: "system",
      content:
        "You are a helpful healthcare assistant. Use patient identifier " +
        "tokens exactly as given. Never ask for real patient information.",
    },
  ];
  private mapping: Record<string, string> = {};

  async chat(userMessage: string): Promise<string> {
    // 1. Tokenize PHI with hipaa_us policy
    const tokenized = await blindfold.tokenize(userMessage, {
      policy: "hipaa_us",
    });

    // 2. Accumulate mapping across turns
    Object.assign(this.mapping, tokenized.mapping);
    this.conversation.push({ role: "user", content: tokenized.text });

    // 3. Send to OpenAI — only tokens, never real PHI
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: this.conversation as any,
    });
    const aiResponse = completion.choices[0].message.content!;

    // 4. Store tokenized response in history
    this.conversation.push({ role: "assistant", content: aiResponse });

    // 5. Detokenize for display to clinician
    const restored = blindfold.detokenize(aiResponse, this.mapping);
    return restored.text;
  }
}

async function main() {
  // Single query example
  const note =
    "Patient John Smith (DOB: 03/15/1982, SSN: 123-45-6789) " +
    "presented with chest pain. Dr. Emily Chen ordered an ECG. " +
    "Contact: john.smith@email.com, (555) 867-5309. MRN: 4820193.";

  const tokenized = await blindfold.tokenize(note, { policy: "hipaa_us" });
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "Summarize this clinical note briefly." },
      { role: "user", content: tokenized.text },
    ],
  });
  const restored = blindfold.detokenize(
    completion.choices[0].message.content!,
    tokenized.mapping
  );
  console.log(restored.text);

  // Multi-turn conversation example
  const chatbot = new HealthcareChatbot();
  const turns = [
    "Look up patient Maria Garcia, DOB 11/22/1975, SSN 987-65-4321. She has persistent cough.",
    "What medications is she currently on? Her insurance ID is BCBS-449281.",
    "Draft a referral to Dr. Robert Kim at Springfield Medical Group.",
  ];
  for (const turn of turns) {
    const response = await chatbot.chat(turn);
    console.log(response);
  }
}

main();
