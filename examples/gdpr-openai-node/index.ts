/**
 * GDPR-Compliant OpenAI Integration with Blindfold (TypeScript)
 *
 * Tokenizes EU user data with the gdpr_eu policy and EU region before
 * sending to OpenAI. The LLM never sees real personal data.
 *
 * Usage:
 *   npm install
 *   cp .env.example .env  # add your API keys
 *   npm start
 */

import "dotenv/config";
import { Blindfold } from "@blindfold/sdk";
import OpenAI from "openai";

const blindfold = new Blindfold({
  apiKey: process.env.BLINDFOLD_API_KEY,
  region: "eu",
});
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

async function gdprSafeChat(userMessage: string): Promise<string> {
  // Step 1: Tokenize with the gdpr_eu policy
  const tokenized = await blindfold.tokenize(userMessage, {
    policy: "gdpr_eu",
  });

  // Step 2: Send ONLY the tokenized text to OpenAI
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "You are a helpful customer service assistant." },
      { role: "user", content: tokenized.text },
    ],
  });
  const aiResponse = completion.choices[0].message.content!;

  // Step 3: Detokenize — restore original values
  const restored = blindfold.detokenize(aiResponse, tokenized.mapping);
  return restored.text;
}

async function main() {
  const message =
    "Hi, my name is Hans Mueller and I need help with my subscription. " +
    "My email is hans.mueller@example.de, phone +49 170 1234567. " +
    "My IBAN is DE89 3704 0044 0532 0130 00. " +
    "I was born on 15/03/1985 and live at Berliner Str. 42, 10115 Berlin.";

  const response = await gdprSafeChat(message);
  console.log(response);
}

main();
