/**
 * Express server with Blindfold PII protection middleware.
 *
 * POST /chat sends user messages to OpenAI with PII automatically
 * tokenized and detokenized.
 *
 * Works in two modes:
 *   - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
 *   - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations
 */

import "dotenv/config";
import express from "express";
import OpenAI from "openai";
import { blindfoldMiddleware, type BlindfoldRequest } from "./middleware.js";

const app = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

app.use(express.json());

// Apply Blindfold middleware to /chat endpoint
// API key is optional — omit it to run in local mode (regex-based, offline)
app.post(
  "/chat",
  blindfoldMiddleware({
    apiKey: process.env.BLINDFOLD_API_KEY,
    policy: "basic",
    textField: "message",
  }),
  async (req: BlindfoldRequest, res) => {
    const tokenizedMessage = req.body.message;
    console.log("Tokenized message:", tokenizedMessage);

    // OpenAI only sees tokenized text — no real PII
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: tokenizedMessage },
      ],
    });

    const aiResponse = completion.choices[0].message.content!;

    // Detokenize the AI response to restore real values
    const response = req.blindfold
      ? req.blindfold.detokenize(aiResponse)
      : aiResponse;

    res.json({ response });
  }
);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log("Try: curl -X POST http://localhost:3000/chat -H 'Content-Type: application/json' -d '{\"message\": \"Email john@example.com about his order\"}'");
});
