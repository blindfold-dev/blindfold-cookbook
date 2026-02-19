/**
 * LangChain.js + Blindfold: PII-safe chains with RunnableLambda.
 *
 * Wraps any LangChain chain with Blindfold tokenization/detokenization
 * so PII never reaches the language model.
 *
 * Usage:
 *   npm install
 *   cp .env.example .env  # add your API keys
 *   npm start
 */

import "dotenv/config";
import { Blindfold } from "@blindfold/sdk";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda } from "@langchain/core/runnables";

const blindfold = new Blindfold({ apiKey: process.env.BLINDFOLD_API_KEY! });

function blindfoldProtect(policy: string = "basic") {
  const mappingStore: Record<string, string> = {};

  const tokenize = RunnableLambda.from(async (text: string) => {
    const result = await blindfold.tokenize(text, { policy });
    Object.assign(mappingStore, result.mapping);
    console.log(`  Tokenized: "${result.text}"`);
    return result.text;
  });

  const detokenize = RunnableLambda.from(async (text: string) => {
    const restored = blindfold.detokenize(text, mappingStore);
    // Clear mapping after use
    for (const key of Object.keys(mappingStore)) {
      delete mappingStore[key];
    }
    return restored.text;
  });

  return [tokenize, detokenize] as const;
}

async function main() {
  const [tokenize, detokenize] = blindfoldProtect("basic");

  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY!,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant."],
    ["user", "{input}"],
  ]);

  // Chain: tokenize → prompt → LLM → extract content → detokenize
  const chain = tokenize
    .pipe(prompt)
    .pipe(llm)
    .pipe(RunnableLambda.from((msg: any) => msg.content as string))
    .pipe(detokenize);

  const message =
    "Write a follow-up email to Jane Doe (jane@example.com) about her order #12345.";
  console.log(`\nUser: ${message}\n`);

  const response = await chain.invoke(message);
  console.log(`\nAssistant: ${response}`);
}

main();
