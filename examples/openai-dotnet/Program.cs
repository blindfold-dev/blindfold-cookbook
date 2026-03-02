// OpenAI + Blindfold: Protect PII in LLM conversations.
//
// Tokenizes user messages before sending to OpenAI, then detokenizes
// the response so real names/emails appear in the final output.
//
// Works in two modes:
//   - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
//   - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations

using Blindfold.Sdk;
using DotNetEnv;
using OpenAI.Chat;

Env.Load();

// API key is optional — omit it to run in local mode (regex-based, offline)
var blindfoldKey = Environment.GetEnvironmentVariable("BLINDFOLD_API_KEY");
var blindfold = string.IsNullOrEmpty(blindfoldKey)
    ? new BlindfoldClient()
    : new BlindfoldClient(blindfoldKey);

var openai = new ChatClient("gpt-4o-mini", Environment.GetEnvironmentVariable("OPENAI_API_KEY"));

async Task<string> ProtectedChat(string userMessage, string policy = "basic")
{
    // 1. Tokenize — replace PII with safe tokens
    var tokenized = await blindfold.TokenizeAsync(userMessage, policy);
    Console.WriteLine($"Tokenized: {tokenized.Text}");

    // 2. Send tokenized text to OpenAI — no real PII leaves your system
    ChatCompletion completion = await openai.CompleteChatAsync(
    [
        new SystemChatMessage("You are a helpful assistant."),
        new UserChatMessage(tokenized.Text),
    ]);
    var aiResponse = completion.Content[0].Text;

    // 3. Detokenize — restore original values in the AI response
    var restored = blindfold.Detokenize(aiResponse, tokenized.Mapping);
    return restored.Text;
}

var message = "Please summarize the account for John Smith (john.smith@acme.com), customer ID 4532-7562-9102-3456.";
Console.WriteLine($"\nUser: {message}\n");

var response = await ProtectedChat(message, "basic");
Console.WriteLine($"\nAssistant: {response}");
