// OpenAI + Blindfold: Protect PII in LLM conversations.
//
// Tokenizes user messages before sending to OpenAI, then detokenizes
// the response so real names/emails appear in the final output.
//
// Works in two modes:
//   - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
//   - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	blindfold "github.com/blindfold-dev/Blindfold/packages/go-sdk"
	"github.com/joho/godotenv"
	openai "github.com/sashabaranov/go-openai"
)

func protectedChat(ctx context.Context, userMessage, policy, model string) (string, error) {
	// API key is optional — omit it to run in local mode (regex-based, offline)
	var opts []blindfold.Option
	if key := os.Getenv("BLINDFOLD_API_KEY"); key != "" {
		opts = append(opts, blindfold.WithAPIKey(key))
	}
	bf := blindfold.New(opts...)
	oa := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	// 1. Tokenize — replace PII with safe tokens
	tokenized, err := bf.Tokenize(ctx, userMessage, blindfold.WithCallPolicy(policy))
	if err != nil {
		return "", fmt.Errorf("tokenize: %w", err)
	}
	fmt.Printf("Tokenized: %s\n", tokenized.Text)

	// 2. Send tokenized text to OpenAI — no real PII leaves your system
	completion, err := oa.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "You are a helpful assistant."},
			{Role: openai.ChatMessageRoleUser, Content: tokenized.Text},
		},
	})
	if err != nil {
		return "", fmt.Errorf("openai: %w", err)
	}
	aiResponse := completion.Choices[0].Message.Content

	// 3. Detokenize — restore original values in the AI response
	restored := bf.Detokenize(aiResponse, tokenized.Mapping)
	return restored.Text, nil
}

func main() {
	_ = godotenv.Load()

	message := "Please summarize the account for John Smith (john.smith@acme.com), customer ID 4532-7562-9102-3456."
	fmt.Printf("\nUser: %s\n\n", message)

	response, err := protectedChat(context.Background(), message, "basic", "gpt-4o-mini")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nAssistant: %s\n", response)
}
