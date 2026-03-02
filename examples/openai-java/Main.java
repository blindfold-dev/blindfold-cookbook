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

import dev.blindfold.sdk.Blindfold;
import dev.blindfold.sdk.models.TokenizeResponse;
import dev.blindfold.sdk.models.DetokenizeResponse;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.ChatModel;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import io.github.cdimascio.dotenv.Dotenv;

public class Main {

    public static String protectedChat(String userMessage, String policy, String model) {
        Dotenv dotenv = Dotenv.load();

        // API key is optional — omit it to run in local mode (regex-based, offline)
        String blindfoldKey = dotenv.get("BLINDFOLD_API_KEY");
        Blindfold blindfold = blindfoldKey != null ? new Blindfold(blindfoldKey) : new Blindfold();

        OpenAIClient openai = OpenAIOkHttpClient.builder()
                .apiKey(dotenv.get("OPENAI_API_KEY"))
                .build();

        // 1. Tokenize — replace PII with safe tokens
        TokenizeResponse tokenized = blindfold.tokenize(userMessage, policy);
        System.out.println("Tokenized: " + tokenized.getText());

        // 2. Send tokenized text to OpenAI — no real PII leaves your system
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4O_MINI)
                .addSystemMessage("You are a helpful assistant.")
                .addUserMessage(tokenized.getText())
                .build();

        ChatCompletion completion = openai.chat().completions().create(params);
        String aiResponse = completion.choices().get(0).message().content().get();

        // 3. Detokenize — restore original values in the AI response
        DetokenizeResponse restored = blindfold.detokenize(aiResponse, tokenized.getMapping());
        return restored.getText();
    }

    public static void main(String[] args) {
        String message = "Please summarize the account for John Smith (john.smith@acme.com), customer ID 4532-7562-9102-3456.";
        System.out.println("\nUser: " + message + "\n");

        String response = protectedChat(message, "basic", "gpt-4o-mini");
        System.out.println("\nAssistant: " + response);
    }
}
