"""
OpenAI + Blindfold: Protect PII in LLM conversations.

Tokenizes user messages before sending to OpenAI, then detokenizes
the response so real names/emails appear in the final output.

Works in two modes:
  - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
  - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations
"""

import os

from blindfold import Blindfold
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def protected_chat(
    user_message: str,
    policy: str = "basic",
    model: str = "gpt-4o-mini",
) -> str:
    """Send a message to OpenAI with PII automatically protected."""
    # API key is optional — omit it to run in local mode (regex-based, offline)
    blindfold = Blindfold(api_key=os.environ.get("BLINDFOLD_API_KEY"))
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 1. Tokenize — replace PII with safe tokens
    tokenized = blindfold.tokenize(user_message, policy=policy)
    print(f"Tokenized: {tokenized.text}")

    # 2. Send tokenized text to OpenAI — no real PII leaves your system
    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": tokenized.text},
        ],
    )
    ai_response = completion.choices[0].message.content

    # 3. Detokenize — restore original values in the AI response
    restored = blindfold.detokenize(ai_response, tokenized.mapping)
    return restored.text


if __name__ == "__main__":
    message = "Please summarize the account for John Smith (john.smith@acme.com), customer ID 4532-7562-9102-3456."
    print(f"\nUser: {message}\n")

    response = protected_chat(message, policy="basic")
    print(f"\nAssistant: {response}")
