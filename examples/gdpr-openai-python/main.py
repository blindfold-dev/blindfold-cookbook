"""
GDPR-Compliant OpenAI Integration with Blindfold

Tokenizes EU user data with the gdpr_eu policy and EU region before
sending to OpenAI. The LLM never sees real personal data.

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import os

from blindfold import Blindfold
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def create_clients():
    blindfold = Blindfold(
        api_key=os.environ["BLINDFOLD_API_KEY"],
        region="eu",
    )
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return blindfold, openai


def gdpr_safe_chat(blindfold: Blindfold, openai_client: OpenAI, user_message: str) -> str:
    # Step 1: Tokenize with the gdpr_eu policy
    tokenized = blindfold.tokenize(user_message, policy="gdpr_eu")

    # Step 2: Send ONLY the tokenized text to OpenAI
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful customer service assistant."},
            {"role": "user", "content": tokenized.text},
        ],
    )
    ai_response = completion.choices[0].message.content

    # Step 3: Detokenize — restore original values in the AI response
    restored = blindfold.detokenize(ai_response, tokenized.mapping)
    return restored.text


def gdpr_batch_categorize(blindfold: Blindfold, openai_client: OpenAI, tickets: list[str]) -> str:
    # Tokenize all tickets in one API call
    batch = blindfold.tokenize_batch(tickets, policy="gdpr_eu")

    # Send tokenized tickets to OpenAI for categorization
    ticket_texts = "\n".join(
        f"Ticket {i+1}: {r['text']}" for i, r in enumerate(batch.results)
    )

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Categorize each support ticket as: billing, data_request, or access_issue. "
                           "Reply with one line per ticket: 'Ticket N: category - brief summary'.",
            },
            {"role": "user", "content": ticket_texts},
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    blindfold, openai_client = create_clients()

    # Single message example
    message = (
        "Hi, my name is Hans Mueller and I need help with my subscription. "
        "My email is hans.mueller@example.de, phone +49 170 1234567. "
        "My IBAN is DE89 3704 0044 0532 0130 00. "
        "I was born on 15/03/1985 and live at Berliner Str. 42, 10115 Berlin."
    )
    response = gdpr_safe_chat(blindfold, openai_client, message)
    print(response)

    # Batch categorization example
    tickets = [
        "Customer Marie Dupont (marie.dupont@example.fr) reports billing issue. Account FR76 3000 6000 0112 3456 7890 189.",
        "Jan Novak (jan.novak@example.cz, +420 123 456 789) requests data export under GDPR Art. 15.",
        "Sofia Garcia, sofia.garcia@example.es, cannot access her account since 2026-01-15.",
    ]
    categories = gdpr_batch_categorize(blindfold, openai_client, tickets)
    print(categories)
