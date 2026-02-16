"""
GDPR-Compliant OpenAI Integration with Blindfold

Demonstrates how to use Blindfold's EU region to process European user data
through OpenAI while staying GDPR-compliant:

- PII is tokenized in the EU region BEFORE it reaches OpenAI
- OpenAI only sees anonymized tokens, never real personal data
- The original values are restored after the AI responds
- The gdpr_eu policy detects EU-relevant entity types (names, emails, IBANs, etc.)
- Full audit trail is maintained for Data Processing Agreement (DPA) compliance

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
    """Initialize Blindfold (EU region) and OpenAI clients."""
    blindfold = Blindfold(
        api_key=os.environ["BLINDFOLD_API_KEY"],
        region="eu",  # Data stays in the EU region
    )
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return blindfold, openai


def gdpr_safe_chat(
    blindfold: Blindfold,
    openai_client: OpenAI,
    user_message: str,
    system_prompt: str = "You are a helpful customer service assistant.",
    model: str = "gpt-4o-mini",
) -> dict:
    """Send a message to OpenAI with GDPR-compliant PII protection.

    Returns a dict with the final response and metadata about what was protected.
    """
    # Step 1: Tokenize with the gdpr_eu policy
    # This replaces all EU-relevant PII (names, emails, phones, IBANs,
    # addresses, dates of birth, etc.) with safe tokens like <Person_1>
    tokenized = blindfold.tokenize(user_message, policy="gdpr_eu")

    print(f"  Detected {tokenized.entities_count} PII entities")
    for entity in tokenized.detected_entities:
        print(f"    - {entity.type}: {entity.text} (confidence: {entity.score:.0%})")

    print(f"  Tokenized text sent to OpenAI:")
    print(f"    \"{tokenized.text}\"")

    # Step 2: Send ONLY the tokenized text to OpenAI
    # No real personal data crosses the EU border
    completion = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": tokenized.text},
        ],
    )
    ai_response = completion.choices[0].message.content
    print(f"  AI response (still tokenized):")
    print(f"    \"{ai_response}\"")

    # Step 3: Detokenize — restore original values in the AI response
    restored = blindfold.detokenize(ai_response, tokenized.mapping)

    return {
        "response": restored.text,
        "entities_detected": tokenized.entities_count,
        "mapping": tokenized.mapping,
    }


# ── Example: Customer support with EU user data ─────────────────────────────

def example_customer_support():
    """Simulate a customer support scenario with EU personal data."""
    blindfold, openai_client = create_clients()

    message = (
        "Hi, my name is Hans Mueller and I need help with my subscription. "
        "My email is hans.mueller@example.de, phone +49 170 1234567. "
        "My IBAN is DE89 3704 0044 0532 0130 00. "
        "I was born on 15/03/1985 and live at Berliner Str. 42, 10115 Berlin."
    )

    print(f"\n{'='*60}")
    print("GDPR-Compliant Customer Support Example")
    print(f"{'='*60}")
    print(f"\nUser message:\n  \"{message}\"\n")

    result = gdpr_safe_chat(blindfold, openai_client, message)

    print(f"\n  Final response (PII restored):")
    print(f"    \"{result['response']}\"")
    print(f"\n  PII entities protected: {result['entities_detected']}")


# ── Example: Batch processing EU support tickets ────────────────────────────

def example_batch_tickets():
    """Process multiple EU support tickets in batch."""
    blindfold, openai_client = create_clients()

    tickets = [
        "Customer Marie Dupont (marie.dupont@example.fr) reports billing issue. Account FR76 3000 6000 0112 3456 7890 189.",
        "Jan Novak (jan.novak@example.cz, +420 123 456 789) requests data export under GDPR Art. 15.",
        "Sofia Garcia, sofia.garcia@example.es, cannot access her account since 2026-01-15.",
    ]

    print(f"\n{'='*60}")
    print("Batch Processing EU Support Tickets")
    print(f"{'='*60}")

    # Tokenize all tickets in one API call
    batch = blindfold.tokenize_batch(tickets, policy="gdpr_eu")

    print(f"\nTokenized {len(batch.results)} tickets:")
    for i, result in enumerate(batch.results):
        print(f"\n  Ticket {i+1}:")
        print(f"    Original:  {tickets[i][:80]}...")
        print(f"    Tokenized: {result.text[:80]}...")
        print(f"    Entities:  {result.entities_count}")

    # Send all tokenized tickets to OpenAI for categorization
    ticket_texts = "\n".join(
        f"Ticket {i+1}: {r.text}" for i, r in enumerate(batch.results)
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

    print(f"\n  AI Categorization (no real PII was sent):")
    print(f"    {completion.choices[0].message.content}")


if __name__ == "__main__":
    example_customer_support()
    example_batch_tickets()
