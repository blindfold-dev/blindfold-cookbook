"""
HIPAA-Compliant Healthcare Chatbot with Blindfold

Demonstrates how to build a healthcare AI assistant that protects
Protected Health Information (PHI) using Blindfold's US region:

- PHI is tokenized in the US region BEFORE it reaches OpenAI
- The hipaa_us policy detects all 18 HIPAA identifiers (names, SSNs, MRNs, etc.)
- OpenAI only sees anonymized tokens, never real patient data
- Supports both single queries and multi-turn conversations
- Audit trail maintained for HIPAA compliance requirements

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import os
from typing import Optional

from blindfold import Blindfold
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def create_clients():
    """Initialize Blindfold (US region) and OpenAI clients."""
    blindfold = Blindfold(
        api_key=os.environ["BLINDFOLD_API_KEY"],
        region="us",  # PHI processed in US region
    )
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return blindfold, openai


# ── Safe Healthcare Chat ─────────────────────────────────────────────────────

class HealthcareChatbot:
    """A HIPAA-compliant chatbot that protects PHI in every message."""

    SYSTEM_PROMPT = (
        "You are a helpful healthcare assistant. You help medical staff "
        "look up patient information, summarize records, and answer clinical "
        "questions. Always be professional and accurate."
    )

    def __init__(self):
        self.blindfold, self.openai = create_clients()
        self.conversation: list[dict] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        # Accumulated mapping across the conversation
        self.mapping: dict[str, str] = {}

    def chat(self, user_message: str) -> str:
        """Send a message with PHI protection. Returns the restored response."""

        # Step 1: Tokenize PHI using the hipaa_us policy
        tokenized = self.blindfold.tokenize(user_message, policy="hipaa_us")

        print(f"\n  PHI detected: {tokenized.entities_count} identifiers")
        for entity in tokenized.detected_entities:
            print(f"    [{entity.type}] {entity.text}")

        # Merge new mappings into conversation-wide mapping
        self.mapping.update(tokenized.mapping)

        # Step 2: Add tokenized message to conversation (no real PHI)
        self.conversation.append({"role": "user", "content": tokenized.text})

        # Step 3: Send to OpenAI — only anonymized tokens
        completion = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation,
        )
        ai_response = completion.choices[0].message.content

        # Keep tokenized response in conversation history
        self.conversation.append({"role": "assistant", "content": ai_response})

        # Step 4: Detokenize for display — restore real PHI
        restored = self.blindfold.detokenize(ai_response, self.mapping)
        return restored.text


# ── Example: Single patient query ────────────────────────────────────────────

def example_patient_query():
    """Simple single-turn query about a patient."""
    blindfold, openai_client = create_clients()

    message = (
        "Patient Sarah Johnson (DOB 03/15/1978, SSN 123-45-6789, "
        "MRN P-4532) presented at Memorial Hospital on 02/10/2026 "
        "with chest pain. Contact: sarah.johnson@email.com, "
        "phone (555) 234-5678. Insurance ID: BC-9876543."
    )

    print(f"\n{'='*60}")
    print("HIPAA-Compliant Patient Query")
    print(f"{'='*60}")
    print(f"\nOriginal message:\n  {message}\n")

    # Tokenize with hipaa_us policy
    tokenized = blindfold.tokenize(message, policy="hipaa_us")

    print(f"PHI identifiers detected: {tokenized.entities_count}")
    for entity in tokenized.detected_entities:
        print(f"  - {entity.type}: \"{entity.text}\" (confidence: {entity.score:.0%})")

    print(f"\nTokenized (safe to send to AI):\n  {tokenized.text}\n")

    # Send to OpenAI
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a healthcare records assistant. Summarize the patient information provided.",
            },
            {"role": "user", "content": tokenized.text},
        ],
    )
    ai_response = completion.choices[0].message.content

    # Detokenize
    restored = blindfold.detokenize(ai_response, tokenized.mapping)

    print(f"AI Summary (PHI restored):\n  {restored.text}")


# ── Example: Multi-turn conversation ─────────────────────────────────────────

def example_conversation():
    """Demonstrate a multi-turn conversation with PHI protection."""
    print(f"\n{'='*60}")
    print("Multi-Turn Healthcare Conversation")
    print(f"{'='*60}")

    bot = HealthcareChatbot()

    messages = [
        "Look up patient record for James Wilson, MRN P-7891, DOB 07/22/1965. "
        "He's at 123 Oak Street, Springfield. SSN 987-65-4321.",

        "What medications should we check for interactions given his age and the "
        "symptoms described? His pharmacy is at CVS on Main St, phone (555) 876-5432.",

        "Please draft a referral note for Dr. Emily Chen at Springfield Cardiology, "
        "email dr.chen@springfieldcardiology.com.",
    ]

    for i, message in enumerate(messages, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {message[:80]}...")

        response = bot.chat(message)
        print(f"\nAssistant: {response[:200]}...")

    print(f"\n  Total PHI mappings accumulated: {len(bot.mapping)}")


# ── Example: Batch processing patient records ────────────────────────────────

def example_batch_records():
    """Process multiple patient records in one API call."""
    blindfold, _ = create_clients()

    records = [
        "Patient: Robert Lee, SSN 111-22-3333, admitted 01/20/2026 for knee surgery. Contact: robert.lee@email.com.",
        "Patient: Maria Santos, DOB 11/05/1990, MRN P-2468. Allergy to penicillin. Phone: (555) 111-2222.",
        "Patient: David Kim, SSN 444-55-6666, referred by Dr. Amanda Torres. Insurance: Aetna #A-789012.",
    ]

    print(f"\n{'='*60}")
    print("Batch PHI Redaction")
    print(f"{'='*60}")

    # Redact PHI from all records at once (irreversible — for safe storage/logging)
    batch = blindfold.redact_batch(records, policy="hipaa_us")

    print(f"\nRedacted {len(batch.results)} patient records:\n")
    for i, result in enumerate(batch.results):
        print(f"  Record {i+1}:")
        print(f"    Original: {records[i][:70]}...")
        print(f"    Redacted: {result.text[:70]}...")
        print(f"    PHI removed: {result.entities_count} identifiers\n")


if __name__ == "__main__":
    example_patient_query()
    example_conversation()
    example_batch_records()
