"""
HIPAA-Compliant Healthcare Chatbot with Blindfold

Tokenizes PHI with the hipaa_us policy and US region before sending
to OpenAI. Supports both single queries and multi-turn conversations.

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
        region="us",
    )
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return blindfold, openai


class HealthcareChatbot:
    """A HIPAA-compliant chatbot that protects PHI in every message."""

    def __init__(self):
        self.blindfold, self.openai = create_clients()
        self.conversation: list[dict] = [
            {
                "role": "system",
                "content": "You are a helpful healthcare assistant. Use patient identifier "
                           "tokens exactly as given. Never ask for real patient information.",
            }
        ]
        self.mapping: dict[str, str] = {}

    def chat(self, user_message: str) -> str:
        # 1. Tokenize PHI using the hipaa_us policy
        tokenized = self.blindfold.tokenize(user_message, policy="hipaa_us")

        # 2. Accumulate mapping across turns
        self.mapping.update(tokenized.mapping)
        self.conversation.append({"role": "user", "content": tokenized.text})

        # 3. Send to OpenAI — only tokens, never real PHI
        completion = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation,
        )
        ai_response = completion.choices[0].message.content

        # 4. Store tokenized response in history
        self.conversation.append({"role": "assistant", "content": ai_response})

        # 5. Detokenize for display to clinician
        restored = self.blindfold.detokenize(ai_response, self.mapping)
        return restored.text


if __name__ == "__main__":
    blindfold, openai_client = create_clients()

    # Single query example
    note = (
        "Patient Sarah Johnson (DOB 03/15/1978, SSN 123-45-6789, "
        "MRN P-4532) presented at Memorial Hospital on 02/10/2026 "
        "with chest pain. Contact: sarah.johnson@email.com, "
        "phone (555) 234-5678. Insurance ID: BC-9876543."
    )

    tokenized = blindfold.tokenize(note, policy="hipaa_us")
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this clinical note briefly."},
            {"role": "user", "content": tokenized.text},
        ],
    )
    restored = blindfold.detokenize(
        completion.choices[0].message.content, tokenized.mapping
    )
    print(restored.text)

    # Multi-turn conversation example
    bot = HealthcareChatbot()
    turns = [
        "Look up patient record for James Wilson, MRN P-7891, DOB 07/22/1965. "
        "He's at 123 Oak Street, Springfield. SSN 987-65-4321.",
        "What medications should we check for interactions given his age and symptoms?",
        "Please draft a referral note for Dr. Emily Chen at Springfield Cardiology.",
    ]
    for turn in turns:
        response = bot.chat(turn)
        print(response)
