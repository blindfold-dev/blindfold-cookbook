"""
GDPR-Compliant Customer Support RAG Pipeline

Multi-turn customer support chatbot with EU sample tickets.
Uses the gdpr_eu policy and EU region to ensure GDPR compliance.
Mapping is accumulated across turns for consistent detokenization.

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import os

import chromadb
from blindfold import Blindfold
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

# EU support tickets with German, French, Spanish, and English PII
EU_SUPPORT_TICKETS = [
    (
        "Ticket #EU-2001: Kunde Hans Mueller (hans.mueller@example.de, "
        "+49 170 9876543) meldet einen Abrechnungsfehler. Konto wurde am "
        "2026-01-20 doppelt belastet. IBAN: DE89 3704 0044 0532 0130 00. "
        "Status: Ruckerstattung eingeleitet."
    ),
    (
        "Ticket #EU-2002: Cliente Marie Dupont (marie.dupont@example.fr, "
        "+33 6 12 34 56 78) demande un export de donnees conformement a "
        "l'article 15 du RGPD. Adresse: 42 Rue de Rivoli, 75001 Paris. "
        "Status: Export envoye par email."
    ),
    (
        "Ticket #EU-2003: Cliente Sofia Garcia (sofia.garcia@example.es, "
        "+34 612 345 678) reporta que no puede acceder a su panel de control "
        "desde el 2026-02-01. DNI: 12345678Z. "
        "Status: Problema resuelto, conflicto de cache."
    ),
    (
        "Ticket #EU-2004: Customer Emma Wilson (emma.wilson@example.co.uk, "
        "+44 20 7946 0958) reports subscription downgrade not reflected in "
        "billing. Previous charge: EUR 29.99, expected: EUR 14.99. "
        "Card ending 4821. Status: Billing adjusted, partial refund issued."
    ),
]


class CustomerSupportRAG:
    """GDPR-compliant customer support RAG with multi-turn conversation."""

    def __init__(self):
        self.blindfold = Blindfold(
            api_key=os.environ["BLINDFOLD_API_KEY"],
            region="eu",
        )
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.chroma = chromadb.Client()
        self.collection = self.chroma.create_collection("eu_support_tickets")
        self.conversation_history: list[dict] = []
        self.accumulated_mapping: dict[str, str] = {}

    def ingest_tickets(self, tickets: list[str]) -> None:
        """Redact PII from tickets and store in ChromaDB."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        print("=== Ingesting EU Support Tickets ===")
        all_chunks = []
        for i, ticket in enumerate(tickets):
            result = self.blindfold.redact(ticket, policy="gdpr_eu")
            entities = [e.type for e in result.detected_entities]
            print(f"  Ticket {i + 1}: {result.entities_count} entities redacted {entities}")
            chunks = splitter.split_text(result.text)
            all_chunks.extend(chunks)

        self.collection.add(
            documents=all_chunks,
            ids=[f"chunk-{i}" for i in range(len(all_chunks))],
        )
        print(f"Stored {len(all_chunks)} chunks in ChromaDB\n")

    def query(self, question: str) -> str:
        """Process a user question with GDPR-safe tokenization."""
        # Tokenize the question
        tokenized = self.blindfold.tokenize(question, policy="gdpr_eu")
        self.accumulated_mapping.update(tokenized.mapping)

        # Retrieve relevant chunks
        results = self.collection.query(
            query_texts=[tokenized.text],
            n_results=3,
        )
        context = "\n\n".join(results["documents"][0])

        # Build conversation with system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a GDPR-aware customer support assistant. "
                    "Answer questions using only the provided context. "
                    "Be concise and helpful.\n\n"
                    f"Context:\n{context}"
                ),
            },
        ]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": tokenized.text})

        # Get AI response
        completion = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        ai_response = completion.choices[0].message.content

        # Store in conversation history (tokenized)
        self.conversation_history.append({"role": "user", "content": tokenized.text})
        self.conversation_history.append({"role": "assistant", "content": ai_response})

        # Detokenize for the user
        restored = self.blindfold.detokenize(ai_response, self.accumulated_mapping)
        return restored.text


def main():
    rag = CustomerSupportRAG()
    rag.ingest_tickets(EU_SUPPORT_TICKETS)

    questions = [
        "What happened with Hans Mueller's billing issue?",
        "Did Marie Dupont's data export request get resolved?",
        "Which tickets are still open or had payment-related issues?",
    ]

    print("=== Multi-Turn Customer Support Chat ===\n")
    for question in questions:
        print(f"Customer: {question}")
        response = rag.query(question)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
