"""
RAG with Selective Redaction: Protect contact info, keep names searchable.

Strategy: At ingestion time, redact only contact info (emails, phones) from
documents using blindfold.redact(). Names are kept in the vector store so
they match directly during search. At query time, the retrieved context and
the user's question are tokenized together in a single call before the LLM,
then the response is detokenized to restore original values.

This is the simplest PII protection strategy for RAG because it requires no
mapping storage, no reverse lookups, and no registry management.

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import os

import chromadb
from blindfold import Blindfold
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Clients ---
blindfold = Blindfold(api_key=os.environ["BLINDFOLD_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- Sample data ---
SUPPORT_TICKETS = [
    "Ticket #1001: Customer Hans Mueller (hans.mueller@example.de, +49 151 12345678) reported a billing error on invoice INV-2024-0047. He was charged twice for the Pro plan in January. Refund requested.",
    "Ticket #1002: Marie Dupont (marie.dupont@example.fr, +33 6 12 34 56 78) cannot access her dashboard after a password reset. She tried three times and is now locked out. Needs urgent unlock.",
    "Ticket #1003: Lars Johansson (lars.johansson@example.se, +46 70 123 4567) asked to export all his personal data under GDPR. He wants a full copy within 30 days as required by regulation.",
    "Ticket #1004: Marie Dupont (marie.dupont@example.fr, +33 6 12 34 56 78) reports a second issue — her subscription was downgraded without notice. She expected Pro features but only has Basic.",
]

SYSTEM_PROMPT = (
    "You are a helpful support assistant. Answer the user's question "
    "based only on the provided context. Keep your answer concise."
)

QUERIES = [
    "What was the issue reported by Hans Mueller?",
    "What problems did Marie Dupont have?",
    "Which tickets involved billing issues?",
]


# ---------------------------------------------------------------------------
# Ingestion: redact contact info, keep names
# ---------------------------------------------------------------------------

def ingest_tickets(collection) -> None:
    """Redact emails and phone numbers from tickets, store in ChromaDB.

    Names are preserved so they remain searchable via vector similarity.
    """
    print("=" * 60)
    print("INGESTION: Redact contact info, keep names")
    print("=" * 60)
    print()

    for idx, ticket in enumerate(SUPPORT_TICKETS):
        doc_id = f"ticket-{idx}"

        # Redact only contact info -- names stay in plain text
        result = blindfold.redact(ticket, entities=["email address", "phone number"])

        collection.add(documents=[result.text], ids=[doc_id])

        print(f"  Ticket {idx + 1}: {result.entities_count} entities redacted")
        preview = result.text[:90] + "..." if len(result.text) > 90 else result.text
        print(f"    Stored: \"{preview}\"")
        print()

    print(f"  Total: {len(SUPPORT_TICKETS)} tickets stored in ChromaDB")
    print()


# ---------------------------------------------------------------------------
# Query: search with original question, tokenize, LLM, detokenize
# ---------------------------------------------------------------------------

def query_rag(collection, question: str) -> str:
    """Search, tokenize context+question, generate answer, detokenize.

    Since names are in the vector store, the original question matches
    directly. Context and question are tokenized in a single call so the
    LLM sees consistent token numbering.
    """
    print(f"  Question: \"{question}\"")
    print()

    # Step 1: Search with the original question -- names match directly
    results = collection.query(query_texts=[question], n_results=2)
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)

    print(f"  Retrieved {len(context_chunks)} chunks:")
    for chunk in context_chunks:
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"    \"{preview}\"")
    print()

    # Step 2: Single tokenize call on context + question
    # This ensures consistent token numbering (e.g., <Person_1> in both
    # context and question refers to the same person)
    prompt_text = f"Context:\n{context}\n\nQuestion: {question}"
    tokenized = blindfold.tokenize(prompt_text)

    print(f"  Tokenized prompt (preview):")
    print(f"    \"{tokenized.text[:120]}...\"")
    print()

    # Step 3: LLM call -- no real PII in the prompt
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tokenized.text},
        ],
    )
    ai_response = completion.choices[0].message.content

    print(f"  LLM response (tokenized):")
    print(f"    \"{ai_response}\"")
    print()

    # Step 4: Detokenize -- restore original values for the end user
    final = blindfold.detokenize(ai_response, tokenized.mapping)

    print(f"  Final answer:")
    print(f"    \"{final.text}\"")

    return final.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("selective_redact")

    # Ingest all tickets
    ingest_tickets(collection)

    # Run each query
    for i, question in enumerate(QUERIES):
        print("-" * 60)
        query_rag(collection, question)
        print()


if __name__ == "__main__":
    main()
