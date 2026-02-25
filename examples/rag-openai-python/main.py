"""
PII-safe RAG pipeline with OpenAI, ChromaDB, and Blindfold.

Two layers of PII protection:
  1. Ingestion-time: Support tickets are redacted before embedding and storage.
  2. Query-time: User questions are tokenized, the LLM generates an answer
     from redacted context, and the response is detokenized to restore
     original values for the end user.

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

blindfold = Blindfold(api_key=os.environ["BLINDFOLD_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SUPPORT_TICKETS = [
    (
        "Ticket #1001: Customer Hans Mueller (hans.mueller@example.de, "
        "+49 151 12345678) reported a billing error on invoice INV-2024-0047. "
        "He was charged twice for the Pro plan in January. Refund requested."
    ),
    (
        "Ticket #1002: Marie Dupont (marie.dupont@example.fr, "
        "+33 6 12 34 56 78) cannot access her dashboard after a password reset. "
        "She tried three times and is now locked out. Needs urgent unlock."
    ),
    (
        "Ticket #1003: Lars Johansson (lars.johansson@example.se, "
        "+46 70 123 4567) asked to export all his personal data under GDPR. "
        "He wants a full copy within 30 days as required by regulation."
    ),
    (
        "Ticket #1004: Sofia Garcia (sofia.garcia@example.es, "
        "+34 612 345 678) reported that her API key was exposed in a public "
        "repo. Key has been rotated. She requests confirmation that the old "
        "key is fully revoked and an audit of recent usage."
    ),
]


def ingest_tickets(collection):
    """Redact PII from support tickets, split, and store in ChromaDB."""
    print("=== Ingestion ===")
    print(f"Redacting PII from {len(SUPPORT_TICKETS)} support tickets...\n")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    all_documents = []
    all_ids = []

    for idx, ticket in enumerate(SUPPORT_TICKETS, start=1):
        result = blindfold.redact(ticket)
        print(f"  Ticket {idx}: {result.entities_count} entities redacted")

        chunks = splitter.split_text(result.text)
        for chunk_idx, chunk in enumerate(chunks):
            all_documents.append(chunk)
            all_ids.append(f"ticket_{idx}_chunk_{chunk_idx}")

    collection.add(documents=all_documents, ids=all_ids)
    print(f"\nStored {len(all_documents)} chunks in ChromaDB\n")


def query_rag(collection, user_question: str) -> str:
    """Tokenize the user question, retrieve context, generate, detokenize."""
    print("=== Query ===")
    print(f'User question: "{user_question}"\n')

    # Step 1: Tokenize PII in the question
    tokenized = blindfold.tokenize(user_question)
    print(f'Tokenized question: "{tokenized.text}"\n')

    # Step 2: Retrieve relevant context from ChromaDB
    results = collection.query(query_texts=[tokenized.text], n_results=2)
    context_chunks = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(context_chunks)

    print("Retrieved context (redacted):")
    for chunk in context_chunks:
        preview = chunk[:90] + "..." if len(chunk) > 90 else chunk
        print(f'  "{preview}"')
    print()

    # Step 3: Generate answer with OpenAI
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful support assistant. Answer the user's question "
                "based only on the provided context. Keep your answer concise.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {tokenized.text}",
            },
        ],
    )
    ai_response = completion.choices[0].message.content
    print(f'AI response (with tokens): "{ai_response}"\n')

    # Step 4: Detokenize to restore original PII for the end user
    final = blindfold.detokenize(ai_response, tokenized.mapping)
    print(f'Final response: "{final.text}"\n')
    return final.text


def main():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("support_tickets")

    ingest_tickets(collection)
    query_rag(collection, "What was the issue reported by Hans Mueller?")


if __name__ == "__main__":
    main()
