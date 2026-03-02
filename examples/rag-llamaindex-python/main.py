"""
LlamaIndex RAG Pipeline with Blindfold PII Protection

Contact info is redacted at ingestion (names kept for searchability).
At query time, context and question are tokenized in a single call
before reaching the LLM, then the response is detokenized.

Works in two modes:
  - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
  - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import os

from blindfold import Blindfold
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# API key is optional — omit it to run in local mode (regex-based, offline)
blindfold = Blindfold(api_key=os.environ.get("BLINDFOLD_API_KEY"))

SUPPORT_TICKETS = [
    Document(text=(
        "Ticket #1001: Customer John Smith (john.smith@example.com, +1-555-0123) "
        "reported a billing discrepancy. Account was charged $49.99 twice on "
        "2026-01-15. Resolution: refund issued within 24 hours."
    )),
    Document(text=(
        "Ticket #1002: Maria Garcia (maria.garcia@example.es, +34 612 345 678) "
        "requested a data export under GDPR Art. 15. Export was generated and "
        "sent to her email within the required 30-day period."
    )),
    Document(text=(
        "Ticket #1003: Customer Li Wei (li.wei@example.cn, +86 138 0013 8000) "
        "reported intermittent API timeouts since 2026-02-01. Root cause: "
        "regional DNS misconfiguration. Fixed within 4 hours, SLA credit applied."
    )),
    Document(text=(
        "Ticket #1004: Emma Johnson (emma.johnson@example.co.uk, +44 20 7946 0958) "
        "could not access her dashboard after a password reset. Issue traced to "
        "a browser cache conflict. Resolved by clearing session cookies."
    )),
]


def main():
    # Configure LlamaIndex settings
    llm = OpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    Settings.llm = llm
    Settings.embed_model = OpenAIEmbedding(api_key=os.environ["OPENAI_API_KEY"])

    # Redact contact info from documents before indexing — keep names searchable
    print("=== Ingestion ===")
    safe_documents = []
    for doc in SUPPORT_TICKETS:
        result = blindfold.redact(doc.text, entities=["email address", "phone number"])
        print(f"  Redacted {result.entities_count} entities from ticket")
        safe_documents.append(Document(text=result.text))

    # Build index from selectively redacted documents
    index = VectorStoreIndex.from_documents(safe_documents)
    print(f"Indexed {len(safe_documents)} documents\n")

    # Query: retrieve first, then single tokenize call before LLM
    print("=== Query ===")
    question = "What happened with John Smith's billing issue?"
    print(f"Original question: {question}\n")

    # Step 1: Retrieve with original question — names match in vector store
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(question)
    context = "\n\n".join(node.text for node in nodes)

    print("Retrieved context:")
    for node in nodes:
        preview = node.text[:90] + "..." if len(node.text) > 90 else node.text
        print(f'  "{preview}"')
    print()

    # Step 2: Single tokenize call — consistent token numbering
    prompt_text = f"Context:\n{context}\n\nQuestion: {question}"
    tokenized = blindfold.tokenize(prompt_text)

    # Step 3: LLM call with tokenized prompt — no PII
    response = llm.complete(
        f"You are a helpful support assistant. Answer the user's question "
        f"based only on the provided context. Keep your answer concise.\n\n"
        f"{tokenized.text}"
    )
    ai_response = str(response)
    print(f"AI response (with tokens): {ai_response}\n")

    # Step 4: Detokenize to restore real names
    final = blindfold.detokenize(ai_response, tokenized.mapping)
    print(f"Answer: {final.text}")


if __name__ == "__main__":
    main()
