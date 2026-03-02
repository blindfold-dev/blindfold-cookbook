"""
RAG with Stored Token Mapping: Tokenize everything, store per-document mappings.

Strategy: At ingestion time, tokenize all PII in each document using
blindfold.tokenize() and store the mapping alongside the tokenized text.
At query time, build a reverse lookup from all stored mappings to translate
real names in the user's question into their token equivalents, search the
vector store with the tokenized query, merge mappings from retrieved docs,
call the LLM, and detokenize the response.

This strategy gives you zero PII in the vector store (full tokenization),
but has a trade-off: the same person gets different tokens in each document
because each tokenize() call generates independent token assignments.

Works in two modes:
  - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
  - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import json
import os

import chromadb
from blindfold import Blindfold
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Clients ---
# API key is optional — omit it to run in local mode (regex-based, offline)
blindfold = Blindfold(api_key=os.environ.get("BLINDFOLD_API_KEY"))
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
# Ingestion: tokenize everything, store mapping per document
# ---------------------------------------------------------------------------

def ingest_tickets(collection) -> dict[str, dict[str, str]]:
    """Tokenize all PII in each ticket, store text and mapping in ChromaDB.

    Returns a mapping_store dict keyed by document ID, containing each
    document's token-to-real-value mapping.
    """
    print("=" * 60)
    print("INGESTION: Tokenize everything, store per-document mappings")
    print("=" * 60)
    print()

    mapping_store: dict[str, dict[str, str]] = {}

    for idx, ticket in enumerate(SUPPORT_TICKETS):
        doc_id = f"ticket-{idx}"

        # Tokenize the entire ticket -- all PII replaced with tokens
        result = blindfold.tokenize(ticket)

        # Store tokenized text in ChromaDB with the mapping as metadata
        collection.add(
            documents=[result.text],
            ids=[doc_id],
            metadatas=[{"mapping": json.dumps(result.mapping)}],
        )

        # Keep mapping in memory for query-time reverse lookup
        mapping_store[doc_id] = result.mapping

        print(f"  Ticket {idx + 1}: {len(result.mapping)} tokens created")
        preview = result.text[:90] + "..." if len(result.text) > 90 else result.text
        print(f"    Stored: \"{preview}\"")

        # Show the mapping for this document
        for token, real_value in result.mapping.items():
            print(f"    {token} = \"{real_value}\"")
        print()

    print(f"  Total: {len(SUPPORT_TICKETS)} tickets stored in ChromaDB")
    print()

    # Show the cross-document inconsistency (same person, different tokens)
    print("  NOTE: Same person gets DIFFERENT tokens per document:")
    for doc_id, mapping in mapping_store.items():
        for token, real_value in mapping.items():
            if "Marie Dupont" in real_value or "marie.dupont" in real_value:
                print(f"    {doc_id}: {token} = \"{real_value}\"")
    print()

    return mapping_store


# ---------------------------------------------------------------------------
# Query: reverse lookup, tokenize query, search, merge mappings, LLM
# ---------------------------------------------------------------------------

def query_rag(
    collection,
    mapping_store: dict[str, dict[str, str]],
    question: str,
) -> str:
    """Build reverse lookup, replace names in query, search, LLM, detokenize.

    The reverse lookup maps each real value to a list of (doc_id, token) pairs.
    When the user mentions a real name, we replace it with one of its known
    tokens so the vector search has a better chance of matching.
    """
    print(f"  Question: \"{question}\"")
    print()

    # Step 1: Build reverse lookup from all stored mappings
    # Maps real_value -> list of (doc_id, token) pairs
    reverse_lookup: dict[str, list[tuple[str, str]]] = {}
    for doc_id, mapping in mapping_store.items():
        for token, real_value in mapping.items():
            reverse_lookup.setdefault(real_value, []).append((doc_id, token))

    print(f"  Reverse lookup: {len(reverse_lookup)} unique real values")

    # Step 2: Replace real names in the question with their tokens
    tokenized_query = question
    replacements_made = []
    for real_value, entries in reverse_lookup.items():
        if real_value in tokenized_query:
            # Use the first known token for this real value
            token = entries[0][1]
            tokenized_query = tokenized_query.replace(real_value, token)
            all_tokens = [f"{e[1]} ({e[0]})" for e in entries]
            replacements_made.append(
                f"\"{real_value}\" -> {token} (also known as: {', '.join(all_tokens)})"
            )

    if replacements_made:
        print(f"  Replacements in query:")
        for r in replacements_made:
            print(f"    {r}")
    else:
        print("  No known values found in query (content-based search only)")
    print(f"  Search query: \"{tokenized_query}\"")
    print()

    # Step 3: Search the vector store with the tokenized query
    results = collection.query(query_texts=[tokenized_query], n_results=2)
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)

    print(f"  Retrieved {len(context_chunks)} chunks:")
    for chunk in context_chunks:
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"    \"{preview}\"")
    print()

    # Step 4: Merge mappings from all retrieved documents
    # This gives the detokenize call all possible token->value pairs
    merged_mapping: dict[str, str] = {}
    for doc_id in results["ids"][0]:
        if doc_id in mapping_store:
            merged_mapping.update(mapping_store[doc_id])

    print(f"  Merged mapping ({len(merged_mapping)} entries from {len(results['ids'][0])} docs)")
    print()

    # Step 5: LLM call -- context is already tokenized from ingestion
    prompt_text = f"Context:\n{context}\n\nQuestion: {tokenized_query}"
    print(f"  Prompt to LLM (preview):")
    print(f"    \"{prompt_text[:120]}...\"")
    print()

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
    )
    ai_response = completion.choices[0].message.content

    print(f"  LLM response (tokenized):")
    print(f"    \"{ai_response}\"")
    print()

    # Step 6: Detokenize using the merged mapping
    final = blindfold.detokenize(ai_response, merged_mapping)

    print(f"  Final answer:")
    print(f"    \"{final.text}\"")

    return final.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("stored_mapping")

    # Ingest all tickets and get the mapping store
    mapping_store = ingest_tickets(collection)

    # Run each query
    for i, question in enumerate(QUERIES):
        print("-" * 60)
        query_rag(collection, mapping_store, question)
        print()


if __name__ == "__main__":
    main()
