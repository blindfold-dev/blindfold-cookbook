"""
RAG with Consistent Token Registry: Same person = same token everywhere.

Strategy: Use Blindfold as a PII detector (via tokenize()), then assign
consistent tokens from a global registry. Unlike per-document tokenization,
"Hans Mueller" is always <Person_1> regardless of which document or when
it was ingested. This gives you zero PII in the vector store AND perfect
search quality because tokens match consistently across all documents.

The TokenRegistry class manages the mapping between real values and their
consistent tokens. In production, back this with a database for persistence.

Works in two modes:
  - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
  - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations

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
# Token Registry: assigns consistent tokens across all documents
# ---------------------------------------------------------------------------

class TokenRegistry:
    """Global registry that assigns consistent tokens across all documents.

    "Hans Mueller" is always <Person_1>, regardless of which document
    or when it was ingested. In production, back this with a database.
    """

    def __init__(self):
        self.registry: dict[str, str] = {}  # real_value -> consistent token
        self.reverse: dict[str, str] = {}   # consistent token -> real_value
        self._counters: dict[str, int] = {} # entity_type -> next count

    def _extract_entity_type(self, token: str) -> str:
        """Extract entity type from a Blindfold token like '<Person_1>'.

        Strips angle brackets, splits on the last underscore, and returns
        the entity type portion (e.g., "Person", "Email_Address").
        """
        inner = token.strip("<>")
        parts = inner.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else inner

    def get_or_create(self, real_value: str, blindfold_token: str) -> str:
        """Get the consistent token for a real value, or create one.

        If this real value has been seen before, returns the same token.
        Otherwise, creates a new token with a sequential counter for that
        entity type (e.g., <Person_1>, <Person_2>, etc.).
        """
        if real_value in self.registry:
            return self.registry[real_value]

        entity_type = self._extract_entity_type(blindfold_token)
        count = self._counters.get(entity_type, 0) + 1
        self._counters[entity_type] = count
        consistent_token = f"<{entity_type}_{count}>"

        self.registry[real_value] = consistent_token
        self.reverse[consistent_token] = real_value
        return consistent_token

    def replace_in_text(self, text: str) -> str:
        """Replace all known real values in text with their consistent tokens.

        Sorts by length descending so "Marie Dupont" is replaced before
        "Marie" (prevents partial replacements).
        """
        result = text
        for real_value in sorted(self.registry, key=len, reverse=True):
            result = result.replace(real_value, self.registry[real_value])
        return result

    def restore_text(self, text: str) -> str:
        """Replace all consistent tokens with real values (detokenize).

        Sorts by length descending for the same reason as replace_in_text.
        """
        result = text
        for token in sorted(self.reverse, key=len, reverse=True):
            result = result.replace(token, self.reverse[token])
        return result


# ---------------------------------------------------------------------------
# Ingestion: detect PII with Blindfold, assign consistent tokens
# ---------------------------------------------------------------------------

def ingest_tickets(collection, registry: TokenRegistry) -> None:
    """Use Blindfold to detect PII, then replace with consistent tokens.

    For each ticket:
    1. Call blindfold.tokenize() to detect all PII entities
    2. Register each entity with the global TokenRegistry
    3. Apply consistent tokens to the ORIGINAL text
    4. Store the consistently-tokenized text in ChromaDB
    """
    print("=" * 60)
    print("INGESTION: Detect PII, assign consistent tokens")
    print("=" * 60)
    print()

    for idx, ticket in enumerate(SUPPORT_TICKETS):
        doc_id = f"ticket-{idx}"

        # Step 1: Use Blindfold to detect entities in the ticket
        result = blindfold.tokenize(ticket)

        # Step 2: Register each detected entity with a consistent token
        for bf_token, real_value in result.mapping.items():
            consistent = registry.get_or_create(real_value, bf_token)

        # Step 3: Replace PII in the ORIGINAL text with consistent tokens
        # (not the Blindfold-tokenized text, which has per-call tokens)
        consistent_text = registry.replace_in_text(ticket)

        # Step 4: Store in ChromaDB
        collection.add(documents=[consistent_text], ids=[doc_id])

        print(f"  Ticket {idx + 1}: {len(result.mapping)} entities detected")
        preview = consistent_text[:90] + "..." if len(consistent_text) > 90 else consistent_text
        print(f"    Stored: \"{preview}\"")
        print()

    # Print the full registry dump
    print("  " + "-" * 56)
    print(f"  REGISTRY DUMP ({len(registry.registry)} entries):")
    print("  " + "-" * 56)
    for real_value, token in registry.registry.items():
        print(f"    {token:25s} = \"{real_value}\"")
    print()

    # Highlight consistency: Marie Dupont has the same token in both tickets
    print("  Consistency check: Marie Dupont appears in tickets #2 and #4")
    print(f"    Both use: {registry.registry.get('Marie Dupont', 'N/A')}")
    print()


# ---------------------------------------------------------------------------
# Query: replace names with consistent tokens, search, LLM, restore
# ---------------------------------------------------------------------------

def query_rag(collection, registry: TokenRegistry, question: str) -> str:
    """Replace known names with consistent tokens, search, LLM, restore.

    Because the registry maps real values to consistent tokens, and the
    same tokens exist in the vector store, search quality is excellent.
    """
    print(f"  Question: \"{question}\"")
    print()

    # Step 1: Replace any known real values in the question
    tokenized_query = registry.replace_in_text(question)

    replacements = []
    for real_value, token in registry.registry.items():
        if real_value in question:
            replacements.append(f"\"{real_value}\" -> {token}")

    if replacements:
        print(f"  Replacements in query:")
        for r in replacements:
            print(f"    {r}")
    else:
        print("  No known values found in query (content-based search only)")
    print(f"  Search query: \"{tokenized_query}\"")
    print()

    # Step 2: Search the vector store -- consistent tokens match perfectly
    results = collection.query(query_texts=[tokenized_query], n_results=2)
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)

    print(f"  Retrieved {len(context_chunks)} chunks:")
    for chunk in context_chunks:
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"    \"{preview}\"")
    print()

    # Step 3: LLM call -- all PII replaced with consistent tokens
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

    # Step 4: Restore real values using the registry (not blindfold.detokenize)
    final = registry.restore_text(ai_response)

    print(f"  Final answer:")
    print(f"    \"{final}\"")

    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("consistent_registry")
    registry = TokenRegistry()

    # Ingest all tickets with consistent token assignment
    ingest_tickets(collection, registry)

    # Run each query
    for i, question in enumerate(QUERIES):
        print("-" * 60)
        query_rag(collection, registry, question)
        print()


if __name__ == "__main__":
    main()
