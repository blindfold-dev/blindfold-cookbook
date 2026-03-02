"""
RAG Strategy Comparison: Three PII Protection Approaches

Runs the SAME support tickets and queries through three strategies
side by side so you can see how each handles ingestion, search, and
LLM interaction.

Strategy A — Selective Redact (recommended):
  - Ingestion: redact contact info (emails, phones), keep names
  - Query: search with original question, single tokenize(context+question), LLM, detokenize

Strategy B — Tokenize with Stored Mapping:
  - Ingestion: tokenize everything, store mapping per document
  - Query: reverse-lookup names->tokens, search with tokenized query, LLM, detokenize
  - Problem: same person gets different tokens in each document

Strategy C — Consistent Token Registry:
  - Ingestion: use Blindfold to detect PII, then replace with consistent
    tokens from a global registry ("Hans Mueller" = <Person_1> everywhere)
  - Query: lookup names in registry, search with consistent tokens, LLM, reverse replace
  - Best searchability + zero PII in vector store, but more complex

Works in two modes:
  - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
  - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import argparse
import json
import os

import chromadb
from blindfold import Blindfold
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# API key is optional — omit it to run in local mode (regex-based, offline)
blindfold = Blindfold(api_key=os.environ.get("BLINDFOLD_API_KEY"))
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
        "Ticket #1004: Marie Dupont (marie.dupont@example.fr, "
        "+33 6 12 34 56 78) reports a second issue — her subscription was "
        "downgraded without notice. She expected Pro features but only has Basic."
    ),
]

SYSTEM_PROMPT = (
    "You are a helpful support assistant. Answer the user's question "
    "based only on the provided context. Keep your answer concise."
)


# ---------------------------------------------------------------------------
# Strategy A: Selective Redact
# ---------------------------------------------------------------------------

def strategy_a_ingest(chroma_client: chromadb.Client):
    """Redact contact info, keep names. Simple, stateless."""
    collection = chroma_client.create_collection("strategy_a")

    print("  Ingesting tickets (redact emails + phones, keep names)...")
    for idx, ticket in enumerate(SUPPORT_TICKETS):
        result = blindfold.redact(ticket, entities=["email address", "phone number"])
        collection.add(documents=[result.text], ids=[f"ticket-{idx}"])
        preview = result.text[:80] + "..."
        print(f"    Ticket {idx + 1}: {result.entities_count} entities redacted")
        print(f"      Stored: \"{preview}\"")

    return collection


def strategy_a_query(collection, question: str) -> str:
    """Search with original question, single tokenize, LLM, detokenize."""
    # Search — names match because they are in the vector store
    results = collection.query(query_texts=[question], n_results=2)
    context = "\n\n".join(results["documents"][0])

    print(f"  Retrieved {len(results['documents'][0])} chunks")
    for doc in results["documents"][0]:
        print(f"    \"{doc[:80]}...\"")

    # Single tokenize call on context + question
    prompt_text = f"Context:\n{context}\n\nQuestion: {question}"
    tokenized = blindfold.tokenize(prompt_text)
    print(f"\n  Tokenized prompt (first 120 chars):")
    print(f"    \"{tokenized.text[:120]}...\"")

    # LLM call
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tokenized.text},
        ],
    )
    ai_response = completion.choices[0].message.content
    print(f"\n  LLM response (tokenized): \"{ai_response}\"")

    # Detokenize
    final = blindfold.detokenize(ai_response, tokenized.mapping)
    return final.text


# ---------------------------------------------------------------------------
# Strategy B: Tokenize with Stored Mapping (per-document)
# ---------------------------------------------------------------------------

def strategy_b_ingest(chroma_client: chromadb.Client):
    """Tokenize everything, store per-document mapping."""
    collection = chroma_client.create_collection("strategy_b")
    mapping_store: dict[str, dict[str, str]] = {}

    print("  Ingesting tickets (tokenize everything, store mappings)...")
    for idx, ticket in enumerate(SUPPORT_TICKETS):
        result = blindfold.tokenize(ticket)
        doc_id = f"ticket-{idx}"
        collection.add(
            documents=[result.text],
            ids=[doc_id],
            metadatas=[{"mapping": json.dumps(result.mapping)}],
        )
        mapping_store[doc_id] = result.mapping
        preview = result.text[:80] + "..."
        print(f"    Ticket {idx + 1}: {len(result.mapping)} tokens created")
        print(f"      Stored: \"{preview}\"")

    return collection, mapping_store


def strategy_b_query(
    collection,
    mapping_store: dict[str, dict[str, str]],
    question: str,
) -> str:
    """Reverse-lookup names to tokens, search, LLM, detokenize."""
    # Build reverse lookup: real value -> list of (doc_id, token) pairs
    reverse_lookup: dict[str, list[tuple[str, str]]] = {}
    for doc_id, mapping in mapping_store.items():
        for token, real_value in mapping.items():
            reverse_lookup.setdefault(real_value, []).append((doc_id, token))

    print(f"  Reverse lookup has {len(reverse_lookup)} unique real values")

    # Replace real names in query with their tokens
    tokenized_query = question
    matched_tokens = []
    for real_value, entries in reverse_lookup.items():
        if real_value in tokenized_query:
            # Use the first token we find for this real value
            token = entries[0][1]
            tokenized_query = tokenized_query.replace(real_value, token)
            matched_tokens.append(f"{real_value} -> {token}")

    if matched_tokens:
        print(f"  Replaced in query: {', '.join(matched_tokens)}")
    else:
        print("  No known values found in query (content-based search only)")
    print(f"  Search query: \"{tokenized_query}\"")

    # Search — tokens match tokens in vector store
    results = collection.query(query_texts=[tokenized_query], n_results=2)
    context = "\n\n".join(results["documents"][0])

    print(f"\n  Retrieved {len(results['documents'][0])} chunks")
    for doc in results["documents"][0]:
        print(f"    \"{doc[:80]}...\"")

    # Merge mappings from all retrieved documents
    merged_mapping: dict[str, str] = {}
    for doc_id in results["ids"][0]:
        if doc_id in mapping_store:
            merged_mapping.update(mapping_store[doc_id])

    # LLM call — already tokenized context and query
    prompt_text = f"Context:\n{context}\n\nQuestion: {tokenized_query}"
    print(f"\n  Prompt to LLM (first 120 chars):")
    print(f"    \"{prompt_text[:120]}...\"")

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
    )
    ai_response = completion.choices[0].message.content
    print(f"\n  LLM response (tokenized): \"{ai_response}\"")

    # Detokenize using merged mappings
    final = blindfold.detokenize(ai_response, merged_mapping)
    return final.text


# ---------------------------------------------------------------------------
# Strategy C: Consistent Token Registry
# ---------------------------------------------------------------------------

class TokenRegistry:
    """Global registry that assigns consistent tokens across all documents.

    "Hans Mueller" is always <Person_1>, regardless of which document
    or when it was ingested. In production, back this with a database.
    """

    def __init__(self):
        # real_value -> consistent token
        self.registry: dict[str, str] = {}
        # consistent token -> real_value (reverse)
        self.reverse: dict[str, str] = {}
        # counters per entity type: "Person" -> 3 means next is <Person_4>
        self._counters: dict[str, int] = {}

    def _extract_entity_type(self, token: str) -> str:
        """Extract entity type from a Blindfold token like '<Person_1>'."""
        # Remove < and >, split on last _, take everything before the number
        inner = token.strip("<>")
        parts = inner.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else inner

    def get_or_create(self, real_value: str, blindfold_token: str) -> str:
        """Get the consistent token for a real value, or create one."""
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

        Sorts by length descending so "Marie Dupont" is replaced before "Marie".
        """
        result = text
        for real_value in sorted(self.registry, key=len, reverse=True):
            result = result.replace(real_value, self.registry[real_value])
        return result

    def restore_text(self, text: str) -> str:
        """Replace all consistent tokens with real values (detokenize)."""
        result = text
        for token in sorted(self.reverse, key=len, reverse=True):
            result = result.replace(token, self.reverse[token])
        return result


def strategy_c_ingest(chroma_client: chromadb.Client):
    """Use Blindfold to detect PII, then replace with consistent tokens."""
    collection = chroma_client.create_collection("strategy_c")
    registry = TokenRegistry()

    print("  Ingesting tickets (detect PII, apply consistent tokens)...")
    for idx, ticket in enumerate(SUPPORT_TICKETS):
        # Step 1: Use Blindfold to detect entities
        result = blindfold.tokenize(ticket)

        # Step 2: Register each entity with a consistent token
        for bf_token, real_value in result.mapping.items():
            registry.get_or_create(real_value, bf_token)

        # Step 3: Replace PII in the ORIGINAL text with consistent tokens
        consistent_text = registry.replace_in_text(ticket)

        collection.add(documents=[consistent_text], ids=[f"ticket-{idx}"])
        preview = consistent_text[:80] + "..."
        print(f"    Ticket {idx + 1}: {len(result.mapping)} entities detected")
        print(f"      Stored: \"{preview}\"")

    print(f"\n  Registry ({len(registry.registry)} entries):")
    for real_value, token in registry.registry.items():
        print(f"    {token} = \"{real_value}\"")

    return collection, registry


def strategy_c_query(
    collection,
    registry: TokenRegistry,
    question: str,
) -> str:
    """Replace known names with consistent tokens, search, LLM, restore."""
    # Replace any known real values in the question with their tokens
    tokenized_query = registry.replace_in_text(question)

    replacements = []
    for real_value, token in registry.registry.items():
        if real_value in question:
            replacements.append(f"{real_value} -> {token}")

    if replacements:
        print(f"  Replaced in query: {', '.join(replacements)}")
    else:
        print("  No known values found in query (content-based search only)")
    print(f"  Search query: \"{tokenized_query}\"")

    # Search — consistent tokens match consistently in vector store
    results = collection.query(query_texts=[tokenized_query], n_results=2)
    context = "\n\n".join(results["documents"][0])

    print(f"\n  Retrieved {len(results['documents'][0])} chunks")
    for doc in results["documents"][0]:
        print(f"    \"{doc[:80]}...\"")

    # LLM call — all PII replaced with consistent tokens
    prompt_text = f"Context:\n{context}\n\nQuestion: {tokenized_query}"
    print(f"\n  Prompt to LLM (first 120 chars):")
    print(f"    \"{prompt_text[:120]}...\"")

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
    )
    ai_response = completion.choices[0].message.content
    print(f"\n  LLM response (tokenized): \"{ai_response}\"")

    # Restore real values using the registry
    final = registry.restore_text(ai_response)
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

QUERIES = [
    "What was the issue reported by Hans Mueller?",
    "What problems did Marie Dupont have?",
    "Which tickets involved billing issues?",
]


def run_strategy_a(chroma_client: chromadb.Client):
    print("=" * 70)
    print("STRATEGY A: Selective Redact")
    print("  Contact info redacted, names kept in vector store")
    print("=" * 70)
    print()

    collection = strategy_a_ingest(chroma_client)
    print()

    for question in QUERIES:
        print(f"  Question: \"{question}\"")
        answer = strategy_a_query(collection, question)
        print(f"\n  Final answer: \"{answer}\"")
        print("-" * 50)


def run_strategy_b(chroma_client: chromadb.Client):
    print("=" * 70)
    print("STRATEGY B: Tokenize with Stored Mapping (per-document)")
    print("  Everything tokenized, mappings stored per document")
    print("  NOTE: same person gets DIFFERENT tokens in each document")
    print("=" * 70)
    print()

    collection, mapping_store = strategy_b_ingest(chroma_client)
    print()

    for question in QUERIES:
        print(f"  Question: \"{question}\"")
        answer = strategy_b_query(collection, mapping_store, question)
        print(f"\n  Final answer: \"{answer}\"")
        print("-" * 50)


def run_strategy_c(chroma_client: chromadb.Client):
    print("=" * 70)
    print("STRATEGY C: Consistent Token Registry")
    print("  Blindfold detects PII, app assigns consistent tokens")
    print("  Same person = same token in EVERY document")
    print("=" * 70)
    print()

    collection, registry = strategy_c_ingest(chroma_client)
    print()

    for question in QUERIES:
        print(f"  Question: \"{question}\"")
        answer = strategy_c_query(collection, registry, question)
        print(f"\n  Final answer: \"{answer}\"")
        print("-" * 50)


STRATEGY_MAP = {
    "a": run_strategy_a,
    "b": run_strategy_b,
    "c": run_strategy_c,
}


def main():
    parser = argparse.ArgumentParser(
        description="RAG Strategy Comparison: Three PII Protection Approaches",
    )
    parser.add_argument(
        "strategies",
        nargs="*",
        default=["a", "b", "c"],
        choices=["a", "b", "c"],
        metavar="STRATEGY",
        help="Strategies to run: a (selective redact), b (stored mapping), "
             "c (consistent registry). Default: all three.",
    )
    args = parser.parse_args()

    chroma_client = chromadb.Client()

    for i, key in enumerate(args.strategies):
        if i > 0:
            print()
        STRATEGY_MAP[key](chroma_client)

    if len(args.strategies) > 1:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
  Strategy A (Selective Redact):
    + Simple, stateless — no mapping storage needed
    + Each document ingested independently
    + Names match directly in vector search
    - Names are stored in the vector store (internal infra risk)
    - Contact info permanently lost

  Strategy B (Tokenize with Stored Mapping):
    + Zero PII in vector store — tokens only
    + Fully reversible — can restore all original data
    - Same person gets DIFFERENT tokens per document (<Person_1> in doc A,
      <Person_3> in doc B) — reverse lookup has collisions
    - Requires per-document mapping storage
    - Grows in complexity as documents increase

  Strategy C (Consistent Token Registry):
    + Zero PII in vector store — tokens only
    + Same person = same token EVERYWHERE (<Person_1> always = Hans Mueller)
    + Perfect vector search — tokens match exactly across all documents
    + Simple reverse lookup — one token per real value
    + Fully reversible
    - Requires a persistent token registry (DB)
    - Uses Blindfold as PII detector, does replacement itself
    - detokenize() not used — app does its own string replacement
    - Registry must be available at both ingestion and query time

  All three strategies protect PII at the LLM boundary — the external
  third party (OpenAI, Anthropic, etc.) never sees real personal data.
""")


if __name__ == "__main__":
    main()
