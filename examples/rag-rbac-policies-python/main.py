"""
RAG with Role-Based PII Control (RBAC Policies)

Same vector store, same retrieval -- but different user roles see different
levels of PII protection. The role determines which Blindfold policy is
applied at query time before the LLM sees the prompt.

This is useful when multiple teams share a knowledge base but have different
data access requirements. For example, in a healthcare system:

  - Doctors need patient names and conditions, but not billing details
  - Nurses need names and conditions, but not SSNs or financial data
  - Billing staff need names and insurance info, but not clinical details
  - Researchers need fully de-identified data for analytics

The key insight is that the vector store holds the SAME documents for all
roles. The privacy boundary is applied at query time by choosing which
Blindfold entities to tokenize before the LLM call. Each role maps to a
different set of entities (or a built-in policy like "strict").

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py                  # run all roles
    python main.py --role doctor    # run a single role
"""

import argparse
import os

import chromadb
from blindfold import Blindfold
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Clients ---
blindfold = Blindfold(api_key=os.environ["BLINDFOLD_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- Sample healthcare patient records ---
PATIENT_RECORDS = [
    (
        "Patient Record #PR-2024-001: Sarah Chen (sarah.chen@email.com, "
        "+1-555-0142, SSN 412-55-6789, DOB 1985-03-15) was diagnosed with "
        "Type 2 Diabetes on 2024-01-15. Prescribed Metformin 500mg twice "
        "daily. Insurance: BlueCross policy BC-2847193. Primary care "
        "physician: Dr. James Wilson."
    ),
    (
        "Patient Record #PR-2024-002: Marcus Johnson (marcus.j@email.com, "
        "+1-555-0198, SSN 331-78-4521, DOB 1992-07-22) presented with acute "
        "bronchitis on 2024-02-03. Prescribed Azithromycin 250mg for 5 days. "
        "Insurance: Aetna policy AE-9182736. Referred by Dr. Lisa Park."
    ),
    (
        "Patient Record #PR-2024-003: Elena Rodriguez (elena.r@email.com, "
        "+1-555-0267, SSN 528-91-3456, DOB 1978-11-08) underwent knee "
        "replacement surgery on 2024-03-10. Post-op recovery normal. "
        "Insurance: UnitedHealth policy UH-5529384. Surgeon: Dr. Robert Kim. "
        "Billing: $47,500 pre-insurance."
    ),
    (
        "Patient Record #PR-2024-004: Sarah Chen (sarah.chen@email.com, "
        "+1-555-0142) follow-up visit on 2024-04-20. HbA1c improved from "
        "8.2% to 7.1%. Continuing Metformin, added Lisinopril 10mg for "
        "blood pressure management. Next appointment in 3 months."
    ),
    (
        "Patient Record #PR-2024-005: Marcus Johnson (marcus.j@email.com, "
        "+1-555-0198) emergency visit on 2024-05-15. Severe allergic reaction "
        "to shellfish. Administered epinephrine. Prescribed EpiPen for future "
        "emergencies. Insurance claim filed with Aetna policy AE-9182736."
    ),
]

# --- Role definitions ---
# Each role maps to a different set of entities to tokenize at query time.
#
# In production, you would create custom policies (role_doctor, role_nurse,
# etc.) via the Blindfold dashboard and use policy="role_doctor" instead of
# entities=[...]. Custom policies let you manage access centrally without
# code changes. This example uses the entities parameter so it works without
# any dashboard setup.
ROLES = {
    "doctor": {
        "policy": "role_doctor",
        "description": (
            "Full clinical access -- sees patient names, conditions, "
            "medications. Contact info and financial data redacted."
        ),
        "entities": [
            "email address",
            "phone number",
            "social security number",
            "credit card number",
            "iban",
        ],
    },
    "nurse": {
        "policy": "role_nurse",
        "description": (
            "Clinical care access -- sees names and conditions. "
            "No SSN, contact, or financial data."
        ),
        "entities": [
            "email address",
            "phone number",
            "social security number",
            "credit card number",
            "iban",
            "date of birth",
        ],
    },
    "billing": {
        "policy": "role_billing",
        "description": (
            "Financial access -- sees names and insurance info. "
            "Clinical details and contact info redacted."
        ),
        "entities": [
            "email address",
            "phone number",
            "social security number",
            "medical condition",
            "medication",
        ],
    },
    "researcher": {
        "policy": "strict",
        "description": (
            "De-identified access -- all PII removed. "
            "Content-based search only."
        ),
        # Uses policy="strict" which covers all 60+ entity types
        "entities": None,
    },
}

SYSTEM_PROMPT = (
    "You are a helpful medical records assistant. Answer the user's question "
    "based only on the provided context. Keep your answer concise and factual."
)

QUERIES = [
    "What conditions has Sarah Chen been treated for?",
    "What insurance does Marcus Johnson have?",
    "Which patients had surgery?",
]


# ---------------------------------------------------------------------------
# Ingestion: redact contact info, keep names for searchability
# ---------------------------------------------------------------------------

def ingest_records(collection) -> None:
    """Redact emails and phone numbers from patient records, store in ChromaDB.

    Names are preserved so they remain searchable via vector similarity.
    This is the same approach as Strategy A (selective redact).
    """
    print("=" * 70)
    print("INGESTION: Redact contact info, keep names for searchability")
    print("=" * 70)
    print()

    for idx, record in enumerate(PATIENT_RECORDS):
        # Redact only contact info -- names stay in plain text for search
        result = blindfold.redact(
            record,
            entities=["email address", "phone number"],
        )

        collection.add(documents=[result.text], ids=[f"record-{idx}"])

        print(f"  Record {idx + 1}: {result.entities_count} entities redacted")
        preview = result.text[:90] + "..." if len(result.text) > 90 else result.text
        print(f"    Stored: \"{preview}\"")
        print()

    print(f"  Total: {len(PATIENT_RECORDS)} records stored in ChromaDB")
    print()


# ---------------------------------------------------------------------------
# Query: apply role-specific tokenization before the LLM
# ---------------------------------------------------------------------------

def query_for_role(
    collection,
    question: str,
    role_name: str,
    show_tokenized_prompt: bool = False,
) -> str:
    """Search the vector store, tokenize with role-specific entities, query LLM.

    The role determines which entities are tokenized. A doctor sees names and
    clinical data but not contact info. A researcher sees nothing identifiable.
    """
    role = ROLES[role_name]

    # Step 1: Search with the original question -- names match directly
    results = collection.query(query_texts=[question], n_results=3)
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)

    # Step 2: Tokenize context + question with the role's entities
    # In production, use policy=role["policy"] with custom dashboard policies.
    # Here we use entities=[...] so no dashboard setup is needed.
    prompt_text = f"Context:\n{context}\n\nQuestion: {question}"

    if role_name == "researcher":
        # The researcher role uses the built-in "strict" policy which
        # tokenizes all 60+ entity types -- full de-identification.
        tokenized = blindfold.tokenize(prompt_text, policy="strict")
    else:
        # Other roles use an explicit entity list to control what is redacted.
        # In production: blindfold.tokenize(prompt_text, policy=role["policy"])
        tokenized = blindfold.tokenize(prompt_text, entities=role["entities"])

    if show_tokenized_prompt:
        print(f"    Tokenized prompt ({role_name}):")
        # Show a meaningful excerpt of the tokenized prompt
        lines = tokenized.text.split("\n")
        for line in lines[:6]:
            if line.strip():
                print(f"      {line[:100]}{'...' if len(line) > 100 else ''}")
        if len(lines) > 6:
            print(f"      ... ({len(lines)} lines total)")
        print()

    # Step 3: LLM call -- PII is tokenized according to role
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tokenized.text},
        ],
    )
    ai_response = completion.choices[0].message.content

    # Step 4: Detokenize -- restore original values for the end user
    final = blindfold.detokenize(ai_response, tokenized.mapping)
    return final.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG with Role-Based PII Control -- same data, different privacy levels",
    )
    parser.add_argument(
        "--role",
        choices=list(ROLES.keys()),
        default=None,
        help="Run a single role. Default: run all roles for comparison.",
    )
    args = parser.parse_args()

    roles_to_run = [args.role] if args.role else list(ROLES.keys())

    # --- Set up ChromaDB ---
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("patient_records")

    # --- Ingest records (once, shared by all roles) ---
    ingest_records(collection)

    # --- Print role overview ---
    print("=" * 70)
    print("ROLES")
    print("=" * 70)
    print()
    for name in roles_to_run:
        role = ROLES[name]
        print(f"  [{name}] {role['description']}")
        if role["entities"]:
            print(f"    Entities tokenized: {', '.join(role['entities'])}")
        else:
            print(f"    Policy: \"{role['policy']}\" (all entity types)")
    print()

    # --- Run queries ---
    print("=" * 70)
    print("QUERIES: Same question, different roles")
    print("=" * 70)
    print()

    for q_idx, question in enumerate(QUERIES):
        print(f"  Question: \"{question}\"")
        print()

        # Show the tokenized prompt only for the first query to illustrate
        # how different roles see different levels of redaction
        show_prompt = q_idx == 0

        if show_prompt:
            print("  What each role sends to the LLM (first query only):")
            print()

        for role_name in roles_to_run:
            answer = query_for_role(
                collection,
                question,
                role_name,
                show_tokenized_prompt=show_prompt,
            )
            label = f"[{role_name}]"
            print(f"    {label:<14} Final answer: \"{answer}\"")

        print()
        print("-" * 70)
        print()

    # --- Summary ---
    if len(roles_to_run) > 1:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
  Same vector store, same retrieval, same LLM -- different privacy levels.

  The role determines which Blindfold entities are tokenized before the
  prompt reaches the LLM. This means:

    - Doctors see clinical details but not contact info or financials
    - Nurses see clinical details but not SSNs, DOBs, or financials
    - Billing staff see insurance info but not clinical details
    - Researchers see fully de-identified data (policy="strict")

  In production, create custom policies via the Blindfold dashboard
  (role_doctor, role_nurse, etc.) and use policy="role_doctor" instead
  of entities=[...]. This moves access control to configuration rather
  than code, and lets you update policies without redeploying.
""")


if __name__ == "__main__":
    main()
