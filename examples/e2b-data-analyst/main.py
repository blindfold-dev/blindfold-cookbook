"""
Blindfold + E2B: PII-Safe AI Data Analyst

Demonstrates how to use Blindfold and E2B together so an AI can write
data-analysis code without ever seeing real personal data:

1. Tokenize a CSV with Blindfold — PII is replaced with safe tokens
2. Send the tokenized CSV to OpenAI — it writes analysis code based on
   column structure and token placeholders, never real names or emails
3. Execute the AI-generated code in an E2B sandbox with the ORIGINAL data
   — the analysis runs on real values for accurate results

The AI writes correct pandas code because the column names and structure
are preserved. It just never sees the actual PII.

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import os

from blindfold import Blindfold
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from openai import OpenAI

load_dotenv()

# Sample healthcare dataset with PII
SAMPLE_CSV = """name,email,ssn,age,diagnosis,medication,doctor,city
John Smith,john.smith@email.com,123-45-6789,45,Type 2 Diabetes,Metformin 500mg,Dr. Sarah Johnson,Boston
Maria Garcia,maria.garcia@company.org,987-65-4321,32,Hypertension,Lisinopril 10mg,Dr. Michael Chen,New York
Robert Wilson,rwilson@hospital.net,456-78-9012,58,Type 2 Diabetes,Metformin 1000mg,Dr. Sarah Johnson,Boston
Emily Davis,emily.d@example.com,234-56-7890,27,Asthma,Albuterol Inhaler,Dr. Lisa Park,Chicago
James Brown,jbrown@mail.com,345-67-8901,51,Hypertension,Amlodipine 5mg,Dr. Michael Chen,New York
Anna Martinez,anna.m@health.org,567-89-0123,39,Asthma,Fluticasone Inhaler,Dr. Lisa Park,San Francisco
David Lee,david.lee@work.com,678-90-1234,63,Type 2 Diabetes,Insulin Glargine,Dr. Sarah Johnson,Boston
Sarah Thompson,s.thompson@email.com,789-01-2345,44,Migraine,Sumatriptan 50mg,Dr. Amy Wilson,Chicago""".strip()


def create_clients():
    """Initialize Blindfold and OpenAI clients."""
    blindfold = Blindfold(api_key=os.environ["BLINDFOLD_API_KEY"])
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return blindfold, openai_client


def ask_openai_for_code(openai_client: OpenAI, tokenized_csv: str, task: str) -> str:
    """Ask OpenAI to write Python analysis code based on tokenized CSV data.

    OpenAI only sees tokens like <Person_1>, <Email Address_1> — never real PII.
    """
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data analyst. Write Python code using pandas to analyze CSV data. "
                    "The code should read the CSV from '/tmp/data.csv' using pandas. "
                    "Print all results clearly with labels. "
                    "Return ONLY the Python code, no markdown, no explanation, no backticks."
                ),
            },
            {
                "role": "user",
                "content": f"Here is the dataset:\n\n{tokenized_csv}\n\nTask: {task}",
            },
        ],
    )
    return completion.choices[0].message.content


def run_in_sandbox(code: str, original_csv: str) -> str:
    """Execute AI-generated code in an E2B sandbox with the original data.

    The sandbox runs on real data so the analysis results are accurate,
    even though the AI only saw tokenized data when writing the code.
    """
    with Sandbox.create() as sandbox:
        # Write the ORIGINAL (real) CSV — the sandbox is isolated
        sandbox.files.write("/tmp/data.csv", original_csv)

        # Execute the AI-generated analysis code
        execution = sandbox.run_code(code)

        if execution.error:
            return f"Error: {execution.error.name}: {execution.error.value}"

        return "".join(execution.logs.stdout)


# ── Example 1: Basic data analysis ───────────────────────────────────────────


def example_data_analysis():
    """AI writes analysis code from tokenized data, runs it on real data."""
    blindfold, openai_client = create_clients()

    print(f"\n{'='*60}")
    print("Example 1: PII-Safe Data Analysis")
    print(f"{'='*60}")

    # Step 1: Tokenize the CSV
    print("\n[Step 1] Tokenizing CSV with Blindfold...")
    tokenized = blindfold.tokenize(SAMPLE_CSV, policy="strict")

    print(f"  Detected {tokenized.entities_count} PII entities:")
    for entity in tokenized.detected_entities:
        print(f"    - {entity.type}: {entity.text} (confidence: {entity.score:.0%})")

    print(f"\n  Tokenized CSV (this is what OpenAI sees):")
    for line in tokenized.text.split("\n")[:4]:
        print(f"    {line}")
    print(f"    ... ({len(SAMPLE_CSV.split(chr(10)))} rows total)")

    # Step 2: Ask OpenAI to write analysis code
    print("\n[Step 2] Asking OpenAI to write analysis code...")
    print("  (OpenAI only sees tokens — never real names, emails, or SSNs)")

    task = (
        "Analyze this healthcare dataset: "
        "1) Count patients per diagnosis. "
        "2) Calculate average age per diagnosis. "
        "3) Show which doctor treats the most patients. "
        "4) List patients per city."
    )

    code = ask_openai_for_code(openai_client, tokenized.text, task)

    print(f"\n  AI-generated code:")
    for line in code.split("\n"):
        print(f"    {line}")

    # Step 3: Execute in E2B sandbox with the ORIGINAL data
    print("\n[Step 3] Executing code in E2B sandbox with real data...")
    print("  (Sandbox runs on original CSV — results are accurate)")

    results = run_in_sandbox(code, SAMPLE_CSV)

    print(f"\n  Analysis results:")
    for line in results.split("\n"):
        print(f"    {line}")


# ── Example 2: Multi-step analysis ───────────────────────────────────────────


def example_multi_step():
    """Two-step analysis: initial exploration, then follow-up question."""
    blindfold, openai_client = create_clients()

    print(f"\n{'='*60}")
    print("Example 2: Multi-Step Analysis")
    print(f"{'='*60}")

    # Step 1: Tokenize
    print("\n[Step 1] Tokenizing CSV with Blindfold...")
    tokenized = blindfold.tokenize(SAMPLE_CSV, policy="strict")
    print(f"  Detected {tokenized.entities_count} PII entities")

    # Step 2: First analysis — summary statistics
    print("\n[Step 2] Asking OpenAI for summary statistics code...")

    task_1 = (
        "Generate a summary of this dataset: "
        "basic statistics for numerical columns, "
        "value counts for categorical columns (diagnosis, medication, city). "
        "Print everything clearly."
    )

    code_1 = ask_openai_for_code(openai_client, tokenized.text, task_1)

    print("\n  Running first analysis in E2B sandbox...")
    results_1 = run_in_sandbox(code_1, SAMPLE_CSV)

    print(f"\n  Summary statistics:")
    for line in results_1.split("\n"):
        print(f"    {line}")

    # Step 3: Follow-up — deeper analysis based on first results
    print("\n[Step 3] Asking OpenAI for follow-up analysis...")
    print("  (Sharing first results + tokenized CSV, still no real PII to OpenAI)")

    # Tokenize the results too in case they contain real values
    tokenized_results = blindfold.tokenize(results_1, policy="strict")

    task_2 = (
        "Based on these initial results:\n"
        f"{tokenized_results.text}\n\n"
        "Write code to find: "
        "1) Which city has the highest average patient age? "
        "2) What is the most common medication? "
        "3) Create a cross-tabulation of diagnosis vs city."
    )

    code_2 = ask_openai_for_code(openai_client, tokenized.text, task_2)

    print("\n  Running follow-up analysis in E2B sandbox...")
    results_2 = run_in_sandbox(code_2, SAMPLE_CSV)

    print(f"\n  Follow-up results:")
    for line in results_2.split("\n"):
        print(f"    {line}")


if __name__ == "__main__":
    example_data_analysis()
    example_multi_step()
