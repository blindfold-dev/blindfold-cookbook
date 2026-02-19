"""
Blindfold + E2B: PII-Safe AI Data Analyst

OpenAI writes analysis code from tokenized data (never sees real PII),
then E2B executes the code on the original data for accurate results.

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

SAMPLE_CSV = """name,email,ssn,age,diagnosis,medication,doctor,city
John Smith,john.smith@email.com,123-45-6789,45,Type 2 Diabetes,Metformin 500mg,Dr. Sarah Johnson,Boston
Maria Garcia,maria.garcia@company.org,987-65-4321,32,Hypertension,Lisinopril 10mg,Dr. Michael Chen,New York
Robert Wilson,rwilson@hospital.net,456-78-9012,58,Type 2 Diabetes,Metformin 1000mg,Dr. Sarah Johnson,Boston
Emily Davis,emily.d@example.com,234-56-7890,27,Asthma,Albuterol Inhaler,Dr. Lisa Park,Chicago
James Brown,jbrown@mail.com,345-67-8901,51,Hypertension,Amlodipine 5mg,Dr. Michael Chen,New York
Anna Martinez,anna.m@health.org,567-89-0123,39,Asthma,Fluticasone Inhaler,Dr. Lisa Park,San Francisco
David Lee,david.lee@work.com,678-90-1234,63,Type 2 Diabetes,Insulin Glargine,Dr. Sarah Johnson,Boston
Sarah Thompson,s.thompson@email.com,789-01-2345,44,Migraine,Sumatriptan 50mg,Dr. Amy Wilson,Chicago""".strip()


def main():
    blindfold = Blindfold(api_key=os.environ["BLINDFOLD_API_KEY"])
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Step 1: Tokenize — replace PII with safe tokens
    tokenized = blindfold.tokenize(SAMPLE_CSV, policy="strict")

    # Step 2: Ask OpenAI to write analysis code (it only sees tokens)
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data analyst. Write Python code using pandas to analyze CSV data. "
                    "Read the CSV from '/tmp/data.csv'. Print all results with labels. "
                    "Return ONLY Python code, no markdown, no backticks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the dataset:\n\n{tokenized.text}\n\n"
                    "Analyze it: count patients per diagnosis, average age per diagnosis, "
                    "which doctor treats the most patients, and patients per city."
                ),
            },
        ],
    )
    code = completion.choices[0].message.content

    # Step 3: Execute in E2B sandbox with the ORIGINAL data
    with Sandbox.create() as sandbox:
        sandbox.files.write("/tmp/data.csv", SAMPLE_CSV)
        execution = sandbox.run_code(code)

    if execution.error:
        print(f"Error: {execution.error.name}: {execution.error.value}")
    else:
        print("".join(execution.logs.stdout))


if __name__ == "__main__":
    main()
