"""
LangChain RAG Pipeline with Blindfold PII Protection

Uses BlindfoldPIITransformer for selective document ingestion (redact
contact info, keep names) and explicit retrieve-then-tokenize for
query-time protection with FAISS.

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
from langchain_blindfold import BlindfoldPIITransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# API key is optional — omit it to run in local mode (regex-based, offline)
blindfold_client = Blindfold(api_key=os.environ.get("BLINDFOLD_API_KEY"))

# Sample support ticket documents
TICKETS = [
    Document(
        page_content=(
            "Ticket #1001: Customer Sarah Chen (sarah.chen@acme.com, +1-555-234-5678) "
            "reported a billing discrepancy. Account was charged $49.99 twice on "
            "2026-01-15. Resolution: refund issued within 24 hours."
        ),
        metadata={"source": "tickets.csv", "row": 1},
    ),
    Document(
        page_content=(
            "Ticket #1002: James Rivera (james.rivera@example.com) reported "
            "intermittent API timeouts since 2026-02-01. Root cause: regional DNS "
            "misconfiguration. Fixed within 4 hours, SLA credit applied."
        ),
        metadata={"source": "tickets.csv", "row": 2},
    ),
    Document(
        page_content=(
            "Ticket #1003: Maria Garcia (maria.garcia@example.es, +34 612 345 678) "
            "requested a GDPR data export under Art. 15. Export generated and sent "
            "to her email within 30 days."
        ),
        metadata={"source": "tickets.csv", "row": 3},
    ),
    Document(
        page_content=(
            "Ticket #1004: Emma Johnson (emma.johnson@example.co.uk, +44 20 7946 0958) "
            "could not access dashboard after password reset. Issue traced to browser "
            "cache conflict. Resolved by clearing session cookies."
        ),
        metadata={"source": "tickets.csv", "row": 4},
    ),
]


def ingest_documents():
    """Redact contact info from documents and store in FAISS.

    Names are kept so the vector store can match name-based queries.
    """
    print("=== Document Ingestion ===")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(TICKETS)

    # Redact contact info only — keep names searchable
    transformer = BlindfoldPIITransformer(
        pii_method="redact",
        entities=["email address", "phone number"],
    )
    safe_chunks = transformer.transform_documents(chunks)

    for i, (original, safe) in enumerate(zip(chunks, safe_chunks)):
        print(f"  Doc {i + 1}:")
        print(f"    Original: {original.page_content[:80]}...")
        print(f"    Protected: {safe.page_content[:80]}...")

    # Store in FAISS
    vectorstore = FAISS.from_documents(safe_chunks, OpenAIEmbeddings())
    print(f"\nStored {len(safe_chunks)} documents in FAISS\n")
    return vectorstore


def build_rag_chain(vectorstore):
    """Build a RAG chain with retrieve-then-tokenize flow."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the question using only the context below.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")

    def retrieve_and_tokenize(question: str) -> dict:
        """Retrieve with original question, then tokenize context+question together."""
        # Search with original question — names match in vector store
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Single tokenize call — consistent token numbering
        prompt_text = f"Context:\n{context}\n\nQuestion: {question}"
        tokenized = blindfold_client.tokenize(prompt_text)
        return {"tokenized_text": tokenized.text, "mapping": tokenized.mapping}

    def extract_and_generate(data: dict) -> str:
        """Split tokenized prompt back into context/question for the LLM template."""
        text = data["tokenized_text"]
        # The tokenized text has "Context:\n...\n\nQuestion: ..." format
        parts = text.split("\n\nQuestion: ", 1)
        context = parts[0].removeprefix("Context:\n")
        question = parts[1] if len(parts) > 1 else text

        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        ai_text = StrOutputParser().invoke(response)

        # Detokenize to restore real names
        final = blindfold_client.detokenize(ai_text, data["mapping"])
        return final.text

    chain = RunnableLambda(retrieve_and_tokenize) | RunnableLambda(extract_and_generate)
    return chain


def main():
    vectorstore = ingest_documents()
    chain = build_rag_chain(vectorstore)

    print("=== RAG Query ===")
    question = "What was Sarah Chen's issue?"
    print(f"Question: {question}")

    answer = chain.invoke(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
