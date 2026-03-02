"""
LangChain + Blindfold: PII-safe chains with RunnableLambda.

Wraps any LangChain chain with Blindfold tokenization/detokenization
so PII never reaches the language model.

Works in two modes:
  - Local mode (no API key): PII detected via built-in regex patterns (emails, cards, SSNs, etc.)
  - Cloud mode (with API key): NLP-powered detection adds names, addresses, organizations
"""

import os

from blindfold import Blindfold
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

# API key is optional — omit it to run in local mode (regex-based, offline)
blindfold = Blindfold(api_key=os.environ.get("BLINDFOLD_API_KEY"))


def blindfold_protect(policy: str = "basic"):
    """Create a tokenize/detokenize wrapper for any LangChain chain."""
    mapping_store: dict[str, str] = {}

    def tokenize(text: str) -> str:
        result = blindfold.tokenize(text, policy=policy)
        mapping_store.update(result.mapping)
        return result.text

    def detokenize(text: str) -> str:
        result = blindfold.detokenize(text, mapping_store)
        mapping_store.clear()
        return result.text

    return RunnableLambda(tokenize), RunnableLambda(detokenize)


def main():
    tokenize, detokenize = blindfold_protect(policy="basic")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
        ]
    )

    # Chain: tokenize → prompt → LLM → detokenize
    chain = tokenize | prompt | llm | (lambda msg: msg.content) | detokenize

    message = "Write a follow-up email to Jane Doe (jane@example.com) about her order #12345."
    print(f"User: {message}\n")

    response = chain.invoke(message)
    print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
