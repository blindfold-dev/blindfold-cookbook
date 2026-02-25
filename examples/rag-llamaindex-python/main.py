"""
LlamaIndex RAG Pipeline with Blindfold PII Protection

Custom BlindfoldNodePostprocessor tokenizes retrieved nodes before
they reach the LLM. Responses are detokenized to restore real data.

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API keys
    python main.py
"""

import os

from blindfold import Blindfold
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

blindfold = Blindfold(api_key=os.environ["BLINDFOLD_API_KEY"])


class BlindfoldNodePostprocessor(BaseNodePostprocessor):
    """Tokenizes PII in retrieved nodes before they reach the LLM."""

    _mapping: dict = {}

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        for node_with_score in nodes:
            result = blindfold.tokenize(node_with_score.node.text)
            node_with_score.node.text = result.text
            self._mapping.update(result.mapping)
        return nodes

    def detokenize_response(self, response_text: str) -> str:
        result = blindfold.detokenize(response_text, self._mapping)
        self._mapping = {}
        return result.text


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
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    Settings.embed_model = OpenAIEmbedding(api_key=os.environ["OPENAI_API_KEY"])

    # Redact PII from documents before indexing
    print("=== Ingestion ===")
    safe_documents = []
    for doc in SUPPORT_TICKETS:
        result = blindfold.redact(doc.text)
        print(f"  Redacted {result.entities_count} entities from ticket")
        safe_documents.append(Document(text=result.text))

    # Build index from redacted documents
    index = VectorStoreIndex.from_documents(safe_documents)
    print(f"Indexed {len(safe_documents)} documents\n")

    # Create postprocessor for query-time tokenization
    postprocessor = BlindfoldNodePostprocessor()
    query_engine = index.as_query_engine(node_postprocessors=[postprocessor])

    # Query with PII
    print("=== Query ===")
    question = "What happened with John Smith's billing issue?"

    # Tokenize the question
    tokenized_q = blindfold.tokenize(question)
    print(f"Original question: {question}")
    print(f"Tokenized question: {tokenized_q.text}\n")

    # Query with tokenized question
    response = query_engine.query(tokenized_q.text)

    # Detokenize with combined mappings
    combined_mapping = {**tokenized_q.mapping, **postprocessor._mapping}
    final = blindfold.detokenize(str(response), combined_mapping)
    postprocessor._mapping = {}

    print(f"Answer: {final.text}")


if __name__ == "__main__":
    main()
