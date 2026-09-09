from pathlib import Path
from typing import Dict, Any, List
import json
import math
import re
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = (
    PROJECT_ROOT
    / "src"
    / "rag"
    / "knowledge_index.json"
)


class KnowledgeRetriever:
    """
    Lightweight lexical RAG retriever.

    Retrieves relevant knowledge-base chunks using
    token overlap and IDF-weighted scoring.

    No external LLM or API is required for retrieval.
    """

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
    ):

        self.index_path = Path(
            index_path
        )

        self.chunks: List[
            Dict[str, Any]
        ] = []

        self.document_frequency = Counter()

        self._load_index()

    # ================================================================
    # LOAD INDEX
    # ================================================================

    def _load_index(self) -> None:

        if not self.index_path.exists():

            raise FileNotFoundError(
                "Knowledge index not found.\n"
                "Run:\n"
                "python -m src.rag.ingest"
            )

        data = json.loads(
            self.index_path.read_text(
                encoding="utf-8"
            )
        )

        self.chunks = data.get(
            "chunks",
            [],
        )

        if not self.chunks:

            raise ValueError(
                "Knowledge index contains no chunks."
            )

        for chunk in self.chunks:

            tokens = set(
                self._tokenize(
                    chunk["text"]
                )
            )

            for token in tokens:
                self.document_frequency[
                    token
                ] += 1

    # ================================================================
    # TOKENIZATION
    # ================================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> List[str]:

        return re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

    # ================================================================
    # IDF
    # ================================================================

    def _idf(
        self,
        token: str,
    ) -> float:

        total_documents = len(
            self.chunks
        )

        frequency = self.document_frequency.get(
            token,
            0,
        )

        return math.log(
            (total_documents + 1)
            / (frequency + 1)
        ) + 1

    # ================================================================
    # SCORE
    # ================================================================

    def _score(
        self,
        query_tokens: List[str],
        document_text: str,
    ) -> float:

        document_tokens = self._tokenize(
            document_text
        )

        if not document_tokens:
            return 0.0

        document_counts = Counter(
            document_tokens
        )

        score = 0.0

        for token in query_tokens:

            if token not in document_counts:
                continue

            tf = (
                document_counts[token]
                / len(document_tokens)
            )

            score += (
                tf
                * self._idf(token)
            )

        # Small bonus when exact multi-word
        # query phrase appears.
        return score

    # ================================================================
    # RETRIEVE
    # ================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():

            raise ValueError(
                "query cannot be empty"
            )

        if top_k < 1:

            raise ValueError(
                "top_k must be at least 1"
            )

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        results = []

        for chunk in self.chunks:

            score = self._score(
                query_tokens,
                chunk["text"],
            )

            if score <= 0:
                continue

            result = chunk.copy()

            result["score"] = round(
                score,
                6,
            )

            results.append(
                result
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]

    # ================================================================
    # CONTEXT BUILDER
    # ================================================================

    def build_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:

        results = self.retrieve(
            query=query,
            top_k=top_k,
        )

        context_parts = []

        for result in results:

            context_parts.append(
                f"Source: {result['source']}\n"
                f"Section: {result['title']}\n"
                f"{result['text']}"
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        return {
            "query": query,
            "results": results,
            "result_count": len(results),
            "context": context,
        }


# ======================================================================
# SINGLETON
# ======================================================================

_retriever = None


def get_knowledge_retriever() -> KnowledgeRetriever:

    global _retriever

    if _retriever is None:

        _retriever = KnowledgeRetriever()

    return _retriever


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def retrieve_knowledge(
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:

    retriever = get_knowledge_retriever()

    return retriever.build_context(
        query=query,
        top_k=top_k,
    )


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RAG RETRIEVER TEST")
    print("=" * 70)

    retriever = get_knowledge_retriever()

    queries = [
        "abnormal pump vibration",
        "bearing mechanical problem",
        "pressure leakage",
        "incident escalation",
    ]

    for query in queries:

        print("\n" + "-" * 70)
        print(
            f"QUERY: {query}"
        )
        print("-" * 70)

        results = retriever.retrieve(
            query=query,
            top_k=3,
        )

        if not results:

            print(
                "No relevant knowledge found."
            )

            continue

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\nResult {index}"
            )

            print(
                f"Source : "
                f"{result['source']}"
            )

            print(
                f"Section: "
                f"{result['title']}"
            )

            print(
                f"Score  : "
                f"{result['score']}"
            )

            print(
                f"Text   : "
                f"{result['text'][:300]}..."
            )

    print("\n" + "=" * 70)
    print("RAG RETRIEVER TEST COMPLETE")
    print("=" * 70)