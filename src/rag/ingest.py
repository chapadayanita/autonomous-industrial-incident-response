from pathlib import Path
from typing import List, Dict, Any
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_DIR = (
    PROJECT_ROOT
    / "src"
    / "rag"
    / "knowledge_base"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "src"
    / "rag"
    / "knowledge_index.json"
)


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving readable text."""

    text = re.sub(
        r"\r\n?",
        "\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def _split_into_chunks(
    text: str,
    source: str,
) -> List[Dict[str, Any]]:
    """
    Split Markdown documents into section-based chunks.
    """

    text = _clean_text(text)

    sections = re.split(
        r"\n(?=##\s+)",
        text,
    )

    chunks = []

    for index, section in enumerate(sections):

        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()

        title = lines[0].strip(
            "# "
        )

        chunks.append(
            {
                "chunk_id": (
                    f"{Path(source).stem}_{index}"
                ),
                "source": source,
                "title": title,
                "text": section,
            }
        )

    return chunks


def ingest_knowledge_base() -> Dict[str, Any]:
    """
    Read all Markdown documents and create a searchable
    JSON knowledge index.
    """

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: "
            f"{KNOWLEDGE_BASE_DIR}"
        )

    documents = sorted(
        KNOWLEDGE_BASE_DIR.glob("*.md")
    )

    if not documents:
        raise FileNotFoundError(
            "No Markdown knowledge documents found."
        )

    chunks = []

    for document in documents:

        text = document.read_text(
            encoding="utf-8"
        )

        document_chunks = _split_into_chunks(
            text,
            document.name,
        )

        chunks.extend(
            document_chunks
        )

        print(
            f"Loaded: {document.name} "
            f"({len(document_chunks)} chunks)"
        )

    index = {
        "version": "1.0",
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    INDEX_PATH.write_text(
        json.dumps(
            index,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "status": "ingestion_complete",
        "documents": len(documents),
        "chunks": len(chunks),
        "index_path": str(INDEX_PATH),
    }


if __name__ == "__main__":

    print("=" * 70)
    print("RAG KNOWLEDGE BASE INGESTION")
    print("=" * 70)

    result = ingest_knowledge_base()

    print("\n" + "=" * 70)
    print("INGESTION RESULT")
    print("=" * 70)

    for key, value in result.items():
        print(
            f"{key:<20}: {value}"
        )

    print("\n" + "=" * 70)
    print("RAG INGESTION COMPLETE")
    print("=" * 70)