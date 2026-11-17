"""
Local ChromaDB wrapper for document indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR
from src.processing.chunker import ChunkRecord
from src.retrieval.embeddings import EmbeddingModel, get_embedding_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """Chunk plus retrieval metadata returned from ChromaDB."""

    chunk_id: str
    text: str
    source_file: str
    page_number: int
    section_heading: str | None
    distance: float | None = None


class DocumentVectorStore:
    """Persistent local vector store backed by ChromaDB."""

    def __init__(
        self,
        db_dir: Path = CHROMA_DB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Releases the underlying Chroma client resources."""
        self.client.close()

    def __enter__(self) -> "DocumentVectorStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def add_chunks(self, chunks: list[ChunkRecord]) -> int:
        """Adds or updates chunks in the collection and returns how many were written."""
        if not chunks:
            return 0

        embeddings = self.embedding_model.encode_texts([chunk.text for chunk in chunks])
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "section_heading": chunk.section_heading or "",
                "language": chunk.language,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Indexed %d chunk(s) into %s", len(chunks), self.collection_name)
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int,
        source_file: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieves the most relevant chunks for a query."""
        where: dict[str, Any] | None = {"source_file": source_file} if source_file else None
        query_embedding = self.embedding_model.encode_texts([query_text])
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        if not result.get("ids"):
            return chunks

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=document,
                    source_file=str(metadata.get("source_file", "")),
                    page_number=int(metadata.get("page_number", 0)),
                    section_heading=(metadata.get("section_heading") or None),
                    distance=float(distance) if distance is not None else None,
                )
            )

        return chunks
