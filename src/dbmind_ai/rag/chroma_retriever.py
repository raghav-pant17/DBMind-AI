"""Simple ChromaDB retrieval for PDF chunks."""

from pathlib import Path

import chromadb
from langchain_core.documents import Document

from dbmind_ai.domain import BusinessRule


class ChromaRetriever:
    """Index and retrieve PDF chunks from one local ChromaDB collection."""

    def __init__(self, chroma_path: Path, collection_name: str = "dbmind_documents") -> None:
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def rebuild(self, chunks: list[Document]) -> int:
        """Replace the collection with freshly loaded PDF chunks."""

        if not chunks:
            raise ValueError("Cannot build a Chroma index without PDF chunks.")

        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.collection.upsert(
            documents=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[chunk.id or f"ID{index}" for index, chunk in enumerate(chunks)],
        )
        return len(chunks)

    def retrieve(self, question: str, top_k: int = 4) -> list[BusinessRule]:
        """Return the PDF chunks most relevant to the question."""

        if self.collection.count() == 0:
            raise RuntimeError(
                "The PDF RAG index is empty. Run `dbmind-ai index-rag` before `ask`."
            )

        results = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, self.collection.count()),
        )
        return [
            BusinessRule(
                id=f"doc-{id_}",
                name=(
                    f"{metadata['source']} page {int(metadata['page']) + 1}, "
                    f"chunk {int(metadata['chunk']) + 1}"
                ),
                description=(
                    f"Source: {metadata['source']}, page {int(metadata['page']) + 1}\n\n{text}"
                ),
            )
            for id_, text, metadata in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                strict=True,
            )
            if text is not None and metadata is not None
        ]

    def count(self) -> int:
        return self.collection.count()

    def all(self) -> list[BusinessRule]:
        return []
