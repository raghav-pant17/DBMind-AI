"""Load and split PDF documents for retrieval."""

import hashlib
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 300
CHUNK_OVERLAP = 100


def load_pdf_documents(directory: Path) -> list[Document]:
    """Load and split every PDF in the configured directory."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Business context directory does not exist: {directory}")

    pages = PyPDFDirectoryLoader(str(directory), mode="page").load()
    for page in pages:
        page.metadata["source"] = Path(str(page.metadata["source"])).name
    chunks = split_pdf_documents(pages)

    if not chunks:
        raise ValueError("No text chunks were loaded from the configured PDFs.")
    return chunks


def split_pdf_documents(documents: list[Document]) -> list[Document]:
    """Split PDF pages and attach stable source, page, and chunk metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    page_chunk_counts: defaultdict[tuple[str, int], int] = defaultdict(int)

    for chunk in chunks:
        source = Path(str(chunk.metadata.get("source", "document.pdf"))).name
        page = int(chunk.metadata.get("page", 0))
        key = (source, page)
        chunk_number = page_chunk_counts[key]
        page_chunk_counts[key] += 1
        chunk_id = hashlib.sha256(
            f"{source}:{page}:{chunk_number}:{chunk.page_content}".encode()
        ).hexdigest()
        chunk.id = chunk_id
        chunk.metadata = {
            "source": source,
            "page": page,
            "chunk": chunk_number,
            "chunk_id": chunk_id,
        }

    return chunks
