import json
import os

import anthropic
from langchain_core.tools import tool

from app.ingestion.embedder import embed_text
from app.storage import vector_store as vs


@tool
def vector_search(
    query: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
) -> str:
    """Search for document chunks most relevant to a query using semantic similarity.

    Returns JSON array of chunks, each with: id, text, doc_id, chunk_index, page_num, score.
    Use this as your primary tool to find relevant content. Make the query specific for
    better results. Call multiple times with different queries if needed.

    Args:
        query: The search query — make it specific and focused
        top_k: Number of chunks to return (default 5, max 10)
        doc_ids: Optional list of document IDs to restrict search to specific documents
    """
    query_vector = embed_text(query)
    results = vs.search(query_vector, top_k=min(top_k, 10), doc_ids=doc_ids)
    return json.dumps(results)


@tool
def get_chunk_by_id(chunk_id: str) -> str:
    """Retrieve a specific chunk by its Qdrant point ID.

    Returns JSON with: id, text, doc_id, chunk_index, page_num.
    Use this when vector_search returned a chunk you want to read in full,
    or when you need to verify exact wording from a known chunk.

    Args:
        chunk_id: The Qdrant point ID of the chunk (from vector_search results)
    """
    result = vs.get_chunk_by_id(chunk_id)
    return json.dumps(result) if result else json.dumps({"error": f"Chunk {chunk_id} not found"})


@tool
def summarize_doc(doc_id: str) -> str:
    """Generate a summary of an entire document.

    Retrieves the document's chunks and uses Claude to produce a concise summary.
    Use when the user asks for an overview or summary of a specific document.

    Args:
        doc_id: The document ID to summarize
    """
    chunks = vs.get_chunks_for_doc(doc_id, limit=20)
    if not chunks:
        return f"No content found for document {doc_id}."

    combined_text = "\n\n".join(
        f"[chunk {c['chunk_index']}, page {c.get('page_num')}]\n{c['text']}"
        for c in chunks
    )

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Summarize the following document content concisely:\n\n{combined_text}",
        }],
    )
    return response.content[0].text


@tool
def compare_docs(doc_ids: list[str]) -> str:
    """Compare the content of multiple documents side by side.

    Retrieves content from each document and uses Claude to highlight similarities
    and differences. Use when asked to compare, contrast, or find differences between documents.

    Args:
        doc_ids: List of at least 2 document IDs to compare
    """
    if len(doc_ids) < 2:
        return "Need at least 2 document IDs to compare."

    sections: list[str] = []
    for doc_id in doc_ids:
        chunks = vs.get_chunks_for_doc(doc_id, limit=15)
        if not chunks:
            sections.append(f"=== Document {doc_id} ===\nNo content found.")
        else:
            text = "\n\n".join(c["text"] for c in chunks)
            sections.append(f"=== Document {doc_id} ===\n{text}")

    all_content = "\n\n".join(sections)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                "Compare and contrast the following documents, "
                "highlighting key similarities and differences:\n\n"
                + all_content
            ),
        }],
    )
    return response.content[0].text


TOOLS = [vector_search, get_chunk_by_id, summarize_doc, compare_docs]