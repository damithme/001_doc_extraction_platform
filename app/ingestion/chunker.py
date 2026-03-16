from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split extracted pages into chunks.
    Each chunk gets: {text, page_num, chunk_index}
    """
    chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for text in page_chunks:
            chunks.append({
                "text": text,
                "page_num": page.get("page_num"),
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return chunks