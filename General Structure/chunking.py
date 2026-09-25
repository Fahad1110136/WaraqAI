from typing import List
import config

def _split_into_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    current = ""

    for para in paragraphs:
        # paragraph itself too big -> hard-split it
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end].strip())
                start = end - overlap if end - overlap > start else end
            continue

        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current.strip())
            # start new chunk, carry over the tail of the previous one for overlap
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n{para}".strip()

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]