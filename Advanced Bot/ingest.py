import os
import json
import pickle

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import config
from chunking import chunk_text
from lang_utils import tokenize_for_bm25

def _read_txt_or_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text)
    return "\n\n".join(pages_text)

def load_documents(docs_dir: str):
    # Read every .txt / .md / .pdf file in docs_dir. Returns list of (filename, text)
    documents = []
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Documents folder not found: {docs_dir}")

    for fname in sorted(os.listdir(docs_dir)):
        lower = fname.lower()
        path = os.path.join(docs_dir, fname)

        if lower.endswith((".txt", ".md")):
            text = _read_txt_or_md(path)
        elif lower.endswith(".pdf"):
            try:
                text = _read_pdf(path)
            except Exception as e:
                print(f"  Warning: could not extract text from {fname}: {e}")
                continue
            if not text.strip():
                print(f"  Warning: {fname} produced no extractable text "
                      f"(likely a scanned/image-only PDF) — skipping.")
                continue
        else:
            continue

        if text.strip():
            documents.append((fname, text))

    return documents

def build_chunk_records(documents):
    # Chunk every document, return a list of dicts: {id, source, text}
    records = []
    next_id = 0
    for fname, text in documents:
        for chunk in chunk_text(text):
            records.append({"id": next_id, "source": fname, "text": chunk})
            next_id += 1
    return records


def build_faiss_index(records, embedder):
    texts = [r["text"] for r in records]
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # so inner product == cosine similarity
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def build_bm25_index(records):
    # Arabic-aware tokenization (normalizes alef forms, strips diacritics);
    # falls through to plain lowercase+split for non-Arabic text.
    tokenized_corpus = [tokenize_for_bm25(r["text"]) for r in records]
    return BM25Okapi(tokenized_corpus)

def main():
    os.makedirs(config.INDEX_DIR, exist_ok=True)

    print(f"Loading documents from {config.DOCS_DIR} ...")
    documents = load_documents(config.DOCS_DIR)
    if not documents:
        print("No .txt/.md/.pdf documents found. Add some files to the documents/ folder and re-run.")
        return
    print(f"Loaded {len(documents)} document(s).")

    print("Chunking ...")
    records = build_chunk_records(documents)
    print(f"Produced {len(records)} chunk(s).")

    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} (multilingual) ...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("Building FAISS (dense) index ...")
    faiss_index = build_faiss_index(records, embedder)
    faiss.write_index(faiss_index, config.FAISS_INDEX_PATH)

    print("Building BM25 (sparse) index ...")
    bm25_index = build_bm25_index(records)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(bm25_index, f)

    print("Saving chunk metadata ...")
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Done. Index files written to:", config.INDEX_DIR)

if __name__ == "__main__":
    main()