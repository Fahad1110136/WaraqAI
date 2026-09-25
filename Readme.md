# 📖 Arabic Chat — Hybrid RAG Chatbot (دردشة عربية)

A multi-user, Arabic first **Retrieval-Augmented Generation (RAG)** chatbot with a Streamlit UI. It combines **dense vector search (FAISS)** and **sparse keyword search (BM25)**, fuses them with **Reciprocal Rank Fusion**, re-ranks with a **cross-encoder**, and generates grounded Arabic answers with an LLM — falling back to the model's general knowledge when no relevant document is found. It also auto translates every Arabic answer to English for convenience.

The `LLM Switching` folder is a variant of the same app that adds the ability to switch the answer generation backend between **Groq (Cloud API)** and **Ollama (Local Models)** at runtime.

---

## ✨ Key Features

- **Hybrid retrieval** — Dense (FAISS, `BAAI/bge-m3` multilingual embeddings) + Sparse (BM25 with Arabic-aware tokenization) fused via Reciprocal Rank Fusion (RRF).
- **Cross-encoder re-ranking** — `BAAI/bge-reranker-v2-m3` re-scores fused candidates for final relevance ordering.
- **Language-aware search routing** — English queries use dense-only search; Arabic (and other) queries use the full hybrid pipeline.
- **Grounded + fallback answering** — If retrieved chunks pass a relevance-score threshold, the LLM answers from context; otherwise it answers from general knowledge (and the UI shows an amber "no context" badge instead of green).
- **Always Arabic answers** — The LLM is instructed to always answer in natural Modern Standard Arabic, regardless of the question's language.
- **Automatic Arabic → English translation** — Every answer is translated for the English-speaking user, shown in a toggleable panel.
- **Multi user auth** — Username/password accounts stored in `users.json` (SHA-256 salted hashes), with signed session tokens so a login persists via URL query params.
- **Per user chat history** — Each user's conversations are saved to `chat_history/<username>.json`; chats can be created, renamed, deleted, and reloaded from the sidebar.
- **Document management UI** — Upload `.pdf` / `.txt` / `.md` files (size- and type-validated), preview them in-app, and (re)build or delete the index — all from the sidebar.
- **RTL/LTR-aware chat UI** — Custom CSS renders Arabic text right-to-left and English text left-to-right, ChatGPT-style bubbles, dark theme.
- **CLI mode** — `main.py` offers `ingest`, `query`, and `chat` subcommands for terminal use without Streamlit.
- **Quran splitter utility** (`splitter.py`) — a standalone script that splits a Tanzil-format Quran text file into 30 Juz (`Para1.txt` … `Para30.txt`) documents, ready to drop into `documents/` and ingest.
- **(LLM Switching variant only)** — Toggle between **Groq** (cloud, `llama-3.3-70b-versatile`) and **Ollama** (local, `qwen2.5:7b`) as the answer-generation backend, with per-message response-time display.

---

## 🏗️ Architecture / How It Works

```
                          ┌─────────────────────┐
   documents/ (.pdf/.txt/.md)       chunking.py │
        │                           (paragraph-aware,
        ▼                           char-based chunks)
   ingest.py  ───────────────────────┘
        │
        ├─► FAISS dense index (faiss.index)      — bge-m3 embeddings, cosine via inner product
        ├─► BM25 sparse index (bm25.pkl)         — Arabic-normalized tokens
        └─► chunk metadata (chunks.json)         — {id, source, text}, index-aligned
   User query
        │
        ▼
   lang_utils.detect_language()  ──►  route: Dense-only (English) or Hybrid (Arabic/other)
        │
        ▼
   retriever.HybridRetriever.search()
        ├─ dense_search()   (FAISS, top-K)
        ├─ sparse_search()  (BM25, top-K)         [Hybrid mode only]
        ├─ reciprocal_rank_fusion()                [Hybrid mode only]
        └─ cross-encoder rerank() → top-N chunks with rerank_score
        │
        ▼
   relevance check: top chunk's rerank_score ≥ RELEVANCE_SCORE_THRESHOLD ?
        │                                   │
      Yes (context found)                No (no relevant context)
        │                                   │
        ▼                                   ▼
   llm.answer_query()                llm.answer_query()
   GROUNDED_SYSTEM_PROMPT            FALLBACK_SYSTEM_PROMPT
   (answer from context, in Arabic)  (answer from general knowledge, in Arabic)
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
              llm.translate_to_english()
                        │
                        ▼
              Streamlit UI: Arabic answer + English translation + source snippets + relevance badge
```

**Chunking strategy** (`chunking.py`): text is split into paragraphs (blank-line separated), then greedily packed into chunks up to `CHUNK_SIZE` characters, carrying a `CHUNK_OVERLAP`-character tail forward for context continuity. Oversized paragraphs are hard-split.

**Fusion** (`retriever.py`): standard RRF — `score(doc) = Σ 1 / (RRF_K + rank + 1)` across the dense and sparse ranked lists — combines the two ranking signals before re-ranking trims it down to the final top-K.

---

## 📁 Project Structure

```
Arabic Chatbot/

├── .vscode
├── Advanced Bot/        # Adds Groq ⇄ Ollama provider switching
├── General Structure/
│   ├── .streamlit/
│   ├── chat_history/            
│   │      └── <username>.json
│   ├── documents/                
│   ├── index_store/                
│   ├── .session_secret         
│   ├── chunking.py                
│   ├── config.py                 
│   ├── ingest.py                  
│   ├── lang_utils.py              
│   ├── llm.py                     
│   ├── main.py                   
│   ├── retriever.py               
│   ├── splitter.py                 
│   ├── streamlit_app.py            
│   ├── users.json                
├── .env
├── .gitignore
├── Readme.md
└── requirements.txt
```

> Note: In the uploaded structure, `Advanced Bot` sits inside the main project root as a self-contained sibling copy of the app (its own `config.py`, `llm.py`, `streamlit_app.py`, `users.json`, `chat_history/`), rather than being imported by the main app. Run either app independently depending on whether you want Groq-only or Groq+Ollama switching.

---

## ⚙️ Configuration Reference (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `DOCS_DIR` | `documents/` | Folder scanned by `ingest.py` for `.pdf`/`.txt`/`.md` source files |
| `INDEX_DIR` | `index_store/` | Where FAISS index, BM25 pickle, and chunk metadata are saved |
| `CHUNK_SIZE` | `200` | Max characters per chunk |
| `CHUNK_OVERLAP` | `40` | Overlap (chars) carried between consecutive chunks |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Multilingual dense embedding model |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder re-ranking model |
| `DENSE_ONLY_LANGUAGES` | `{"English"}` | Languages routed to dense-only search |
| `HYBRID_LANGUAGES` | `{"Arabic"}` | Languages routed to full hybrid search |
| `DEFAULT_SEARCH_MODE` | `"Hybrid"` | Fallback mode for languages not in either set |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | `20` / `20` | Candidates pulled from each retriever before fusion |
| `RRF_K` | `60` | RRF constant (higher = flatter rank weighting) |
| `FUSED_TOP_N` | `15` | Candidates kept after fusion, before re-ranking |
| `FINAL_TOP_K` | `5` | Final chunks passed to the LLM as context |
| `RELEVANCE_SCORE_THRESHOLD` | `0.3` | Minimum top rerank score to treat context as "relevant" |
| `ANSWER_LANGUAGE` | `"Arabic"` | Target answer language |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` | Groq-hosted LLM used for generation & translation |
| `GROQ_API_KEY` | env var name `GROQ_API_KEY` | Read from `.env` |
| `MAX_CONTEXT_CHARS` | `12000` | Max characters of context stuffed into the LLM prompt |

**`LLM Switching` variant adds:**

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | env var `LLM_PROVIDER`, default `"groq"` | `"groq"` or `"ollama"` |
| `OLLAMA_MODEL_NAME` | `qwen2.5:7b` | Local Ollama model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_TOP_K` / `TOP_P` / `MIN_P` | `40` / `0.9` / `0.0` | Sampling parameters |
| `OLLAMA_REPEAT_PENALTY` / `REPEAT_LAST_N` | `1.1` / `64` | Repetition control |
| `OLLAMA_NUM_CTX` / `NUM_PREDICT` | `4096` / `1024` | Context window / max generated tokens |
| `OLLAMA_SEED` | `None` | Optional deterministic seed |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ recommended
- A [Groq API key](https://console.groq.com/) (free tier available)
- (Optional, for `LLM Switching`) [Ollama](https://ollama.com/) installed locally with the `qwen2.5:7b` model pulled: `ollama pull qwen2.5:7b`

### 2. Install dependencies
```bash
cd "Arabic Chatbot"          # or "LLM Switching" for that variant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root (this is git-ignored, never commit it):
```env
GROQ_API_KEY=your_groq_api_key_here

# Only used by the LLM Switching variant, optional, defaults to "groq"
LLM_PROVIDER=groq
```

### 4. Add source documents
Place `.pdf`, `.txt`, or `.md` files into a `documents/` folder in the project root (create it if it doesn't exist), **or** upload them later through the Streamlit sidebar.

> To ingest the Quran: run `python splitter.py path/to/quran-simple.txt` (Tanzil format, `sura|ayah|text` per line) — this writes `Para1.txt` … `Para30.txt` directly into `documents/`.

### 5. Build the index
```bash
python main.py ingest
```
This chunks every document, embeds it with `bge-m3`, builds the FAISS + BM25 indexes, and saves everything under `index_store/`. You can also do this from the Streamlit sidebar's **"(Re)build index"** button.

### 6. Run the app

**Web UI:**
```bash
streamlit run streamlit_app.py
```
Open the local URL Streamlit prints (typically `http://localhost:8501`), register an account, and start chatting.

**CLI:**
```bash
python main.py chat                 # interactive terminal chat
python main.py query --q "سؤالك هنا"  # one-off question
python main.py ingest                # (re)build the index only
```

---

## 🖥️ Using the Web App

- **Register / Login** — Accounts are created from the sidebar-free auth screen; passwords are salted and SHA-256 hashed into `users.json`. A signed session token is stored in the URL so a page refresh keeps you logged in.
- **Chat** — Type in English or Arabic in the chat box. The bot always answers in Arabic and (optionally) shows an English translation underneath.
- **Relevance badge** — 🟢 green badge = answer grounded in your documents; 🟠 amber badge = no relevant document found, answered from general knowledge.
- **Sources panel** — Expand "📎 Sources" under any answer to see which document chunks were retrieved and their rerank scores.
- **Chat History** — All conversations are saved per-user; create new chats, rename, or delete old ones from the sidebar.
- **Documents panel** — Upload new files (PDF/TXT/MD, 15 MB max each), preview them inline, and rebuild or delete the index without leaving the browser.
- **Display Settings** — Toggle source display and English translation on/off; a read-only panel shows the current retrieval/model configuration.
- **(LLM Switching only) Model backend** — Choose "Groq (cloud)" or "Ollama (local)" from the sidebar radio button to change which LLM generates answers and translations; each assistant message also shows its response time.

---

## 🔐 Security Notes

- Passwords are stored as **salted SHA-256 hashes**, never in plaintext.
- Session tokens are HMAC-style signed with a locally generated `.session_secret` (auto-created on first run, git-ignored).
- Uploaded filenames are sanitized (`secure_filename`) and restricted to `.pdf`/`.txt`/`.md`, with a per-file size cap.
- Basic client-side rate limiting (`MIN_SECONDS_BETWEEN_MESSAGES`) throttles rapid-fire message submission.
- `users.json`, `chat_history/`, `.env`, `.session_secret`, `documents/`, and `index_store/` are all git-ignored — don't commit them; they contain credentials or user data.

---

## 🧩 Module Reference

| File | Responsibility |
|---|---|
| `config.py` | Loads `.env`, defines all paths, model names, and retrieval/LLM hyperparameters |
| `chunking.py` | `chunk_text()` — paragraph-aware, overlap-preserving text chunker |
| `lang_utils.py` | `detect_language()`, `get_search_mode()`, Arabic normalization (`normalize_arabic`) and BM25 tokenization |
| `ingest.py` | Loads documents (PDF/TXT/MD), chunks them, builds & saves FAISS + BM25 indexes and chunk metadata |
| `retriever.py` | `HybridRetriever` — dense search, sparse search, RRF fusion, cross-encoder reranking |
| `llm.py` | Groq API calls for grounded/fallback answer generation and Arabic→English translation *(the `LLM Switching` version also dispatches to Ollama via `_generate()`)* |
| `main.py` | CLI: `ingest`, `query --q "..."`, `chat` subcommands; also has an Arabic RTL console formatter |
| `splitter.py` | Standalone script to split a Tanzil-format Quran text file into 30 Juz documents |
| `streamlit_app.py` | Full web app: auth, chat UI, chat history, document upload/preview, index management, settings |

---

## 📦 Dependencies

Core: `faiss-cpu`, `sentence-transformers`, `rank_bm25`, `groq`, `numpy`, `pypdf`, `pymupdf`, `python-dotenv`, `langdetect`, `pyarabic`, `pillow`, `arabic-reshaper`, `python-bidi`, `streamlit>=1.38`.

`LLM Switching` variant adds: `ollama`.

---

## 🛣️ Possible Future Improvements

- Consolidate the `LLM Switching` variant back into the main app behind a single config flag rather than a duplicated codebase.
- Add automated tests for chunking, retrieval fusion, and language routing.
- Support additional document types (`.docx`, `.epub`).
- Add citation-level highlighting of exactly which sentence(s) an answer drew from.