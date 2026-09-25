import os
from dotenv import load_dotenv

load_dotenv() 

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")       
INDEX_DIR = os.path.join(BASE_DIR, "index_store")    

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

# ---- Chunking ----
CHUNK_SIZE = 200         
CHUNK_OVERLAP = 40     

# ---- Embedding model ----
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# ---- Reranker model ----
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ---- Language aware retrieval routing ----
DENSE_ONLY_LANGUAGES = {"English"}   
HYBRID_LANGUAGES = {"Arabic"}           
DEFAULT_SEARCH_MODE = "Hybrid"     

# ---- Retrieval hyperparameters ----
DENSE_TOP_K = 20         
SPARSE_TOP_K = 20        
RRF_K = 60               
FUSED_TOP_N = 15          
FINAL_TOP_K = 5         

# ---- Relevance threshold ----
RELEVANCE_SCORE_THRESHOLD = 0.3

# ---- Answer language ----
ANSWER_LANGUAGE = "Arabic"

# ---- Groq LLM ----
GROQ_MODEL_NAME = "openai/gpt-oss-120b" # "llama-3.3-70b-versatile"
GROQ_API_KEY = "GROQ_API_KEY"  
MAX_CONTEXT_CHARS = 12000  