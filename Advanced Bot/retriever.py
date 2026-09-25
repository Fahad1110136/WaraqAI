import json
import pickle
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import config
from lang_utils import get_search_mode, tokenize_for_bm25

class HybridRetriever:
    def __init__(self):
        print("Loading indexes ...")
        self.faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)

        with open(config.BM25_PATH, "rb") as f:
            self.bm25_index = pickle.load(f)

        with open(config.CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.records = json.load(f)   # list of {id, source, text}, index-aligned with both indexes

        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.reranker = CrossEncoder(config.RERANKER_MODEL_NAME)

    # ---------- individual retrievers ----------

    def _dense_search(self, query: str, top_k: int) -> List[int]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        _, indices = self.faiss_index.search(query_vec, top_k)
        return [int(i) for i in indices[0] if i != -1]

    def _sparse_search(self, query: str, top_k: int) -> List[int]:
        tokenized_query = tokenize_for_bm25(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in ranked]

    # ---------- fusion ----------

    @staticmethod
    def _reciprocal_rank_fusion(ranked_lists: List[List[int]], k: int = 60) -> List[int]:
        scores: Dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        fused = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
        return fused

    # ---------- reranking ----------

    def _rerank(self, query: str, candidate_ids: List[int], top_k: int) -> List[Dict]:
        pairs = [(query, self.records[doc_id]["text"]) for doc_id in candidate_ids]
        scores = self.reranker.predict(pairs)

        scored = list(zip(candidate_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored[:top_k]:
            record = self.records[doc_id]
            results.append({
                "id": doc_id,
                "source": record["source"],
                "text": record["text"],
                "rerank_score": float(score),
            })
        return results

    # ---------- public entry point ----------

    def search(self, query: str) -> List[Dict]:
        mode, detected_lang = get_search_mode(query)
        
        if detected_lang == "en":
            print(f"\nDetected language: English --> Search mode: Dense")
        elif detected_lang == "ar":
            print(f"\nDetected language: Arabic --> Search mode: {mode}")
            
        dense_ids = self._dense_search(query, config.DENSE_TOP_K)

        if mode == "Hybrid" and detected_lang == "ar":
            sparse_ids = self._sparse_search(query, config.SPARSE_TOP_K)
            fused_ids = self._reciprocal_rank_fusion(
                [dense_ids, sparse_ids], k=config.RRF_K
            )
        else:
            # dense-only mode: nothing to fuse, just use the dense ranking as-is
            fused_ids = dense_ids

        fused_ids = fused_ids[:config.FUSED_TOP_N]

        return self._rerank(query, fused_ids, config.FINAL_TOP_K)