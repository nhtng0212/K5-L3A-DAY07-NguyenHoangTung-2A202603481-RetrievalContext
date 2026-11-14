"""Step 2: Implement multiple retrieval methods for Day 7 RAG benchmark.

Methods implemented:
1. TfidfRetriever: Lexical search using TF-IDF cosine similarity (from starter).
2. BM25Retriever: Lexical search using Okapi BM25 with length normalization.
3. DensePgRetriever: Vector similarity search using Day 6 PostgreSQL + pgvector.
4. HybridRRFRetriever: Combines Dense and BM25 rankings using Reciprocal Rank Fusion.
"""

import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
from openai import OpenAI

# Import starter baseline utilities (tokenize, TfidfIndex)
DAY7_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = DAY7_ROOT / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from baseline import TfidfIndex, tokenize  # noqa: E402


def match_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check if a chunk matches all key-value pairs in the metadata filter."""
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    return all(str(meta.get(k)) == str(v) for k, v in filters.items())


class TfidfRetriever:
    """Wrapper around starter TfidfIndex that also supports metadata filtering."""

    def __init__(self, chunks: list[dict[str, Any]], use_filter: bool = False):
        self.chunks = chunks
        self.use_filter = use_filter
        self.index = TfidfIndex(chunks)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Search top_k chunks using TF-IDF cosine similarity."""
        active_filter = filters if self.use_filter else None
        if not active_filter:
            return self.index.search(query, top_k=top_k)

        # Search full pool then filter by trusted metadata
        all_hits = self.index.search(query, top_k=len(self.chunks))
        filtered = [c for c in all_hits if match_filter(c, active_filter)]
        return filtered[:top_k]


class BM25Retriever:
    """Lexical retriever using Okapi BM25 algorithm."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        use_filter: bool = True,
    ):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.use_filter = use_filter

        # Pre-tokenize all chunks once at initialization
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = (
            sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1.0
        )

        # Compute Inverse Document Frequency (IDF) for every unique word
        df = Counter(term for freq in self.doc_freqs for term in freq)
        n_docs = len(chunks)
        self.idf = {
            term: math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Score chunks with BM25 and return top_k matches."""
        q_terms = tokenize(query)
        active_filter = filters if self.use_filter else None
        results = []

        for chunk, freq, doc_len in zip(
            self.chunks, self.doc_freqs, self.doc_lens
        ):
            if not match_filter(chunk, active_filter):
                continue

            score = 0.0
            for term in q_terms:
                if term not in freq:
                    continue
                tf = freq[term]
                idf = self.idf.get(term, 0.0)
                # BM25 formula: saturates repeated terms and normalizes document length
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self.avgdl)
                )
                score += idf * (numerator / denominator)

            if score > 0:
                results.append(
                    {**chunk, "score": round(float(score), 6), "strategy": "bm25"}
                )

        results.sort(key=lambda c: (-c["score"], c["id"]))
        return results[:top_k]


class DensePgRetriever:
    """Dense vector retriever connected to Day 6 PostgreSQL + pgvector."""

    ALLOWED_FILTERS = frozenset({"category", "document_type", "language", "volume"})

    def __init__(
        self,
        db_url: str | None = None,
        index_strategy: str = "B_PARAGRAPH",
        embedding_model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        use_filter: bool = True,
    ):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab",
        )
        self.index_strategy = index_strategy
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.use_filter = use_filter
        self.client = OpenAI()
        self.total_tokens = 0
        # Cache query embeddings in memory to avoid paying twice for the same query
        self._query_cache: dict[str, list[float]] = {}

    def embed_query(self, query: str) -> list[float]:
        """Convert query string into a 1536-dim float vector using OpenAI API."""
        if query not in self._query_cache:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[query],
                dimensions=self.dimensions,
                encoding_format="float",
            )
            if response.usage:
                self.total_tokens += response.usage.total_tokens
            self._query_cache[query] = response.data[0].embedding
        return self._query_cache[query]

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Run exact cosine similarity search against pgvector table."""
        q_vec = self.embed_query(query)
        vec_literal = "[" + ",".join(format(float(x), ".17g") for x in q_vec) + "]"

        active_filter = (filters or {}) if self.use_filter else {}
        where_clauses = ["strategy = %s"]
        params: list[Any] = [self.index_strategy]

        for key, value in active_filter.items():
            if key in self.ALLOWED_FILTERS and value is not None:
                where_clauses.append("metadata ->> %s = %s")
                params.extend([key, str(value)])

        where_sql = " AND ".join(where_clauses)
        sql_query = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        full_params = [vec_literal, *params, vec_literal, top_k]

        with psycopg.connect(self.db_url) as conn, conn.cursor() as cur:
            cur.execute(sql_query, full_params)
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "source_id": row[1],
                "source": row[2],
                "page": int(row[3]),
                "text": row[4],
                "metadata": row[5],
                "score": round(float(row[6]), 6),
                "strategy": "dense_pgvector",
            }
            for row in rows
        ]


class HybridRRFRetriever:
    """Combine Dense (pgvector) and Lexical (BM25) rankings using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DensePgRetriever,
        bm25_retriever: BM25Retriever,
        candidate_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        use_filter: bool = True,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.use_filter = use_filter

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Retrieve candidate pools from both retrievers and merge by RRF score."""
        active_filter = filters if self.use_filter else None
        k_pool = max(self.candidate_k, top_k * 3)

        dense_hits = self.dense.search(query, filters=active_filter, top_k=k_pool)
        bm25_hits = self.bm25.search(query, filters=active_filter, top_k=k_pool)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}
        rank_details: dict[str, dict[str, Any]] = {}

        # Accumulate RRF score from Dense rank
        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight * (
                1.0 / (self.rrf_k + rank)
            )
            rank_details.setdefault(cid, {})["dense_rank"] = rank
            rank_details[cid]["dense_score"] = item["score"]

        # Accumulate RRF score from BM25 rank
        for rank, item in enumerate(bm25_hits, start=1):
            cid = item["id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.bm25_weight * (
                1.0 / (self.rrf_k + rank)
            )
            rank_details.setdefault(cid, {})["bm25_rank"] = rank
            rank_details[cid]["bm25_score"] = item["score"]

        # Build final sorted list
        fused = []
        for cid, score in rrf_scores.items():
            base_chunk = dict(chunk_map[cid])
            base_chunk["score"] = round(score, 6)
            base_chunk["strategy"] = "hybrid_rrf"
            base_chunk["fusion_info"] = rank_details[cid]
            fused.append(base_chunk)

        fused.sort(key=lambda c: (-c["score"], c["id"]))
        return fused[:top_k]

