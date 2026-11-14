"""Step 2: Run retrieval and context packing experiments on dev and test splits."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DAY7_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = DAY7_ROOT / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from baseline import hit, pack_context, validate  # noqa: E402
from retrievers import (  # noqa: E402
    BM25Retriever,
    DensePgRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
)


def load_env_files() -> None:
    """Load environment variables from workspace .env files."""
    for env_path in (
        DAY7_ROOT / ".env",
        DAY7_ROOT.parent / ".env",
        DAY7_ROOT.parent.parent / ".env",
    ):
        if env_path.is_file():
            load_dotenv(env_path)


def evaluate_config(
    name: str,
    description: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    split: str,
    budget: int,
    top_k: int = 3,
    dense_ref: DensePgRetriever | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one retrieval configuration on the selected split and record metrics."""
    tokens_before = dense_ref.total_tokens if dense_ref else 0
    traces = []

    for q in questions:
        start = time.perf_counter()
        q_filter = q.get("filter")
        results = retriever.search(q["question"], filters=q_filter, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Evaluate top-3 for standardized Hit@1 and Evidence Hit@3 comparison
        top3 = results[:3]
        marked_top3 = [{**c, "hit": hit(c, q["gold"])} for c in top3]
        marked_all = [{**c, "hit": hit(c, q["gold"])} for c in results]

        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "filter": q_filter,
                "config_name": name,
                "strategy": results[0]["strategy"] if results else name,
                "top_k": top_k,
                "budget_chars": budget,
                "gold": q["gold"],
                "results": marked_all,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 3),
                "hit1": bool(marked_top3 and marked_top3[0]["hit"]),
                "hit3": any(c["hit"] for c in marked_top3),
                "context_hit": any(c["hit"] and c["id"] in ids for c in marked_all),
            }
        )

    tokens_used = (dense_ref.total_tokens - tokens_before) if dense_ref else 0
    # OpenAI text-embedding-3-small costs $0.02 per 1M tokens
    cost_usd = round(tokens_used / 1_000_000 * 0.02, 8)
    n = len(traces)

    summary = {
        "config_name": name,
        "description": description,
        "split": split,
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4),
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4),
        "context_hit": sum(t["context_hit"] for t in traces),
        "context_hit_rate": round(sum(t["context_hit"] for t in traces) / n, 4),
        "avg_context_chars": round(
            statistics.mean(t["context_chars"] for t in traces), 1
        ),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 3),
        "total_ms": round(sum(t["elapsed_ms"] for t in traces), 2),
        "embedding_tokens": tokens_used,
        "service_cost_usd": cost_usd,
        "missed_hit3_ids": [t["query_id"] for t in traces if not t["hit3"]],
        "missed_context_ids": [t["query_id"] for t in traces if not t["context_hit"]],
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Day 7 retrieval benchmark across multiple methods and configs."
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()

    load_env_files()
    data_dir = DAY7_ROOT / "starter" / "data"
    out_base = DAY7_ROOT / "starter" / "results" / f"retrieval_{args.split}"
    out_base.mkdir(parents=True, exist_ok=True)

    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    all_questions = json.loads(
        (data_dir / "questions.json").read_text(encoding="utf-8")
    )
    questions = [q for q in all_questions if q["split"] == args.split]
    validate(chunks, questions)

    # Initialize retrievers
    tfidf_no_filter = TfidfRetriever(chunks, use_filter=False)
    bm25_no_filter = BM25Retriever(chunks, use_filter=False)
    bm25_with_filter = BM25Retriever(chunks, use_filter=True)
    dense_no_filter = DensePgRetriever(use_filter=False)
    dense_with_filter = DensePgRetriever(use_filter=True)
    # Share embedding cache between dense instances so we only embed each query once
    dense_with_filter._query_cache = dense_no_filter._query_cache

    hybrid_no_filter = HybridRRFRetriever(
        dense_no_filter, bm25_no_filter, candidate_k=10, use_filter=False
    )
    hybrid_with_filter = HybridRRFRetriever(
        dense_with_filter, bm25_with_filter, candidate_k=15, use_filter=True
    )

    if args.split == "dev":
        configs = [
            (
                "1_baseline_tfidf_b800",
                "Method 1 (Lexical Baseline): TF-IDF, top_k=3, no filter, budget=800 chars",
                tfidf_no_filter,
                800,
                3,
                None,
            ),
            (
                "2_bm25_b800",
                "Method 2 (Lexical BM25): Okapi BM25 (k1=1.5, b=0.75), top_k=3, no filter, budget=800 chars",
                bm25_no_filter,
                800,
                3,
                None,
            ),
            (
                "3_dense_pgvector_b800",
                "Method 3 (Dense Day 6): pgvector cosine (B_PARAGRAPH), top_k=3, no filter, budget=800 chars",
                dense_no_filter,
                800,
                3,
                dense_no_filter,
            ),
            (
                "4_improve1_dense_b4000_filter",
                "Improvement 1 (Budget + Filter): Dense pgvector, top_k=3, trusted category filter, budget=4000 chars",
                dense_with_filter,
                4000,
                3,
                dense_with_filter,
            ),
            (
                "5_improve2_hybrid_rrf_b4000_filter",
                "Improvement 2 (Method 4 - Hybrid RRF + Budget + Filter): Dense + BM25 RRF (pool=15), top_k=3, trusted filter, budget=4000 chars",
                hybrid_with_filter,
                4000,
                3,
                dense_with_filter,
            ),
        ]
    else:
        # On test split, run only locked Baseline vs Selected Configuration
        configs = [
            (
                "1_baseline_tfidf_b800",
                "Baseline: TF-IDF, top_k=3, no filter, budget=800 chars",
                tfidf_no_filter,
                800,
                3,
                None,
            ),
            (
                "5_selected_hybrid_rrf_b4000_filter",
                "Selected Config: Hybrid RRF (Dense + BM25, pool=15), top_k=3, trusted filter, budget=4000 chars",
                hybrid_with_filter,
                4000,
                3,
                dense_with_filter,
            ),
        ]

    all_summaries = []
    for name, desc, retr, budget, top_k, d_ref in configs:
        summary, traces = evaluate_config(
            name, desc, retr, questions, args.split, budget, top_k, d_ref
        )
        all_summaries.append(summary)
        cfg_dir = out_base / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (cfg_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{name}] Hit@1={summary['hit1']}/{summary['n']} | "
            f"Hit@3={summary['hit3']}/{summary['n']} | "
            f"ContextHit={summary['context_hit']}/{summary['n']} | "
            f"Median={summary['median_ms']}ms"
        )

    (out_base / "comparison_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved all {args.split} retrieval comparisons to {out_base}")


if __name__ == "__main__":
    main()
