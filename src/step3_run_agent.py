"""Step 3: Real LLM Controller and Agentic RAG runner for Day 7."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

DAY7_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = DAY7_ROOT / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from agentic import run_agent  # noqa: E402
from retrievers import (  # noqa: E402
    BM25Retriever,
    DensePgRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
)

SYSTEM_PROMPT = """Bạn là một trợ lý nghiên cứu pháp luật chính xác (Agentic RAG).
Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong phần CONTEXT được cung cấp từ công cụ search. Tuyệt đối không dùng kiến thức bên ngoài hay tự suy đoán.
2. Nếu chưa có CONTEXT hoặc CONTEXT chưa đủ thông tin để trả lời câu hỏi:
   - Hãy gọi hành động "search" với một câu truy vấn tiếng Việt ngắn gọn, tập trung vào các từ khóa pháp lý cốt lõi của câu hỏi.
   - Không lặp lại y hệt các câu truy vấn đã dùng trước đó.
   - Nếu bạn đã tìm kiếm từ 2 lượt trở lên với các từ khóa khác nhau mà CONTEXT vẫn không có thông tin trực tiếp giải đáp câu hỏi (ví dụ câu hỏi hỏi về quy định của BLDS 2015, lãi suất vay, tuổi kết hôn, thừa kế, phạt vi phạm hợp đồng... không có trong giáo trình Tập 1), hãy dừng tìm kiếm và trả về hành động "answer" với answer="" và citations=[] để hệ thống ghi nhận từ chối (abstain).
3. Nếu CONTEXT đã có đủ bằng chứng trả lời:
   - Trả về hành động "answer".
   - Câu trả lời phải bám sát văn bản, nêu đầy đủ các điều kiện/ý chính và BẮT BUỘC đính kèm mã trích dẫn dạng [chunk_id] ngay trong câu trả lời (ví dụ: "... [paragraph:giao-trinh-luat-dan-su-viet-nam-tap-1:p0022:c0002].").
   - Mảng "citations" chỉ được chứa các chunk_id có thật trong danh sách chunk_id hợp lệ của CONTEXT và thực sự hỗ trợ câu trả lời.

Bạn PHẢI trả về duy nhất một JSON object hợp lệ theo một trong hai mẫu sau:
Mẫu 1 (Cần tìm kiếm):
{"action": "search", "query": "từ khóa tìm kiếm"}

Mẫu 2 (Trả lời khi đủ bằng chứng, hoặc để trống answer="" và citations=[] khi từ chối):
{"action": "answer", "answer": "Nội dung trả lời kèm [chunk_id]", "citations": ["chunk_id"]}
"""


def load_env_files() -> None:
    """Load environment variables from workspace .env files."""
    for env_path in (
        DAY7_ROOT / ".env",
        DAY7_ROOT.parent / ".env",
        DAY7_ROOT.parent.parent / ".env",
    ):
        if env_path.is_file():
            load_dotenv(env_path)


class OpenAIController:
    """Real LLM controller using OpenAI Chat Completions with JSON mode."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call OpenAI model to choose between 'search' and 'answer'."""
        if not history:
            # First search always uses the original user question to preserve full semantic & lexical signal
            return {"action": "search", "query": question}

        latest = history[-1]
        past_queries = latest.get("queries", [])
        context = latest.get("context", "")
        context_ids = latest.get("context_ids", [])
        user_prompt = (
            f"Câu hỏi: {question}\n\n"
            f"Các truy vấn đã thử ({len(past_queries)} lượt): {past_queries}\n"
            f"Danh sách chunk_id hợp lệ hiện có trong CONTEXT: {context_ids}\n\n"
            f"CONTEXT HIỆN TẠI:\n{context if context else '(Trống - không có đoạn văn nào vừa với context budget)'}\n\n"
            "Hãy quyết định bước tiếp theo: gọi 'search' với truy vấn mới nếu thiếu thông tin, "
            "hoặc 'answer' kèm [chunk_id] nếu đã đủ bằng chứng, hoặc 'answer' rỗng nếu tài liệu không có thông tin."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        self.llm_calls += 1
        if response.usage:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens

        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        if parsed.get("action") == "search":
            query = str(parsed.get("query", "")).strip() or question
            return {"action": "search", "query": query}

        answer = str(parsed.get("answer", "")).strip()
        citations = [str(c).strip() for c in parsed.get("citations", []) if str(c).strip()]
        # If the model explicitly states that the textbook does not specify/contain the answer,
        # normalize it to an empty answer so run_agent records a clean 'abstain' status.
        lower_ans = answer.lower()
        abstain_phrases = (
            "giáo trình không quy định",
            "tài liệu không quy định",
            "không có thông tin",
            "không đủ bằng chứng",
            "không đề cập",
            "chưa có thông tin",
        )
        if any(phrase in lower_ans for phrase in abstain_phrases):
            return {"action": "answer", "answer": "", "citations": []}

        # Keep only citations that exist in the current context_ids
        valid_citations = [c for c in citations if c in context_ids]
        # Ensure every valid citation tag appears literally as [chunk_id] in the answer string
        if answer and valid_citations:
            missing_tags = [f"[{c}]" for c in valid_citations if f"[{c}]" not in answer]
            if missing_tags:
                answer = f"{answer} {' '.join(missing_tags)}"
        return {"action": "answer", "answer": answer, "citations": valid_citations}


def run_agent_suite(
    config_name: str,
    cases: list[dict[str, Any]],
    retriever: Any,
    model_name: str,
    max_searches: int,
    top_k: int,
    budget: int,
    dense_ref: DensePgRetriever | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the agent across all cases in a split and record detailed traces."""
    controller = OpenAIController(model_name=model_name)
    emb_tokens_before = dense_ref.total_tokens if dense_ref else 0
    traces = []

    for idx, case in enumerate(cases, start=1):
        qid = case["id"]
        question = case["question"]
        q_filter = case.get("filter")
        p_before = controller.prompt_tokens
        c_before = controller.completion_tokens
        calls_before = controller.llm_calls
        started = time.perf_counter()

        try:
            res = run_agent(
                question,
                controller,
                retriever,
                filters=q_filter,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            search_events = [e for e in res["events"] if e.get("event") == "search"]
            trace = {
                "query_id": qid,
                "split": case["split"],
                "question": question,
                "answerable": case["answerable"],
                "filter": q_filter,
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "search_count": len(search_events),
                "llm_calls": controller.llm_calls - calls_before,
                "prompt_tokens": controller.prompt_tokens - p_before,
                "completion_tokens": controller.completion_tokens - c_before,
                "elapsed_ms": round(elapsed_ms, 2),
                "events": res["events"],
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            trace = {
                "query_id": qid,
                "split": case["split"],
                "question": question,
                "answerable": case["answerable"],
                "filter": q_filter,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "answer": "",
                "citations": [],
                "search_count": 0,
                "llm_calls": controller.llm_calls - calls_before,
                "prompt_tokens": controller.prompt_tokens - p_before,
                "completion_tokens": controller.completion_tokens - c_before,
                "elapsed_ms": round(elapsed_ms, 2),
                "events": [],
            }

        traces.append(trace)
        print(
            f"  [{idx}/{len(cases)}] {qid} (answerable={case['answerable']}): "
            f"status={trace['status']} | searches={trace['search_count']} | "
            f"citations={trace['citations']} | {trace['elapsed_ms']}ms"
        )

    emb_tokens = (dense_ref.total_tokens - emb_tokens_before) if dense_ref else 0
    # gpt-4o-mini pricing: $0.15 / 1M input tokens, $0.60 / 1M output tokens
    llm_cost = (
        controller.prompt_tokens / 1_000_000 * 0.15
        + controller.completion_tokens / 1_000_000 * 0.60
    )
    emb_cost = emb_tokens / 1_000_000 * 0.02
    total_cost = round(llm_cost + emb_cost, 6)

    run_summary = {
        "config_name": config_name,
        "model": model_name,
        "max_searches": max_searches,
        "top_k": top_k,
        "budget_chars": budget,
        "cases_total": len(traces),
        "answered_count": sum(t["status"] == "answered" for t in traces),
        "abstain_count": sum(t["status"] == "abstain" for t in traces),
        "error_count": sum(t["status"] == "error" for t in traces),
        "search_count_distribution": {
            str(k): sum(t["search_count"] == k for t in traces)
            for k in sorted({t["search_count"] for t in traces})
        },
        "avg_searches": round(statistics.mean(t["search_count"] for t in traces), 2),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "total_ms": round(sum(t["elapsed_ms"] for t in traces), 2),
        "prompt_tokens": controller.prompt_tokens,
        "completion_tokens": controller.completion_tokens,
        "embedding_tokens": emb_tokens,
        "estimated_cost_usd": total_cost,
    }
    return run_summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 7 Agentic RAG benchmark.")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--mode",
        choices=("selected", "baseline", "both"),
        default="selected",
        help="Which configuration(s) to run with the real LLM agent.",
    )
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    load_env_files()
    data_dir = DAY7_ROOT / "starter" / "data"
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    all_cases = json.loads((data_dir / "cases.json").read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["split"] == args.split]

    tfidf_baseline = TfidfRetriever(chunks, use_filter=False)
    bm25_with_filter = BM25Retriever(chunks, use_filter=True)
    dense_with_filter = DensePgRetriever(use_filter=True)
    hybrid_selected = HybridRRFRetriever(
        dense_with_filter, bm25_with_filter, candidate_k=15, use_filter=True
    )

    to_run = []
    if args.mode in ("baseline", "both"):
        to_run.append(
            (
                "baseline_tfidf_b800",
                tfidf_baseline,
                3,
                3,
                800,
                None,
            )
        )
    if args.mode in ("selected", "both"):
        to_run.append(
            (
                "selected_hybrid_rrf_b4000",
                hybrid_selected,
                3,
                3,
                4000,
                dense_with_filter,
            )
        )

    for cfg_name, retr, max_s, top_k, budget, d_ref in to_run:
        print(f"\n=== Running Agent [{cfg_name}] on split='{args.split}' ===")
        summary, traces = run_agent_suite(
            cfg_name, cases, retr, args.model, max_s, top_k, budget, d_ref
        )
        out_dir = DAY7_ROOT / "starter" / "results" / f"agent_{args.split}_{cfg_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "run_info.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved agent traces to {out_dir}")


if __name__ == "__main__":
    main()
