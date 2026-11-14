"""Step 1: Export chunks from Day 6 pgvector and build Day 7 benchmark files."""

import json
import os
from pathlib import Path
import psycopg

# Define paths for Day 6 and Day 7 folders
DAY7_ROOT = Path(__file__).resolve().parents[1]
DAY6_ROOT = DAY7_ROOT.parent / "K5-L3A-DAY06-NguyenHoangTung-2A202603481-DataIngestion"
OUTPUT_DIR = DAY7_ROOT / "starter" / "data"

# Default local PostgreSQL connection string from Day 6
DEFAULT_DB_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
STRATEGY = "B_PARAGRAPH"


def export_chunks_from_db(db_url: str, strategy: str) -> list[dict]:
    """Read chunks from Day 6 PostgreSQL table and convert to Day 7 format."""
    query = """
        SELECT chunk_id, document_id, source, page, text, metadata
        FROM document_chunks
        WHERE strategy = %s
        ORDER BY page ASC, chunk_id ASC
    """
    chunks = []
    # Connect to PostgreSQL and fetch all rows for B_PARAGRAPH
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(query, (strategy,))
        for row in cur.fetchall():
            chunk_id, doc_id, source, page, text, metadata = row
            chunks.append({
                "id": chunk_id,          # Unique chunk ID
                "source_id": doc_id,     # Document identifier
                "source": source,        # File path or URL
                "page": int(page),       # Physical page number
                "text": text,            # Normalized text content
                "metadata": metadata,    # Includes 'category', 'chunk_index', etc.
            })
    return chunks


def convert_day6_questions(day6_questions_path: Path) -> list[dict]:
    """Convert Day 6 questions.json to Day 7 schema."""
    raw_questions = json.loads(day6_questions_path.read_text(encoding="utf-8"))
    converted = []
    for q in raw_questions:
        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            # Day 7 allows multiple valid gold evidences per question
            "gold": [
                {
                    "source_id": q["gold_source_id"],
                    "page": int(q["gold_page"]),
                    "evidence": q["evidence_quote"],
                }
            ],
        }
        # Keep trusted metadata filter only if provided in Day 6
        if q.get("filter"):
            item["filter"] = q["filter"]
        converted.append(item)
    return converted


def build_agent_cases(answerable_questions: list[dict]) -> list[dict]:
    """Combine 20 answerable questions with 4 unanswerable questions (2 dev, 2 test)."""
    cases = []
    for q in answerable_questions:
        case = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
        }
        if "filter" in q:
            case["filter"] = q["filter"]
        cases.append(case)

    # Add 4 unanswerable questions (2 dev, 2 test).
    # Topic is still Civil Law, but these specific details are NOT in Volume 1.
    unanswerable_cases = [
        {
            "id": "u01",
            "split": "dev",
            "question": "Theo giáo trình, mức lãi suất tối đa đối với hợp đồng vay tài sản theo Bộ luật Dân sự 2015 là bao nhiêu phần trăm một năm?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Giáo trình Tập 1 viết theo BLDS 2005 và chỉ gồm phần Giới thiệu, Cá nhân, Pháp nhân, Tài sản - Quyền sở hữu; không có phần Hợp đồng vay hay mức lãi suất của BLDS 2015.",
        },
        {
            "id": "u02",
            "split": "dev",
            "question": "Giáo trình quy định thời hiệu khởi kiện để người thừa kế yêu cầu chia di sản đối với bất động sản là bao nhiêu năm?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ 142 trang nội dung Tập 1: chỉ đề cập thời hiệu xác lập quyền sở hữu do chiếm hữu, không có chương Pháp luật Thừa kế hay thời hiệu chia di sản.",
        },
        {
            "id": "u03",
            "split": "test",
            "question": "Theo giáo trình, trường hợp tài sản ký quỹ tại ngân hàng bị phong tỏa thì chi phí quản lý tài sản ký quỹ do bên nào chịu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ 142 trang nội dung Tập 1: không đề cập đến biện pháp bảo đảm thực hiện nghĩa vụ bằng ký quỹ hay chi phí quản lý tài sản ký quỹ.",
        },
        {
            "id": "u04",
            "split": "test",
            "question": "Giáo trình quy định mức phạt vi phạm hợp đồng tối đa trong dân sự là bao nhiêu phần trăm giá trị phần nghĩa vụ bị vi phạm?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tập 1 không bao gồm phần Nghĩa vụ dân sự và Hợp đồng dân sự, nên không có quy định về mức phạt vi phạm hợp đồng.",
        },
    ]
    cases.extend(unanswerable_cases)
    return cases


def main() -> None:
    db_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Export chunks from PostgreSQL
    chunks = export_chunks_from_db(db_url, STRATEGY)
    print(f"Exported {len(chunks)} chunks from strategy '{STRATEGY}'.")

    # 2. Convert Day 6 questions
    day6_q_path = DAY6_ROOT / "benchmark" / "questions.json"
    questions = convert_day6_questions(day6_q_path)
    print(f"Converted {len(questions)} answerable questions.")

    # 3. Build 24 agent cases (17 dev + 7 test)
    cases = build_agent_cases(questions)
    print(f"Built {len(cases)} total agent cases.")

    # Save all files as UTF-8 JSON
    (OUTPUT_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved chunks.json, questions.json, and cases.json to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()