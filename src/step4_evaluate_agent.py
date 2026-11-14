"""Step 4: Build human-verified audit labels (reviews.json) and aggregate agent metrics."""

import json
import sys
from pathlib import Path

DAY7_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = DAY7_ROOT / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from agent_metrics import aggregate  # noqa: E402


def build_reviews() -> dict[str, list[dict]]:
    """Return manual audit reviews for dev_selected, test_baseline, and test_selected."""
    # 1. Reviews for Selected Config on DEV (15 answerable + 2 unanswerable)
    dev_selected_reviews = [
        {
            "query_id": "q01",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Đúng ngày thông qua 14/06/2005 và ngày có hiệu lực 01/01/2006 từ trang 22.",
        },
        {
            "query_id": "q02",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Nêu đủ 4 chủ thể: cá nhân, pháp nhân, hộ gia đình, tổ hợp tác từ trang 23.",
        },
        {
            "query_id": "q03",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Đúng 3 yếu tố: họ và tên, hộ tịch, nơi cư trú từ trang 23.",
        },
        {
            "query_id": "q04",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Đúng khái niệm Bí danh ở trang 23.",
        },
        {
            "query_id": "q05",
            "answer_grade": 2,
            "citation_support": [True, True],
            "context_grade": 2,
            "notes": "Nêu đúng chức năng định vị cá nhân trong không gian tại điểm cố định trên lãnh thổ từ trang 38.",
        },
        {
            "query_id": "q06",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Nêu đủ 2 nhóm người hoàn toàn không có năng lực hành vi ở trang 51.",
        },
        {
            "query_id": "q07",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Đúng định nghĩa quyền nhân thân theo BLDS 2005 ở trang 68.",
        },
        {
            "query_id": "q08",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Nêu đủ 4 đặc điểm cơ bản của pháp nhân ở trang 76.",
        },
        {
            "query_id": "q09",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Nêu đủ 2 điều kiện của quyền tài sản ở trang 93.",
        },
        {
            "query_id": "q10",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Phân biệt đúng hoa lợi và lợi tức ở trang 95.",
        },
        {
            "query_id": "q11",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Đúng quy tắc vật phụ mặc nhiên đi theo vật chính nếu không có thỏa thuận khác ở trang 96.",
        },
        {
            "query_id": "q12",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Đúng định nghĩa vật tiêu hao ở trang 97.",
        },
        {
            "query_id": "q13",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Phân biệt đúng vật cùng loại và vật đặc định ở trang 97.",
        },
        {
            "query_id": "q14",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Giải thích đúng vì tài sản vô hình không thể chiếm hữu vật chất nên chỉ khởi kiện/khiếu nại ở trang 100.",
        },
        {
            "query_id": "q15",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Đúng thời hạn sáng chế 20 năm, giải pháp hữu ích 10 năm, kiểu dáng công nghiệp 15 năm ở trang 100.",
        },
        {
            "query_id": "u01",
            "citation_support": [],
            "abstention_correct": True,
            "notes": "Từ chối đúng sau 3 lượt tìm vì Tập 1 không có quy định lãi suất vay của BLDS 2015.",
        },
        {
            "query_id": "u02",
            "citation_support": [],
            "abstention_correct": True,
            "notes": "Từ chối đúng vì Tập 1 không có chương Thừa kế hay thời hiệu chia di sản.",
        },
    ]

    # 2. Reviews for Baseline (TF-IDF, budget=800) on TEST (5 answerable + 2 unanswerable)
    test_baseline_reviews = [
        {
            "query_id": "q16",
            "answer_grade": 0,
            "citation_support": [],
            "context_grade": 0,
            "notes": "Chunk đúng trang 38 dài 1152 ký tự > budget 800 nên bị pack_context loại bỏ; context trống dẫn đến từ chối nhầm.",
        },
        {
            "query_id": "q17",
            "answer_grade": 0,
            "citation_support": [],
            "context_grade": 0,
            "notes": "Chunk đúng trang 76 dài 1016 ký tự > budget 800 nên bị loại bỏ; từ chối nhầm.",
        },
        {
            "query_id": "q18",
            "answer_grade": 0,
            "citation_support": [],
            "context_grade": 0,
            "notes": "Chunk đúng trang 97 dài 1143 ký tự > budget 800 nên bị loại bỏ; từ chối nhầm.",
        },
        {
            "query_id": "q19",
            "answer_grade": 0,
            "citation_support": [],
            "context_grade": 0,
            "notes": "Không tìm được gold trang 122 và các chunk tìm được cũng bị cắt do budget=800; từ chối nhầm.",
        },
        {
            "query_id": "q20",
            "answer_grade": 0,
            "citation_support": [False],
            "context_grade": 0,
            "notes": "Chunk đúng trang 152 dài 940 ký tự bị loại do budget=800; chunk p0123:c0002 ngắn nên lọt vào context nhưng chỉ nói về đăng ký mua bán chung, không phải quyền sở hữu bề mặt (citation không hỗ trợ mệnh đề quyền sở hữu bề mặt).",
        },
        {
            "query_id": "u03",
            "citation_support": [],
            "abstention_correct": True,
            "notes": "Từ chối đúng câu hỏi về chi phí quản lý tài sản ký quỹ.",
        },
        {
            "query_id": "u04",
            "citation_support": [],
            "abstention_correct": True,
            "notes": "Từ chối đúng câu hỏi về mức phạt vi phạm hợp đồng.",
        },
    ]

    # 3. Reviews for Selected Config (Hybrid RRF + Filter + budget=4000) on TEST
    test_selected_reviews = [
        {
            "query_id": "q16",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Trả lời đúng và đủ rằng mỗi cá nhân chỉ có một nơi cư trú cố định, kèm giải thích và trích dẫn đúng trang 38 [p0038:c0000].",
        },
        {
            "query_id": "q17",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Trả lời đúng pháp nhân nhân danh chính mình tham gia quan hệ pháp luật một cách độc lập, trích dẫn đúng trang 76 [p0076:c0002].",
        },
        {
            "query_id": "q18",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Giải thích đúng và đủ vì sử dụng vật tiêu hao đồng nghĩa với định đoạt và chuyển quyền sở hữu, trích dẫn đúng trang 97 [p0097:c0001].",
        },
        {
            "query_id": "q19",
            "answer_grade": 1,
            "citation_support": [True],
            "context_grade": 1,
            "notes": "Đúng một phần (1/2): nêu đúng điều kiện chiếm hữu ngay tình, liên tục, công khai theo trang 113 [p0113:c0000], nhưng thiếu hai thuật ngữ học thuật 'corpus' và 'animus' nằm ở trang 122.",
        },
        {
            "query_id": "q20",
            "answer_grade": 2,
            "citation_support": [True],
            "context_grade": 2,
            "notes": "Trả lời đúng và đủ điều kiện đăng ký quyền sở hữu bề mặt khi tài sản gắn liền với đất thuộc loại phải đăng ký, trích dẫn đúng trang 152 [p0152:c0000].",
        },
        {
            "query_id": "u03",
            "citation_support": [],
            "abstention_correct": True,
            "notes": "Từ chối đúng sau 3 lượt tìm kiếm vì giáo trình Tập 1 không có quy định về ký quỹ.",
        },
        {
            "query_id": "u04",
            "citation_support": [],
            "abstention_correct": True,
            "notes": "Từ chối đúng sau 3 lượt tìm kiếm vì giáo trình Tập 1 không có quy định về mức phạt vi phạm hợp đồng.",
        },
    ]

    return {
        "agent_dev_selected_hybrid_rrf_b4000": dev_selected_reviews,
        "agent_test_baseline_tfidf_b800": test_baseline_reviews,
        "agent_test_selected_hybrid_rrf_b4000": test_selected_reviews,
    }


def main() -> None:
    cases_path = DAY7_ROOT / "starter" / "data" / "cases.json"
    all_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    reviews_map = build_reviews()

    for folder_name, reviews in reviews_map.items():
        split = "dev" if "_dev_" in folder_name else "test"
        split_cases = [c for c in all_cases if c["split"] == split]
        target_dir = DAY7_ROOT / "starter" / "results" / folder_name
        traces_path = target_dir / "traces.json"
        traces = json.loads(traces_path.read_text(encoding="utf-8"))

        # Save manual reviews
        reviews_path = target_dir / "reviews.json"
        reviews_path.write_text(
            json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        # Aggregate metrics using starter/agent_metrics.py
        agg = aggregate(split_cases, traces, reviews)
        # Also compute context grade distribution (0/1/2) on answerable questions
        ctx_grades = [
            r["context_grade"] for r in reviews if "context_grade" in r
        ]
        agg["context_grades"] = {
            "0": sum(g == 0 for g in ctx_grades),
            "1": sum(g == 1 for g in ctx_grades),
            "2": sum(g == 2 for g in ctx_grades),
            "N": len(ctx_grades),
            "mean": round(sum(ctx_grades) / len(ctx_grades), 4) if ctx_grades else None,
        }

        summary_obj = {"split": split, "config": folder_name, **agg}
        (target_dir / "summary.json").write_text(
            json.dumps(summary_obj, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"=== {folder_name} ({split}) ===")
        print(json.dumps(summary_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

