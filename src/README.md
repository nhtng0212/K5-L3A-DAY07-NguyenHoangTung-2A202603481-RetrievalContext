# Code — Day 7 Agentic RAG

Thư mục này chứa toàn bộ mã nguồn thực thi pipeline **Agentic RAG** (kế thừa dữ liệu và vector store `pgvector` từ Day 6):

| File | Chức năng |
|---|---|
| `step1_prepare_data.py` | Xuất 518 chunks (`B_PARAGRAPH`) từ PostgreSQL + `pgvector` Day 6 ra `starter/data/chunks.json`; chuyển đổi 20 câu hỏi có đáp án (`questions.json`) và bổ sung 4 câu hỏi thiếu bằng chứng (`u01`–`u04`) thành `starter/data/cases.json` (17 `dev` + 7 `test`). |
| `retrievers.py` | Hiện thực 4 phương pháp truy xuất: `TfidfRetriever` (Lexical TF-IDF), `BM25Retriever` (Lexical Okapi BM25), `DensePgRetriever` (Dense vector search với OpenAI `text-embedding-3-small` 1536d trên `pgvector`), và `HybridRRFRetriever` (kết hợp Dense + BM25 bằng Reciprocal Rank Fusion). |
| `step2_run_retrieval_experiments.py` | Chạy benchmark so sánh các phương pháp retrieval và cấu hình cải thiện (`budget=4000`, trusted filter `category`, Hybrid RRF) trên tập `dev`, sau đó đánh giá độc lập trên tập `test`. |
| `step3_run_agent.py` | Hiện thực `OpenAIController` (`gpt-4o-mini`, JSON mode) kết nối với vòng lặp `run_agent`, hỗ trợ multi-turn search, query rewriting, kiểm tra thẻ trích dẫn `[chunk_id]` và chuẩn hóa từ chối (`abstain`). |
| `step4_evaluate_agent.py` | Lưu nhãn đánh giá thủ công (`reviews.json`) và gọi `starter/agent_metrics.py` để tổng hợp kết quả đánh giá Agent (`summary.json`). |

## Hướng dẫn chạy tái lập

```bash
# 1. Xuất dữ liệu từ DB Day 6 và tạo benchmark Day 7
python src/step1_prepare_data.py

# 2. Chạy thực nghiệm Retrieval trên dev và test
python src/step2_run_retrieval_experiments.py --split dev
python src/step2_run_retrieval_experiments.py --split test

# 3. Chạy Agentic RAG trên dev và test
python src/step3_run_agent.py --split dev --mode selected
python src/step3_run_agent.py --split test --mode both

# 4. Tổng hợp điểm đánh giá Agent (Human Audit + agent_metrics)
python src/step4_evaluate_agent.py

# 5. Chạy kiểm thử đơn vị (Unit tests)
python -m unittest discover -s tests -v
```
