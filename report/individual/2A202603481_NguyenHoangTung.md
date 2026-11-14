# Báo cáo Day 7 — Agentic RAG

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - **Tài liệu:** Giáo trình Luật Dân sự Việt Nam (Tập 1), chủ biên TS. Nguyễn Ngọc Điện (`document_id: giao-trinh-luat-dan-su-viet-nam-tap-1`).
  - **Nguồn và quyền sử dụng:** Tải từ `https://fdvn.vn/wp-content/uploads/2019/08/luu-ban-nhap-tu-dong-4.pdf` (SHA-256: `22318749e4ae39e76fef251d660cfe21c72641599f31c78c15fec4cfec59b4a8`). Chưa xác minh quyền tái phân phối công khai nên PDF gốc không được commit lên Git; dữ liệu chunks đã chuẩn hóa NFC được xuất ra `starter/data/chunks.json` phục vụ kiểm chứng lab.
  - **Số trang và số chunk:** 153 trang vật lý, sử dụng **142 trang nội dung** (trang 11–152, loại bìa, mục lục và tài liệu tham khảo). Kế thừa chiến lược `B_PARAGRAPH` (sentence-aware paragraph chunking, `max_size=1200`, `overlap=0`) từ Day 6 gồm **518 chunks** (độ dài từ 232 đến 1.198 ký tự, trung bình 969,62 ký tự). Mỗi chunk giữ đầy đủ `id`, `source_id`, `source`, `page` và `metadata` (`category`, `category_label`, `document_type`, `language`, `volume`, `chunk_index`).
- **Vector store, embedding model, phiên bản và số chiều:**
  - **Vector store:** PostgreSQL `16.16` + `pgvector 0.8.7` (Docker container `day6_student_starter-postgres-1`, bảng `document_chunks`, khóa chính `(strategy, chunk_id)`), truy vấn exact cosine similarity (`1 - (embedding <=> query_vec)`).
  - **Embedding model:** OpenAI `text-embedding-3-small`, **1536 chiều**, `encoding_format="float"`.
- **Agent model, cách chạy và phân chia dev/test:**
  - **Agent controller model:** OpenAI `gpt-4o-mini` (`temperature=0.0`, JSON mode) điều phối vòng lặp `run_agent` (`max_searches=3`, `top_k=3`).
  - **Phân chia benchmark (`starter/data/questions.json` & `starter/data/cases.json`):** Tổng cộng **24 câu**, chia độc lập trước khi thử nghiệm:
    - **Tập `dev` (17 câu):** 15 câu có đáp án (`q01`–`q15`, có 3 câu kèm trusted filter `category`) và 2 câu thiếu bằng chứng (`u01`, `u02`).
    - **Tập `test` (7 câu):** 5 câu có đáp án (`q16`–`q20`, có 2 câu kèm trusted filter `category`: `q17`, `q20`) và 2 câu thiếu bằng chứng (`u03`, `u04`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có đáp án (`q01`–`q20`):** Mỗi câu có `gold` gồm `source_id`, `page` và `evidence` trích nguyên văn từ PDF. Retrieval được tính là hit khi chunk trả về đúng `source_id`, đúng `page` và chứa chuỗi `evidence`.
  - **Câu thiếu bằng chứng (`u01`–`u04`):** Các câu hỏi thuộc cùng lĩnh vực Luật Dân sự nhưng hỏi về nội dung ngoài phạm vi Tập 1 (lãi suất vay theo BLDS 2015 `u01`, thời hiệu chia di sản thừa kế `u02`, chi phí quản lý tài sản ký quỹ bị phong tỏa `u03`, mức phạt vi phạm hợp đồng `u04`). Đã quét toàn bộ 518 chunks (142 trang) để xác nhận không có quy định trả lời cho 4 câu này.

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** (`1_baseline_tfidf_b800`) | **Phương pháp 1 (Lexical TF-IDF):** TF-IDF cosine, `top_k=3`, không filter, `budget=800` ký tự | Baseline mặc định của starter | dev | 12/15 (80,0%) | 13/15 (86,7%) | 1/15 (6,7%) | Median 1,88 ms (tổng 27,9 ms) / \$0 |
| **Phương pháp 2** (`2_bm25_b800`) | **Lexical Okapi BM25:** `k1=1.5, b=0.75`, `top_k=3`, không filter, `budget=800` ký tự | Thử chuẩn hóa độ dài đoạn văn và bão hòa tần suất từ bằng BM25 | dev | 9/15 (60,0%) | 12/15 (80,0%) | 1/15 (6,7%) | Median 2,06 ms (tổng 36,3 ms) / \$0 |
| **Phương pháp 3** (`3_dense_pgvector_b800`) | **Dense Retrieval (`pgvector` Day 6):** `text-embedding-3-small` (1536d), exact cosine trên `B_PARAGRAPH`, `top_k=3`, không filter, `budget=800` ký tự | Dùng ngữ nghĩa vector để bắt các câu paraphrase/suy luận (`q06`, `q08`, `q09`) | dev | 13/15 (86,7%) | 14/15 (93,3%) | 1/15 (6,7%) | Median 268,99 ms / 555 tokens (\$0,000011) |
| **Cải thiện 1** (`4_improve1_dense_b4000_filter`) | **Dense + Context Budget 4000 + Trusted Filter:** Dense `pgvector`, `top_k=3`, áp dụng filter `category` khi câu hỏi cung cấp, tăng `budget=4000` ký tự | Khắc phục lỗi cắt bỏ chunk của `pack_context` khi chunk `B_PARAGRAPH` dài tới 1200 ký tự (> budget 800) | dev | 13/15 (86,7%) | 14/15 (93,3%) | **14/15 (93,3%)** | Median 17,08 ms (cache vector) / \$0 bổ sung |
| **Cải thiện 2 / Phương pháp 4** (`5_improve2_hybrid_rrf_b4000_filter`) | **Hybrid RRF (Dense + BM25):** Kết hợp top-15 Dense + top-15 BM25 bằng Reciprocal Rank Fusion (`rrf_k=60`), `top_k=3`, trusted filter `category`, `budget=4000` ký tự | Kết hợp ngữ nghĩa của Dense với khớp từ khóa chính xác của BM25 để cứu câu `q15` (chứa con số 20 năm, 10 năm, 15 năm) | dev | **14/15 (93,3%)** | **15/15 (100%)** | **15/15 (100%)** | Median 20,00 ms / \$0 bổ sung (dùng chung query embedding) |
| **Cấu hình đã chọn** (`5_selected_hybrid_rrf_b4000_filter`) | **Hybrid RRF (Dense + BM25, pool=15)**, `top_k=3`, trusted filter `category`, `budget=4000` ký tự | Cấu hình tốt nhất đã chốt trên `dev` | **test** | **4/5 (80,0%)** | **4/5 (80,0%)** | **4/5 (80,0%)** | Median 287,19 ms (tổng 2,18 s) / 223 tokens (\$0,00000446) |
| **Baseline** (`1_baseline_tfidf_b800`) | **Lexical TF-IDF**, `top_k=3`, không filter, `budget=800` ký tự | Đối chiếu baseline trên `test` | **test** | 4/5 (80,0%) | 4/5 (80,0%) | **0/5 (0,0%)** | Median 2,57 ms (tổng 14,0 ms) / \$0 |

### Phân tích hai lỗi trên tập `dev`, giả thuyết, kết quả trước/sau và lý do chọn cấu hình

1. **Lỗi Dev 1 — Sụp đổ Context Evidence Rate (1/15 = 6,7%) dù Evidence Hit@3 đạt 13/15–14/15:**
   - **Nguyên nhân:** Ở Day 6, chiến lược `B_PARAGRAPH` tạo các chunk theo đoạn văn với `max_size = 1200` ký tự (trung bình 969,62 ký tự/chunk), kèm thêm header `[id | source | page]` dài ~75 ký tự. Trong khi đó, hàm `pack_context` mặc định có `budget = 800` ký tự và bỏ qua mọi chunk làm tổng độ dài vượt `budget`. Kết quả là 13/14 câu tìm trúng chunk ở top-3 (`q01`–`q10`, `q12`–`q14`) đều bị loại khỏi context, chỉ còn duy nhất `q11` (chunk dài 691 ký tự) lọt qua.
   - **Giả thuyết & Cách thử (Cải thiện 1):** Tăng `budget` lên `4000` ký tự để chứa trọn cả 3 chunks của top-3 (`3 × 1275 = 3825 < 4000`), đồng thời bật `category` metadata filter.
   - **Kết quả trước/sau:** Trên tập `dev`, `Context Evidence Rate` của Dense tăng vọt từ **1/15 (6,7%) lên 14/15 (93,3%)** (và đạt **15/15 = 100%** khi kết hợp Hybrid RRF ở Cải thiện 2). Lưu ý: khi thử mức trung gian `budget = 2400`, câu `q15` có chunk đúng nằm ở rank 3 (dài 1.169 ký tự sau 2 chunk đầu dài 1.193 + 629 ký tự, tổng > 2.400) vẫn bị rớt; nâng lên `4000` giải quyết triệt để.

2. **Lỗi Dev 2 — Dense Retrieval trượt câu `q15` (thời hạn bảo hộ sáng chế 20 năm, giải pháp hữu ích 10 năm, kiểu dáng công nghiệp 15 năm):**
   - **Nguyên nhân:** Với câu `q15`, Dense Retrieval (`3_dense_pgvector`) xếp các đoạn ở trang 102 và 116 (bàn chung về quyền sở hữu công nghiệp và thời hiệu) lên top-3 và đẩy đoạn chứa các con số cụ thể ở trang 100 (`p0100:c0001`) xuống ngoài top-3. Ngược lại, BM25 (`2_bm25_b800`) bắt rất nhạy các cụm từ khóa hiếm `"kiểu dáng công nghiệp"`, `"giải pháp hữu ích"`, `"hết hiệu lực"` và xếp đúng chunk `p0100:c0001` lên hạng cao, nhưng BM25 lại yếu ở các câu diễn giải khái niệm (`q06`, `q08`, `q09` chỉ đạt Hit@3 = 12/15).
   - **Giả thuyết & Cách thử (Cải thiện 2 / Phương pháp 4):** Dùng **Hybrid RRF (`candidate_k=15`, `rrf_k=60`)** hợp nhất thứ hạng của Dense và BM25.
   - **Kết quả trước/sau:** Hybrid RRF giữ nguyên toàn bộ các câu đúng của Dense và kéo `p0100:c0001` của `q15` vào top-3, đưa **Hit@1 trên `dev` từ 13/15 lên 14/15 (93,3%)** và **Evidence Hit@3 từ 14/15 lên 15/15 (100%)**.

3. **Lý do chốt cấu hình `5_selected_hybrid_rrf_b4000_filter` cho tập `test`:**
   - Đạt điểm tuyệt đối trên `dev` ở cả `Evidence Hit@3` (15/15) và `Context Evidence Rate` (15/15), chi phí nhúng bằng đúng Dense (~$0,000011 cho cả tập dev) và độ trễ BM25 cộng thêm chỉ ~2 ms/câu.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Kiến trúc Agent (`src/step3_run_agent.py` + `starter/agentic.py`):** Sử dụng model thật OpenAI `gpt-4o-mini` (`OpenAIController`) kết nối với `HybridRRFRetriever` (hoặc `TfidfRetriever` ở baseline).
  - **Chiến lược gọi tool & Điều kiện dừng:**
    - Lượt tìm kiếm thứ nhất (`search #1`) luôn dùng nguyên văn câu hỏi gốc của người dùng để bảo toàn toàn bộ tín hiệu ngữ nghĩa và từ khóa cho Hybrid RRF (tránh việc LLM tự rút gọn làm mất từ khóa ở lượt đầu).
    - Sau lượt 1, LLM đọc `context` hiện tại:
      - Nếu đã đủ bằng chứng: trả về `{"action": "answer", "answer": "... [chunk_id]", "citations": ["chunk_id"]}` và dừng ngay (1 lượt search).
      - Nếu chưa đủ bằng chứng: LLM tự viết lại truy vấn mới (query rewriting, không lặp lại truy vấn cũ) tối đa `max_searches = 3` lượt.
      - Nếu sau khi tìm kiếm vẫn không có thông tin trong tài liệu hoặc chạm ngưỡng `max_searches = 3`, Agent trả về trạng thái từ chối (`status: "abstain"`).
  - **Chọn context & Kiểm soát trích dẫn:** Hàm `pack_context` khử trùng lặp (`deduplication`) theo `chunk_id` qua các lượt tìm kiếm trong phạm vi `budget = 4000` ký tự. Mọi citation ID được đối chiếu nghiêm ngặt với `context_ids`; nếu thiếu thẻ `[chunk_id]` hợp lệ hoặc câu trả lời xác nhận giáo trình không quy định, hệ thống chuẩn hóa về `abstain`. Lỗi kết nối DB/API được bắt và ghi rõ `status: "error"`, không dùng dữ liệu giả.

### Bảng tổng hợp kết quả Agent (Chấm thủ công + `starter/agent_metrics.py`)

| Cấu hình / split | Tool use | Số lượt gọi (phân bố & TB) | Context Grade 0/1/2 và TB | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|
| **Selected Hybrid RRF (`b=4000`) / `dev`** | 17/17 (100%) | `1`: 16 câu, `3`: 1 câu (TB: **1,12**) | `0`: 0, `1`: 0, `2`: 15 (**2,00/2**) | `0`: 0, `1`: 0, `2`: 15 (**2,00/2**) | 16/16 (100%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0,0%) |
| **Baseline TF-IDF (`b=800`) / `test`** | 7/7 (100%) | `2`: 2 câu, `3`: 5 câu (TB: **2,71**) | `0`: 5, `1`: 0, `2`: 0 (**0,00/2**) | `0`: 5, `1`: 0, `2`: 0 (**0,00/2**) | 0/1 (0,0%) | 1/1 (100%) | 2/2 (100%) | **4/5 (80,0%)** |
| **Selected Hybrid RRF (`b=4000`) / `test`** | 7/7 (100%) | `1`: 5 câu, `3`: 2 câu (TB: **1,57**) | `0`: 0, `1`: 1, `2`: 4 (**1,80/2**) | `0`: 0, `1`: 1, `2`: 4 (**1,80/2**) | **5/5 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0,0%)** |

### Chi tiết từng câu trên tập `test` (7 câu: 5 có đáp án + 2 thiếu bằng chứng) — Cấu hình đã chọn (`Selected Hybrid RRF`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án (`paraphrase`) | 1 lượt `search` (không filter) -> Top-1 trúng `p0038:c0000` (trang 38). Context: 3 chunks (2.825 ký tự). | *"Theo quy định pháp luật, mỗi cá nhân chỉ có một nơi cư trú cố định. Nếu một cá nhân có nhiều hơn một nơi cư trú cố định, thì sẽ được coi là luôn di động giữa các điểm cố định đó... Do đó, về mặt pháp lý, một cá nhân không thể có nhiều nơi cư trú cố định cùng lúc `[paragraph:giao-trinh-luat-dan-su-viet-nam-tap-1:p0038:c0000]`."* | **Context: 2/2 · Answer: 2/2 · Citation: Hợp lệ (`True`)**. Trả lời đúng, đủ ý và trích dẫn chính xác đoạn văn trang 38. (Baseline bị `abstain` nhầm do chunk dài 1.152 ký tự > budget 800). |
| `q17` | Có đáp án (`direct`, filter `phap_nhan`) | 1 lượt `search` (filter `category=phap_nhan`) -> Top-1 trúng `p0076:c0002` (trang 76). Context: 3 chunks (3.448 ký tự). | *"Theo BLDS 1995, pháp nhân tham gia quan hệ pháp luật dưới danh nghĩa của chính mình, tức là nhân danh mình tham gia vào các quan hệ pháp luật một cách độc lập... `[paragraph:giao-trinh-luat-dan-su-viet-nam-tap-1:p0076:c0002]`."* | **Context: 2/2 · Answer: 2/2 · Citation: Hợp lệ (`True`)**. Đúng nguyên văn trang 76, tuân thủ filter `phap_nhan`. |
| `q18` | Có đáp án (`exception`) | 1 lượt `search` -> Top-1 trúng `p0097:c0001` (trang 97). Context: 3 chunks (3.719 ký tự). | *"Vật tiêu hao không thể là đối tượng của hợp đồng cho thuê hoặc cho mượn vì khi sử dụng, vật tiêu hao sẽ mất đi hoặc không giữ được tính chất, hình dáng và tính năng sử dụng ban đầu. Sử dụng tài sản tiêu hao đồng nghĩa với việc định đoạt và chuyển quyền sở hữu... `[paragraph:giao-trinh-luat-dan-su-viet-nam-tap-1:p0097:c0001]`."* | **Context: 2/2 · Answer: 2/2 · Citation: Hợp lệ (`True`)**. Giải thích đúng bản chất định đoạt/chuyển quyền sở hữu ở trang 97. |
| `q19` | Có đáp án (`condition`) | 1 lượt `search` -> Trả về `p0115:c0001`, `p0118:c0000`, `p0113:c0000` (cùng nói về thời hiệu xác lập quyền sở hữu nhưng thiếu `p0122:c0001`). | *"Để xác lập quyền sở hữu theo thời hiệu, việc chiếm hữu phải đáp ứng các yêu cầu sau: người chiếm hữu phải không có căn cứ pháp luật nhưng phải chiếm hữu ngay tình, liên tục và công khai trong thời hạn mười năm đối với động sản và ba mươi năm đối với bất động sản... `[paragraph:giao-trinh-luat-dan-su-viet-nam-tap-1:p0113:c0000]`."* | **Context: 1/2 · Answer: 1/2 · Citation: Hợp lệ (`True`)**. Chunk `p0113:c0000` thực sự hỗ trợ các điều kiện ngay tình, liên tục, công khai (nên citation hợp lệ), nhưng câu trả lời chỉ đạt **1/2** vì thiếu hai thuật ngữ học thuật `corpus` và `animus` ở trang 122. |
| `q20` | Có đáp án (`condition`, filter `tai_san_quyen_so_huu`) | 1 lượt `search` (filter `category=tai_san_quyen_so_huu`) -> Top-1 trúng `p0152:c0000` (trang 152). Context: 3 chunks (3.081 ký tự). | *"Quyền sở hữu bề mặt phải được đăng ký nếu các tài sản gắn liền với đất thuộc loại phải đăng ký quyền sở hữu. Ngược lại, nếu tài sản gắn liền với đất không thuộc loại phải đăng ký quyền sở hữu, thì quyền sở hữu bề mặt cũng không được đăng ký... `[paragraph:giao-trinh-luat-dan-su-viet-nam-tap-1:p0152:c0000]`."* | **Context: 2/2 · Answer: 2/2 · Citation: Hợp lệ (`True`)**. Trả lời đầy đủ điều kiện ở trang 152 (trong khi Baseline trích dẫn sai sang trang 123 nói về mua bán tài sản chung nên bị chấm 0). |
| `u03` | Thiếu bằng chứng (`unanswerable`) | 3 lượt `search`: (1) câu hỏi gốc -> (2) `"chi phí quản lý tài sản ký quỹ bị phong tỏa"` -> (3) `"tài sản ký quỹ ngân hàng phong tỏa chi phí quản lý"`. | `status: "abstain"` — *"Không đủ bằng chứng để trả lời."* (`citations: []`) | **Abstention: Đúng (`True`)**. Agent chủ động thử 3 truy vấn khác nhau, nhận thấy giáo trình không đề cập ký quỹ và từ chối chính xác. |
| `u04` | Thiếu bằng chứng (`unanswerable`) | 3 lượt `search`: (1) câu hỏi gốc -> (2) `"mức phạt vi phạm hợp đồng tối đa dân sự"` -> (3) `"phạt vi phạm hợp đồng phần trăm giá trị nghĩa vụ"`. | `status: "abstain"` — *"Không đủ bằng chứng để trả lời."* (`citations: []`) | **Abstention: Đúng (`True`)**. Agent tìm 3 lượt không có chế định phạt vi phạm hợp đồng trong Tập 1 nên từ chối đúng. |

### Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí

- **Đường dẫn dữ liệu & kết quả:**
  - Dữ liệu chuẩn hóa Day 7: `starter/data/chunks.json` (518 chunks), `starter/data/questions.json` (20 câu), `starter/data/cases.json` (24 cases).
  - Kết quả Retrieval (`dev` & `test`): `starter/results/retrieval_dev/comparison_summary.json` và `starter/results/retrieval_test/comparison_summary.json`.
  - Kết quả Agent, Trace và Nhãn chấm (`reviews.json`, `traces.json`, `summary.json`):
    - `starter/results/agent_dev_selected_hybrid_rrf_b4000/`
    - `starter/results/agent_test_baseline_tfidf_b800/`
    - `starter/results/agent_test_selected_hybrid_rrf_b4000/`
- **Lệnh chạy tái lập kết quả:**
  ```bash
  # 1. Xuất chunks từ PostgreSQL Day 6 và tạo bộ câu hỏi Day 7
  python src/step1_prepare_data.py

  # 2. Chạy thực nghiệm so sánh 5 cấu hình Retrieval trên dev và chốt chạy test
  python src/step2_run_retrieval_experiments.py --split dev
  python src/step2_run_retrieval_experiments.py --split test

  # 3. Chạy Agentic RAG thật (gpt-4o-mini) trên dev và test
  python src/step3_run_agent.py --split dev --mode selected
  python src/step3_run_agent.py --split test --mode both

  # 4. Tổng hợp nhãn đánh giá thủ công bằng agent_metrics
  python src/step4_evaluate_agent.py

  # 5. Chạy bộ unit tests
  python -m unittest discover -s tests -v
  ```
- **Thời gian và chi phí thực tế:**
  - **Retrieval (`dev` + `test`):** Tổng cộng 778 embedding tokens (~**\$0,000016**), thời gian trung vị ~20 ms/câu khi có cache và ~270 ms/câu khi gọi API nhúng mới.
  - **Agent (`dev` — 17 cases, Selected):** 66,6 giây (median 3,79 s/câu), 31.457 prompt tokens + 2.554 completion tokens + 707 embedding tokens = **\$0,006265**.
  - **Agent (`test` — 7 cases, Baseline vs Selected):**
    - Baseline TF-IDF: 26,6 giây (median 3,82 s/câu), 14.212 prompt tokens + 611 completion tokens = **\$0,002498** (nhưng phải tìm 2,71 lượt/câu vì context bị rỗng).
    - Selected Hybrid RRF: 29,6 giây (median 4,31 s/câu), 18.662 prompt tokens + 1.004 completion tokens + 454 embedding tokens = **\$0,003411** (chỉ mất 1,0 lượt search cho các câu có đáp án).

### Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn

- **So sánh Trước (Baseline) và Sau (Selected Hybrid RRF) trên tập `test`:**
  - **Grounded Answer Mean:** Tăng từ **0,00/2** (0/5 câu đúng) lên **1,80/2** (4 câu đạt 2/2 tuyệt đối, 1 câu đạt 1/2).
  - **Từ chối nhầm (False Abstention):** Giảm từ **4/5 (80%)** xuống **0/5 (0%)**.
  - **Độ chính xác trích dẫn (Citation Validity):** Tăng từ **0/1 (0%)** (Baseline lấy nhầm trang 123 cho câu `q20`) lên **5/5 (100%)**.
- **Hai lỗi phát hiện và xử lý trong quá trình phát triển Agent trên `dev`:**
  1. *Lỗi viết lại câu hỏi quá sớm ở lượt tìm kiếm đầu tiên (`q15`):* Khi cho LLM tự rút gọn câu hỏi ngay ở lượt search #1, LLM tóm tắt thành cụm từ chung làm mất từ khóa đặc thù, khiến `q15` bị trượt khỏi top-3. Khi sửa `OpenAIController.decide` để **luôn dùng câu hỏi gốc ở lượt search #1** và chỉ viết lại truy vấn từ lượt search #2 trở đi, `q15` tìm trúng ngay ở lượt 1.
  2. *Lỗi hình thức khi từ chối (`u02`):* Ở câu `u02`, LLM trả lời *"Giáo trình không quy định cụ thể thời hiệu khởi kiện chia di sản..."* nhưng vẫn đính kèm citation trang 116 nói về thời hiệu chiếm hữu, làm hệ thống tính là `answered` thay vì `abstain`. Sau khi chuẩn hóa: nếu câu trả lời khẳng định giáo trình không quy định/không đề cập thì trả về `abstain` rỗng, Agent đạt 2/2 abstention trên cả `dev` và `test`.
- **Ví dụ lỗi còn tồn tại trên `test` (`q19`) và Giới hạn:**
  - Ở câu `q19` (*"Để xác lập quyền sở hữu theo thời hiệu, việc chiếm hữu phải đáp ứng những yêu cầu nào?"*), giáo trình trình bày rải rác ở nhiều trang: trang 113–118 nêu điều kiện luật thực định (*ngay tình, liên tục, công khai*), còn trang 122 mới phân tích học thuyết La Mã (*corpus* và *animus*). Vì cả 3 chunks ở trang 113, 115, 118 đều khớp rất mạnh từ khóa và ngữ nghĩa nên đã lấp đầy `top_k = 3`, khiến Agent dừng lại ở lượt 1 và trả lời thiếu ý `corpus/animus` (chấm 1/2).
  - **Đánh đổi & Hướng cải thiện:** Tăng `top_k` từ 3 lên 5 hoặc thêm bước Cross-Encoder Reranker trên top-15 candidate sẽ giúp đưa cả trang 113 và trang 122 vào cùng một context, đổi lại chi phí prompt token mỗi lượt tăng khoảng 40%.

