# Hướng Dẫn Benchmark Toàn Bộ Dữ Liệu trên Milvus

Tài liệu này hướng dẫn chi tiết từng bước để **nạp 100% corpus** vào Milvus và chạy benchmark **toàn bộ queries + qrels** — đảm bảo mọi tài liệu đáp án đều tồn tại trong cơ sở dữ liệu vector trước khi đánh giá.

---

## 📌 Vấn đề hiện tại

Milvus **mới chỉ chứa một phần nhỏ** dữ liệu so với tổng corpus gốc:

| Bộ dữ liệu | Collection | Đã nạp | Tổng Corpus gốc | Thiếu | Queries | Qrels |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Zalo AI Legal** | `zalo_legal` | **5.000** | **61.425** | 56.425 | 818 | 793 |
| **NQ-VN** | `nq_vn` | **2.000** | **2.681.468** | 2.679.468 | 1.900 | 2.283 |

**Hệ quả:** Khi chạy benchmark trên 818 câu hỏi Zalo, chỉ có **52 câu hỏi** tìm được tài liệu đáp án (vì đáp án của 766 câu còn lại nằm trong 56.425 tài liệu chưa nạp). Kết quả benchmark do đó **chưa phản ánh đúng năng lực thực sự** của hệ thống.

**Mục tiêu:** Nạp đủ 100% corpus → Chạy benchmark trên 100% queries → Đối sánh với 100% qrels → Ra kết quả chính xác tuyệt đối.

---

## 🚀 Phương pháp 1: Benchmark Tự động (Khuyến nghị)

Script `benchmark_full_corpus.py` hỗ trợ **tự động nạp nốt toàn bộ corpus rồi chạy đánh giá** chỉ bằng 1 lệnh duy nhất.

### Bước 1: Benchmark bộ Zalo AI Legal (61.425 điều khoản)

```bash
python benchmark_full_corpus.py --dataset zalo --auto-index-full
```

**Quá trình xử lý tự động:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Kiểm tra: Milvus hiện có 5.000 / 61.425 docs           │
│ 2. Tự động nạp 56.425 docs còn lại (~12-15 phút trên CPU) │
│ 3. Load collection vào bộ nhớ                              │
│ 4. Chạy 818 queries qua Milvus Search                      │
│ 5. Đối sánh từng kết quả với 793 qrels ground-truth        │
│ 6. Xuất báo cáo Recall@K, Latency, QPS                    │
└─────────────────────────────────────────────────────────────┘
```

**Thời gian ước tính:**
- Nạp dữ liệu (embedding + insert): ~12-15 phút (tốc độ ~60 docs/giây trên CPU).
- Đánh giá 818 queries: ~25 giây.

### Bước 2 (Khuyên dùng cho NQ-VN): Benchmark Siêu Tốc bằng Smart Pool (~2 đến 4 phút)

Do bộ NQ-VN có **2.681.468 tài liệu** (nạp toàn bộ mất 11-13 tiếng trên CPU), ta sử dụng script `benchmark_smart_pool.py`:
- Tự động lấy **100% tài liệu đáp án trong `qrels`** (đúng 2.283 tài liệu).
- Bổ sung tài liệu gây nhiễu (distractors) để tạo pool 5.000 hoặc 15.000 docs.
- Đánh giá **100% câu hỏi (1.900 queries)** đối chiếu với đáp án chuẩn.
- Thời gian chạy: **chỉ mất 2 ~ 4 phút**!

```bash
# Benchmark NQ-VN với 100% câu hỏi (Pool 5.000 docs - Chạy cực nhanh ~2 phút):
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --recreate

# Benchmark NQ-VN với Pool 15.000 docs (Độ khó cao hơn, nhiều distractors hơn):
python benchmark_smart_pool.py --dataset nq --pool-size 15000 --recreate
```

### Bước 3 (Tùy chọn): Benchmark NQ-VN toàn bộ 2.68 triệu docs (Cần 11-13 tiếng)

```bash
python benchmark_full_corpus.py --dataset nq --auto-index-full
```

### Bước 3 (Tùy chọn): Xuất kết quả ra file JSON

```bash
python benchmark_full_corpus.py --dataset zalo --auto-index-full --output-json zalo_full_benchmark.json
```

File JSON chứa toàn bộ các số liệu chi tiết: Recall@K từng mức K, phân vị Latency (P50, P90, P95, P99), Sequential QPS, Batch QPS,...

---

## 🔧 Phương pháp 2: Benchmark Thủ công (Từng bước)

Nếu muốn kiểm soát chính xác từng giai đoạn, chạy tách riêng.

### Bước 1: Nạp toàn bộ corpus vào Milvus

```bash
# Nạp toàn bộ 61.425 điều khoản Zalo Legal (xóa dữ liệu cũ, nạp lại từ đầu)
python index_milvus.py --dataset zalo --recreate
```

Theo dõi tiến trình trên màn hình:
```
[*] Bắt đầu tạo embedding và nạp 61,425 tài liệu vào Milvus...
Indexing zalo_legal:  15%|█▌        | 9214/61425 [02:31<14:16, 60.9doc/s]
```

Khi hoàn tất sẽ hiển thị:
```
[✓] Hoàn thành nạp 'zalo_legal': 61,425 docs trong 1012.35s (60.7 docs/s)

📊 TỔNG KẾT TRẠNG THÁI CÁC COLLECTIONS TRONG MILVUS:
  • zalo_legal     : 61,425 bản ghi    ← Đủ 100%
```

### Bước 2: Xác nhận dữ liệu đã nạp đủ

```bash
python -c "from pymilvus import MilvusClient; c=MilvusClient('milvus_data.db'); print(c.get_collection_stats('zalo_legal')); c.close()"
```

Kết quả phải hiển thị:
```
{'row_count': 61425}
```

### Bước 3: Chạy benchmark đánh giá toàn bộ

```bash
python evaluate_milvus.py --dataset zalo --top-k-max 100
```

Lệnh này sẽ:
- Nạp toàn bộ **818 queries** từ dataset.
- Nạp toàn bộ **793 qrels** (đáp án chuẩn).
- Với mỗi câu hỏi: tìm kiếm Top-100 trên Milvus, đối sánh với qrels.
- In bảng **Recall@K**, **Latency Breakdown** (P50/P90/P95/P99) và **Throughput (QPS)**.

---

## 📊 Chỉ số đo lường và ý nghĩa

### 1. Recall@K — Độ chính xác truy xuất

$$\text{Recall@K} = \frac{|\text{Top-K Retrieved} \cap \text{Relevant (qrels)}|}{|\text{Relevant (qrels)}|}$$

| Chỉ số | Ý nghĩa | Cách tính |
| :--- | :--- | :--- |
| **Recall@1** | Tài liệu đứng đầu kết quả tìm kiếm có đúng là đáp án qrels không? | Giao tập hợp tại vị trí Top-1 |
| **Recall@5** | Trong 5 tài liệu đầu tiên, có bao nhiêu phần trăm đáp án qrels được tìm thấy? | Giao tập hợp tại Top-5 |
| **Recall@10** | Tương tự, mở rộng phạm vi tìm kiếm lên Top-10 | Giao tập hợp tại Top-10 |
| **Recall@50** | Trong 50 tài liệu đầu, tìm được bao nhiêu phần trăm tổng đáp án? | Giao tập hợp tại Top-50 |
| **Hit Rate@K** | Có **ít nhất 1** tài liệu đúng trong Top-K hay không? (1 hoặc 0) | Binary hit per query |
| **MRR@K** | Thứ hạng trung bình nghịch đảo (Mean Reciprocal Rank) — thưởng điểm theo vị trí: Top-1 = 1.0 điểm, Top-2 = 0.5 điểm, Top-3 = 0.33 điểm,... | 1/rank của tài liệu đúng đầu tiên |

### 2. Latency — Độ trễ truy vấn

| Chỉ số | Ý nghĩa |
| :--- | :--- |
| **Embedding Latency** | Thời gian để mô hình SentenceTransformer chuyển đổi 1 câu hỏi thành vector 384 chiều (chạy trên CPU) |
| **Milvus Search Latency** | Thời gian Milvus tìm kiếm Top-K vector gần nhất trong cơ sở dữ liệu (ANN Search) |
| **End-to-End Latency** | Tổng = Embedding + Milvus Search (= thời gian user thực sự phải chờ) |
| **P50 / P90 / P95 / P99** | Phân vị: P90 = 90% truy vấn có latency thấp hơn giá trị này |

### 3. Throughput (QPS) — Thông lượng hệ thống

| Chỉ số | Ý nghĩa |
| :--- | :--- |
| **Sequential QPS** | Số truy vấn xử lý mỗi giây khi gửi tuần tự từng câu một (đo bottleneck thực tế) |
| **Batch QPS** | Số truy vấn xử lý mỗi giây khi gửi theo nhóm (batch_size=32) — khai thác song song hóa |
| **Milvus Pure QPS** | Chỉ tính riêng bước Milvus Search (bỏ qua thời gian embedding) |

---

## 📋 Tổng hợp các lệnh chạy benchmark

```bash
# ─────────────────────────────────────────────────────────────
# A. BENCHMARK TOÀN BỘ ZALO LEGAL (61.425 docs) — CÁCH NHANH NHẤT
# ─────────────────────────────────────────────────────────────
python benchmark_full_corpus.py --dataset zalo --auto-index-full

# ─────────────────────────────────────────────────────────────
# B. BENCHMARK TOÀN BỘ ZALO LEGAL — CÁCH THỦ CÔNG TỪNG BƯỚC
# ─────────────────────────────────────────────────────────────
# B1. Nạp toàn bộ corpus vào Milvus
python index_milvus.py --dataset zalo --recreate

# B2. Chạy đánh giá trên toàn bộ queries + qrels
python evaluate_milvus.py --dataset zalo --top-k-max 100

# ─────────────────────────────────────────────────────────────
# C. BENCHMARK NQ-VN (2.68 triệu docs) — CHỈ NẠP 50K ĐỂ THỬ
# ─────────────────────────────────────────────────────────────
python index_milvus.py --dataset nq --limit-nq 50000 --recreate
python evaluate_milvus.py --dataset nq --top-k-max 100

# ─────────────────────────────────────────────────────────────
# D. BENCHMARK NQ-VN TOÀN BỘ (CẦN NHIỀU GIỜ)
# ─────────────────────────────────────────────────────────────
python benchmark_full_corpus.py --dataset nq --auto-index-full

# ─────────────────────────────────────────────────────────────
# E. XUẤT KẾT QUẢ RA FILE JSON
# ─────────────────────────────────────────────────────────────
python benchmark_full_corpus.py --dataset zalo --auto-index-full --output-json zalo_benchmark.json
```

---

## ⏱️ Bảng ước tính thời gian chạy

| Tác vụ | Zalo Legal (61.425 docs) | NQ-VN (2.681.468 docs) |
| :--- | :---: | :---: |
| **Nạp dữ liệu (Embedding + Insert)** | ~12-15 phút | ~11-13 tiếng |
| **Đánh giá Queries (818 / 1.900 queries)** | ~25 giây | ~55 giây |
| **Tổng thời gian benchmark toàn bộ** | **~15-16 phút** | **~11-13 tiếng** |
| **Dung lượng ổ cứng (milvus_data.db)** | ~142 MB | ~6.5 GB |
| **RAM cần thiết** | ~130 MB | ~5.5 GB |

---

## 🔍 Cách đọc kết quả báo cáo

Sau khi chạy xong, báo cáo sẽ hiển thị dạng bảng tương tự:

```
==================================================================================
📊 BÁO CÁO ĐÁNH GIÁ & DỰ BÁO HIỆU NĂNG MILVUS KHI NẠP TOÀN BỘ DỮ LIỆU
   • Bộ dữ liệu:      Zalo AI Legal Text Retrieval (Pháp luật Việt Nam)
   • Quy mô:           61,425 tài liệu (100% corpus)
   • Số câu hỏi:       818 queries đối sánh với 793 qrels ground-truth
==================================================================================

1. ĐỘ CHÍNH XÁC TRUY XUẤT:
----------------------------------------------------------------------------------
Metric             | K = 1    | K = 3    | K = 5    | K = 10   | K = 20   | K = 50
----------------------------------------------------------------------------------
Recall@K           | xx.xx%   | xx.xx%   | xx.xx%   | xx.xx%   | xx.xx%   | xx.xx%
Hit Rate@K         | xx.xx%   | xx.xx%   | xx.xx%   | xx.xx%   | xx.xx%   | xx.xx%
MRR@K              | 0.xxxx   | 0.xxxx   | 0.xxxx   | 0.xxxx   | 0.xxxx   | 0.xxxx
----------------------------------------------------------------------------------

2. ĐỘ TRỄ TRUY VẤN (LATENCY):
----------------------------------------------------------------------------------
Giai đoạn                | Mean     | P50      | P90      | P95      | P99
----------------------------------------------------------------------------------
1. Query Embedding       | xx.xxms  | xx.xxms  | xx.xxms  | xx.xxms  | xx.xxms
2. Milvus ANN Search     | xx.xxms  | xx.xxms  | xx.xxms  | xx.xxms  | xx.xxms
-> End-to-End Total      | xx.xxms  | xx.xxms  | xx.xxms  | xx.xxms  | xx.xxms
----------------------------------------------------------------------------------

3. THÔNG LƯỢNG HỆ THỐNG (QPS):
----------------------------------------------------------------------------------
  • Sequential QPS:   xx.xx queries / giây
  • Batched QPS:      xx.xx queries / giây
==================================================================================
```

**Các chỉ số "tốt" tham khảo cho bài toán Text Retrieval tiếng Việt:**

| Chỉ số | Mức Trung bình | Mức Tốt | Mức Xuất sắc |
| :--- | :---: | :---: | :---: |
| **Recall@10** | 40-55% | 55-75% | > 75% |
| **MRR@10** | 0.25-0.35 | 0.35-0.50 | > 0.50 |
| **End-to-End Latency (P50)** | < 50ms | < 30ms | < 15ms |
| **Sequential QPS** | > 20 q/s | > 40 q/s | > 80 q/s |

