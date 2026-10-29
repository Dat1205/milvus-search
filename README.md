# Hướng dẫn Tìm kiếm và Đánh giá Đa Bộ Dữ liệu với Milvus Vector Database

Dự án này tích hợp và vận hành đồng thời **2 bộ dữ liệu** trên **Milvus Vector Database**:
1. **Zalo AI Legal**: 61,425 điều khoản văn bản pháp luật Việt Nam (`zalo_legal`).
2. **Natural Questions VN (NQ-VN)**: 2.68 triệu bài viết/đoạn trích tri thức bách khoa mở kèm bản dịch và gốc tiếng Anh (`nq_vn`).

Toàn bộ dữ liệu vector của cả 2 bộ được quản lý trong cơ sở dữ liệu thống nhất: **`milvus_data.db`**.

---

## 📁 Cấu trúc Thư mục

```text
milvus/
├── download_data.py          # Script tải dữ liệu từ Hugging Face
├── index_milvus.py           # Script trích xuất vector embedding và nạp vào Milvus
├── search_milvus.py          # Script tìm kiếm (CLI một câu hỏi, REPL tương tác)
├── evaluate_milvus.py        # Script đo lường Recall@K, Latency và QPS trên dữ liệu hiện có
├── benchmark_full_corpus.py  # Script tính toán & dự báo hiệu năng khi nạp 100% TOÀN BỘ dữ liệu
├── benchmark_smart_pool.py   # Script benchmark siêu tốc (Ground-Truth Pool: 100% queries chỉ 3-5 phút)
├── app.py                    # Ứng dụng Web UI tìm kiếm song song bằng Flask
├── templates/
│   └── index.html            # Giao diện Web trực quan (Tailwind CSS, hỗ trợ tab chọn bộ dữ liệu)
├── docker-compose.yml        # File khởi chạy Milvus Standalone Server + Attu Web GUI qua Docker
├── TRIEN_KHAI_MILVUS.md      # Tài liệu hướng dẫn chi tiết toàn tập cách triển khai Milvus
├── HUONG_DAN_BENCHMARK.md    # Hướng dẫn chi tiết cách benchmark toàn bộ dữ liệu
├── HUONG_DAN_CHAY.md         # Hướng dẫn chạy nhanh từng bước (Quickstart Step-by-Step)
├── README.md                 # Tài liệu tổng quan dự án
└── milvus_data.db/           # Thư mục lưu trữ database vector cục bộ của Milvus Lite
```

---

## 📈 1. Tính toán & Dự báo Hiệu năng khi Nạp Toàn bộ Dữ liệu (`benchmark_full_corpus.py`)

Script [benchmark_full_corpus.py](file:///d:/New/milvus/benchmark_full_corpus.py) giải quyết bài toán: **Hiệu năng của Milvus (Recall@K, Latency, QPS và RAM/Disk) sẽ như thế nào khi nạp đủ 100% toàn bộ corpus?**

### Cách chạy:

```bash
# 1. Dự báo ngay hiệu năng khi nạp đủ 61.425 văn bản Zalo Legal:
python benchmark_full_corpus.py --dataset zalo

# 2. Dự báo ngay hiệu năng khi nạp đủ 2.68 triệu văn bản NQ-VN:
python benchmark_full_corpus.py --dataset nq

# 3. Tự động nạp nốt 100% dữ liệu vào Milvus và đo lường trực tiếp toàn diện:
python benchmark_full_corpus.py --dataset zalo --auto-index-full

# 4. Xuất kết quả chi tiết ra file JSON:
python benchmark_full_corpus.py --dataset zalo --output-json full_corpus_report.json
```

---

## ⚡ 2. Benchmark Siêu Tốc Bằng Smart Pool (`benchmark_smart_pool.py`) - KHUYÊN DÙNG

Phương pháp tối ưu cho kho dữ liệu khổng lồ (như NQ-VN 2.68 triệu bài):
- Tự động gom **100% tài liệu đáp án trong `qrels`** (~2.283 docs cho NQ, ~793 docs cho Zalo).
- Bổ sung tài liệu gây nhiễu (Distractors) ngẫu nhiên (ví dụ 5.000 - 15.000 docs).
- Đánh giá **100% câu hỏi** (1.900 queries NQ, 818 queries Zalo) đối chiếu với đáp án chuẩn.
- Thời gian chạy: **chỉ mất 2 ~ 4 phút** thay vì chờ 12 tiếng!

```bash
# 1. Benchmark NQ-VN với 100% câu hỏi (Pool 5.000 docs - Chạy cực nhanh ~2 phút):
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --recreate

# 2. Benchmark NQ-VN với Pool 15.000 docs (Tăng độ khó / nhiều nhiễu hơn):
python benchmark_smart_pool.py --dataset nq --pool-size 15000 --recreate

# 3. Benchmark Zalo Legal bằng Smart Pool:
python benchmark_smart_pool.py --dataset zalo --pool-size 5000 --recreate

# 4. Xuất kết quả ra file JSON:
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --output-json nq_smart_report.json
```

---

## 📊 3. Đánh giá Hiệu năng Hiện tại từ Qrels (`evaluate_milvus.py`)

Script [evaluate_milvus.py](file:///d:/New/milvus/evaluate_milvus.py) tự động đối sánh câu hỏi với nhãn chuẩn **`qrels`** trên số tài liệu hiện có trong Milvus:

```bash
# Đánh giá trên tập Pháp luật (Zalo Legal)
python evaluate_milvus.py --dataset zalo --in-corpus-only

# Đánh giá trên tập Tri thức mở (NQ-VN)
python evaluate_milvus.py --dataset nq --in-corpus-only
```

---

## 🚀 3. Lập chỉ mục dữ liệu vào Milvus (`index_milvus.py`)

Mô hình embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, Cosine metric).

```bash
# Nạp cả 2 bộ dữ liệu cùng lúc
python index_milvus.py --dataset both --limit-zalo 5000 --limit-nq 5000

# Chỉ nạp văn bản Pháp luật (Zalo Legal)
python index_milvus.py --dataset zalo --limit-zalo 5000

# Chỉ nạp Tri thức mở (NQ-VN)
python index_milvus.py --dataset nq --limit-nq 10000

# Nạp lại từ đầu (xóa collection cũ và tạo lại)
python index_milvus.py --dataset nq --limit-nq 5000 --recreate
```

---

## 🔍 4. Tìm kiếm qua dòng lệnh (`search_milvus.py`)

```bash
# Tìm kiếm đồng thời trên cả 2 bộ dữ liệu (Mặc định)
python search_milvus.py --query "Bài hát Không thể không yêu" --dataset both --top-k 3

# Chỉ tìm trong văn bản Pháp luật
python search_milvus.py --query "Quy định xử phạt vi phạm nồng độ cồn" --dataset zalo --top-k 5

# Chế độ tương tác liên tục (REPL)
python search_milvus.py --interactive
```

---

## 🌐 5. Giao diện Web Trực quan (`app.py`)

Khởi động giao diện Web tra cứu song song:

```bash
python app.py
```

Truy cập: **[http://localhost:5000](http://localhost:5000)** trên trình duyệt.
