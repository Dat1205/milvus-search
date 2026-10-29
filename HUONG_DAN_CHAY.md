# HƯỚNG DẪN CHẠY TOÀN BỘ DỰ ÁN MILVUS (STEP-BY-STEP)

Tài liệu này tổng hợp toàn bộ các bước từ **cài đặt môi trường**, **tải dữ liệu**, **nạp vector vào Milvus**, **tìm kiếm (CLI & Web UI)** cho đến **đánh giá Benchmark (Recall@K, Latency, QPS)**.

---

## 📑 Mục Lục Nhanh
1. [Cài đặt môi trường & Thư viện](#1-cài-đặt-môi-trường--thư-viện)
2. [Tải bộ dữ liệu gốc (`download_data.py`)](#2-tải-bộ-dữ-liệu-gốc-download_datapy)
3. [Nạp dữ liệu vào Milvus (`index_milvus.py`)](#3-nạp-dữ-liệu-vào-milvus-index_milvuspy)
4. [Tìm kiếm ngữ nghĩa qua dòng lệnh (`search_milvus.py`)](#4-tìm-kiếm-ngữ-nghĩa-qua-dòng-lệnh-search_milvuspy)
5. [Khởi động giao diện Web trực quan (`app.py`)](#5-khởi-động-giao-diện-web-trực-quan-apppy)
6. [Đánh giá hiệu năng & Benchmark chuẩn Qrels](#6-đánh-giá-hiệu-năng--benchmark-chuẩn-qrels)
7. [Chạy Milvus Server với Docker Compose (Tùy chọn)](#7-chạy-milvus-server-với-docker-compose-tùy-chọn)
8. [Bảng tra cứu nhanh lệnh (Cheat Sheet)](#8-bảng-tra-cứu-nhanh-lệnh-cheat-sheet)

---

## 1. Cài đặt môi trường & Thư viện

Mở terminal (PowerShell hoặc Command Prompt) tại thư mục dự án `d:\New\milvus`:

```bash
# Cài đặt các thư viện cần thiết
pip install pymilvus milvus-lite datasets sentence-transformers flask tqdm numpy
```

> **Ghi chú về phần cứng:**
> * Dự án mặc định chạy trên **Milvus Lite** (cơ sở dữ liệu vector nhúng cục bộ, lưu thành file `milvus_data.db`, không cần cài Docker hay dịch vụ ngoài).
> * Mô hình embedding sử dụng: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (vector 384 chiều, tối ưu cho tiếng Việt).

---

## 2. Tải bộ dữ liệu gốc (`download_data.py`)

Nếu muốn tải hoặc nạp dữ liệu từ Hugging Face về bộ nhớ đệm (cache):

```bash
python download_data.py
```

* **Dữ liệu được lưu tại:** `C:\Users\<username>\.cache\huggingface\hub\`
* **2 bộ dữ liệu gồm:**
  1. `GreenNode/zalo-ai-legal-text-retrieval-vn`: 61.425 điều khoản pháp luật Việt Nam.
  2. `GreenNode/nq-vn`: 2.681.468 đoạn trích bách khoa tri thức dịch từ Google Natural Questions.

---

## 3. Nạp dữ liệu vào Milvus (`index_milvus.py`)

Script này sẽ tính toán vector embedding và nạp vào các collection trong `milvus_data.db`.

### A. Nạp dữ liệu Pháp luật Việt Nam (Zalo Legal)
```bash
# Nạp mẫu 5.000 điều khoản (mất ~1 phút - khuyến nghị để thử nghiệm nhanh):
python index_milvus.py --dataset zalo --limit-zalo 5000 --recreate

# Nạp toàn bộ 61.425 điều khoản Zalo Legal (mất ~12-15 phút):
python index_milvus.py --dataset zalo --recreate
```

### B. Nạp dữ liệu Tri thức Bách khoa (NQ-VN)
```bash
# Nạp 5.000 đoạn trích NQ-VN:
python index_milvus.py --dataset nq --limit-nq 5000 --recreate

# Nạp 20.000 đoạn trích NQ-VN:
python index_milvus.py --dataset nq --limit-nq 20000 --recreate
```

### C. Nạp cả 2 bộ dữ liệu cùng lúc
```bash
python index_milvus.py --dataset both --limit-zalo 5000 --limit-nq 5000 --recreate
```

---

## 4. Tìm kiếm ngữ nghĩa qua dòng lệnh (`search_milvus.py`)

### A. Tìm kiếm một câu hỏi trực tiếp:
```bash
# Tìm kiếm đồng thời trên cả 2 kho dữ liệu:
python search_milvus.py --query "Quy định về bảo vệ đê điều trong mùa lũ" --dataset both --top-k 3

# Tìm bài hát / tri thức trên NQ-VN (hiển thị cả bản dịch và tiếng Anh gốc):
python search_milvus.py --query "Bài hát Không thể không yêu" --dataset nq --top-k 3

# Chỉ tìm trong văn bản Pháp luật:
python search_milvus.py --query "Mức phạt quay đầu xe trên cao tốc" --dataset zalo --top-k 5
```

### B. Chế độ dòng lệnh tương tác liên tục (REPL):
```bash
python search_milvus.py --interactive
```
* **Cách dùng trong chế độ tương tác:**
  * Gõ câu hỏi tiếng Việt bất kỳ và nhấn `Enter`.
  * Gõ `:mode all` để tìm cả 2 bộ dữ liệu.
  * Gõ `:mode zalo` để chỉ tìm văn bản pháp luật.
  * Gõ `:mode nq` để chỉ tìm tri thức bách khoa.
  * Gõ `exit` hoặc `quit` để thoát.

---

## 5. Khởi động giao diện Web trực quan (`app.py`)

Khởi chạy ứng dụng Web Flask:

```bash
python app.py
```

* **Địa chỉ truy cập trình duyệt:** **`http://localhost:5000`** (hoặc `http://127.0.0.1:5000`)
* **Tính năng giao diện:**
  * Có sẵn thanh tìm kiếm và 3 tab lọc: **Tất cả (Song song)**, **Pháp luật (Zalo Legal)**, **Bách khoa Tri thức (NQ-VN)**.
  * Thẻ kết quả phân màu rõ ràng (màu xanh dương cho Pháp luật, màu tím cho Tri thức NQ kèm bản gốc tiếng Anh).
  * Đo lường thời gian trích xuất vector và thời gian tìm kiếm Milvus chính xác theo mili-giây.

---

## 6. Đánh giá hiệu năng & Benchmark chuẩn Qrels

Hệ thống cung cấp các phương pháp đo lường **Recall@K**, **Độ trễ Latency (P50/P90/P95/P99)** và **Thông lượng Throughput (QPS)** đối chiếu trực tiếp với tập nhãn chuẩn `qrels`:

### ⚡ Cách 1: Benchmark Siêu Tốc bằng Smart Pool (Khuyên dùng nhất - Chỉ mất 2 ~ 3 phút)
> Áp dụng kỹ thuật chuẩn Information Retrieval: Gom **100% tài liệu đáp án trong `qrels`** + kèm tập tài liệu gây nhiễu (distractors) $\rightarrow$ Đánh giá chính xác **100% câu hỏi** mà không cần chờ 12 tiếng.

```bash
# 1. Benchmark NQ-VN với 100% câu hỏi (1.900 queries) trên Pool 5.000 docs (~2 phút):
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --recreate

# 2. Benchmark NQ-VN với Pool 15.000 docs (tăng độ khó):
python benchmark_smart_pool.py --dataset nq --pool-size 15000 --recreate

# 3. Benchmark Zalo Legal (818 queries) trên Pool 5.000 docs:
python benchmark_smart_pool.py --dataset zalo --pool-size 5000 --recreate

# 4. Xuất toàn bộ kết quả ra file JSON:
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --output-json nq_report.json
```

---

### 📊 Cách 2: Đo lường trên dữ liệu hiện có trong Milvus (`evaluate_milvus.py`)
```bash
# Đánh giá các câu hỏi có tài liệu nằm trong kho dữ liệu hiện tại:
python evaluate_milvus.py --dataset zalo --in-corpus-only

python evaluate_milvus.py --dataset nq --in-corpus-only
```

---

### 📈 Cách 3: Tính toán & Dự báo khi nạp 100% Corpus (`benchmark_full_corpus.py`)
```bash
# Dự báo lý thuyết O(log N) khi nạp đủ 61.425 docs Zalo:
python benchmark_full_corpus.py --dataset zalo

# Tự động nạp nốt 100% dữ liệu Zalo vào Milvus rồi đo lường thực tế:
python benchmark_full_corpus.py --dataset zalo --auto-index-full
```

---

## 7. Chạy Milvus Server với Docker Compose (Tùy chọn)

Nếu bạn muốn chuyển từ **Milvus Lite** sang **Milvus Server Standalone** có giao diện quản trị đồ họa **Attu Web GUI**:

1. **Khởi chạy container:**
   ```bash
   docker compose up -d
   ```
2. **Truy cập dịch vụ:**
   * **Giao diện quản trị Attu Web GUI:** Mở trình duyệt vào **`http://localhost:8000`**
   * **MinIO Console:** `http://localhost:9001` (User/Pass: `minioadmin` / `minioadmin`)
   * **Milvus Service:** `localhost:19530`
3. **Kết nối mã nguồn tới Docker:**
   Thêm cờ `--db-path http://localhost:19530` vào bất kỳ lệnh nào:
   ```bash
   python search_milvus.py --db-path http://localhost:19530 --query "Luật giao thông"
   ```
4. **Dừng container khi không dùng:**
   ```bash
   docker compose down
   ```

---

## 8. Bảng tra cứu nhanh lệnh (Cheat Sheet)

| Nhu cầu công việc | Lệnh thực thi nhanh |
| :--- | :--- |
| **Nạp dữ liệu Zalo (5.000 docs)** | `python index_milvus.py --dataset zalo --limit-zalo 5000 --recreate` |
| **Nạp dữ liệu NQ-VN (5.000 docs)** | `python index_milvus.py --dataset nq --limit-nq 5000 --recreate` |
| **Tìm kiếm 1 câu bất kỳ** | `python search_milvus.py --query "Nội dung cần tìm" --dataset both` |
| **Tìm kiếm tương tác CLI** | `python search_milvus.py --interactive` |
| **Mở giao diện Web** | `python app.py` $\rightarrow$ Mở `http://localhost:5000` |
| **Benchmark NQ-VN siêu tốc** | `python benchmark_smart_pool.py --dataset nq --pool-size 5000 --recreate` |
| **Benchmark Zalo siêu tốc** | `python benchmark_smart_pool.py --dataset zalo --pool-size 5000 --recreate` |
| **Xem số lượng docs trong Milvus** | `python -c "from pymilvus import MilvusClient; c=MilvusClient('milvus_data.db'); print([(x, c.get_collection_stats(x)) for x in c.list_collections()]); c.close()"` |

