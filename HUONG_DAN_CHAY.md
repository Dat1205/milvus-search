# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN TRÊN MÁY MỚI (A - Z)

> **Tài liệu tổng hợp chuẩn mực:** Hướng dẫn thiết lập môi trường từ con số 0, cài đặt thư viện, nạp dữ liệu, tìm kiếm ngữ nghĩa (CLI & Web) và đánh giá hiệu năng (Benchmark) trên cả **CPU** và **GPU NVIDIA**.

---

## 📑 Mục Lục
1. [Yêu cầu hệ thống (Prerequisites)](#1-yêu-cầu-hệ-thống-prerequisites)
2. [Cài đặt môi trường & Thư viện](#2-cài-đặt-môi-trường--thư-viện)
   - [A. Cài đặt các thư viện cơ bản (`requirements.txt`)](#a-cài-đặt-các-thư-viện-cơ-bản-requirementstxt)
   - [B. Cấu hình GPU NVIDIA CUDA (Tùy chọn - Tăng tốc 15x - 20x)](#b-cấu-hình-gpu-nvidia-cuda-tùy-chọn---tăng-tốc-15x---20x)
3. [Tải dữ liệu từ Hugging Face](#3-tải-dữ-liệu-từ-hugging-face)
4. [Nạp dữ liệu vào cơ sở dữ liệu vector (`index_milvus.py`)](#4-nạp-dữ-liệu-vào-cơ-sở-dữ-liệu-vector-index_milvuspy)
5. [Tìm kiếm ngữ nghĩa (Semantic Search)](#5-tìm-kiếm-ngữ-nghĩa-semantic-search)
   - [A. Tìm kiếm nhanh 1 câu (CLI)](#a-tìm-kiếm-nhanh-1-câu-cli)
   - [B. Chế độ tương tác liên tục (Interactive REPL)](#b-chế-độ-tương-tác-liên-tục-interactive-repl)
   - [C. Khởi chạy Giao diện Web trực quan (`app.py`)](#c-khởi-chạy-giao-diện-web-trực-quan-apppy)
6. [Đánh giá hiệu năng & Benchmark chuẩn Qrels](#6-đánh-giá-hiệu-năng--benchmark-chuẩn-qrels)
   - [Cách 1: Benchmark siêu tốc bằng Smart Pool (Khuyên dùng)](#cách-1-benchmark-siêu-tốc-bằng-smart-pool-khuyên-dùng)
   - [Cách 2: Đánh giá trên dữ liệu hiện có (`evaluate_milvus.py`)](#cách-2-đánh-giá-trên-dữ-liệu-hiện-có-evaluate_milvuspy)
   - [Cách 3: Tính toán & Dự báo quy mô toàn bộ Corpus (`benchmark_full_corpus.py`)](#cách-3-tính-toán--dự-báo-quy-mô-toàn-bộ-corpus-benchmark_full_corpuspy)
7. [Triển khai Milvus Standalone Server với Docker (Tùy chọn)](#7-triển-khai-milvus-standalone-server-với-docker-tùy-chọn)
8. [Bảng tra cứu nhanh lệnh (Cheat Sheet)](#8-bảng-tra-cứu-nhanh-lệnh-cheat-sheet)
9. [Xử lý sự cố thường gặp (Troubleshooting)](#9-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. Yêu cầu hệ thống (Prerequisites)

* **Hệ điều hành:** Windows 10/11, Ubuntu/Debian Linux, hoặc macOS.
* **Python:** Phiên bản **3.10 trở lên** (khuyến nghị Python 3.11, 3.12, hoặc 3.14).
* **Git:** Dùng để clone mã nguồn dự án.
* **Phần cứng:**
  * **RAM tối thiểu:** 8 GB RAM (khuyến nghị 16 GB).
  * **Ổ cứng:** Còn trống tối thiểu 5 - 10 GB.
  * **Card đồ họa (GPU):** Card rời NVIDIA (GTX 1650, RTX 2050/3050/4060/5060, v.v.) nếu muốn tăng tốc độ tính vector. Nếu không có card rời, hệ thống vẫn chạy ổn định trên CPU.

---

## 2. Cài đặt môi trường & Thư viện

Mở terminal (PowerShell trên Windows hoặc Terminal trên Linux/Mac) tại thư mục dự án:

### A. Cài đặt các thư viện cơ bản (`requirements.txt`)

```bash
# Khởi tạo môi trường ảo (Khuyến khích)
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source venv/bin/activate

# Cài đặt toàn bộ dependencies:
pip install -r requirements.txt
```

---

### B. Cấu hình GPU NVIDIA CUDA (Tùy chọn - Tăng tốc 15x - 20x)

Dự án có module tự động nhận diện phần cứng [`device_utils.py`](file:///d:/New/milvus/device_utils.py). Khi có GPU CUDA, tốc độ tính embedding sẽ tăng vọt từ ~60 docs/s (CPU) lên **800 - 1.200 docs/s (GPU)**.

Cài đặt phiên bản PyTorch phù hợp với card đồ họa của bạn:

* **Đối với dòng card mới NVIDIA RTX 50-series (RTX 5060, 5070, 5080, 5090) hoặc máy có CUDA 13.x:**
  ```bash
  pip install torch==2.14.1+cu132 --index-url https://download.pytorch.org/whl/cu132
  ```

* **Đối với các dòng card NVIDIA RTX 20/30/40-series hoặc GTX (CUDA 12.x/11.x phổ biến):**
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```

* **Kiểm tra trạng thái kích hoạt GPU:**
  ```bash
  python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  ```
  *(Nếu hiện `CUDA Ready: True` và tên GPU của bạn là đã sẵn sàng)*.

---

## 3. Tải dữ liệu từ Hugging Face

Dự án sử dụng 2 bộ dữ liệu chuẩn tiếng Việt từ GreenNode:
1. `GreenNode/zalo-ai-legal-text-retrieval-vn`: 61.425 điều khoản pháp luật Việt Nam.
2. `GreenNode/nq-vn`: 2.681.468 đoạn văn bản bách khoa tri thức.

Để tải trước toàn bộ dữ liệu vào bộ nhớ đệm (cache):
```bash
python download_data.py
```
*(Lưu ý: Nếu không chạy lệnh này, các script nạp dữ liệu bên dưới vẫn sẽ tự động tải khi cần).*

---

## 4. Nạp dữ liệu vào cơ sở dữ liệu vector (`index_milvus.py`)

Dữ liệu vector được lưu vào cơ sở dữ liệu nhúng nhẹ **`milvus_data.db`** (Milvus Lite - không cần cài đặt server hay Docker).

### A. Nạp dữ liệu Pháp luật Việt Nam (Zalo Legal)
```bash
# Nạp mẫu 5.000 điều khoản (mất ~1 phút trên CPU, ~15 giây trên GPU):
python index_milvus.py --dataset zalo --limit-zalo 5000 --recreate

# Tăng tốc tối đa với GPU và FP16:
python index_milvus.py --dataset zalo --limit-zalo 5000 --device cuda --fp16 --recreate

# Nạp toàn bộ 61.425 điều khoản Zalo Legal:
python index_milvus.py --dataset zalo --recreate --device cuda --fp16
```

### B. Nạp dữ liệu Tri thức Bách khoa (NQ-VN)
```bash
# Nạp 5.000 đoạn trích NQ-VN (có trường song ngữ title/og_title):
python index_milvus.py --dataset nq --limit-nq 5000 --device cuda --fp16 --recreate
```

### C. Nạp đồng thời cả 2 bộ dữ liệu:
```bash
python index_milvus.py --dataset both --limit 5000 --device cuda --fp16 --recreate
```

---

## 5. Tìm kiếm ngữ nghĩa (Semantic Search)

### A. Tìm kiếm nhanh 1 câu (CLI)
```bash
# Tìm kiếm trên cả 2 bộ dữ liệu (tự động phát hiện GPU/CPU):
python search_milvus.py --query "Quy định về thời hiệu xử phạt vi phạm hành chính" --dataset both

# Chỉ tìm trong bộ dữ liệu Pháp luật (Zalo Legal):
python search_milvus.py --query "Điều kiện cấp giấy phép lái xe ô tô" --dataset zalo --top-k 3

# Chỉ tìm trong bộ dữ liệu Bách khoa (NQ-VN):
python search_milvus.py --query "Ai là người đầu tiên đặt chân lên mặt trăng?" --dataset nq --top-k 3
```

### B. Chế độ tương tác liên tục (Interactive REPL)
```bash
python search_milvus.py --interactive
```
* Trong cửa sổ tương tác:
  * Nhập câu hỏi và nhấn `Enter` để xem kết quả ngay lập tức.
  * Gõ `:zalo` để chuyển mục tiêu sang Pháp luật.
  * Gõ `:nq` để chuyển sang Bách khoa tri thức.
  * Gõ `:both` để tìm kiếm song song cả hai.
  * Gõ `:k 10` để đổi số lượng top kết quả thành 10.
  * Gõ `quit` hoặc `exit` để thoát.

### C. Khởi chạy Giao diện Web trực quan (`app.py`)
```bash
python app.py
```
* Mở trình duyệt web truy cập vào: **`http://localhost:5000`**
* **Tính năng:**
  * Giao diện hiện đại (Tailwind CSS, Dark/Light mode).
  * Tab chuyển đổi linh hoạt: *Tất cả*, *Pháp luật Việt Nam*, hoặc *Bách khoa tri thức NQ-VN*.
  * Hiển thị điểm tương đồng Cosine, độ trễ thời gian thực (Embedding + Milvus Search).
  * Hiển thị song ngữ (Tiêu đề dịch & Tiêu đề gốc tiếng Anh cho NQ-VN).

---

## 6. Đánh giá hiệu năng & Benchmark chuẩn Qrels

Hệ thống đánh giá độ chính xác theo chuẩn Information Retrieval (IR) quốc tế, đo lường các chỉ số:
* **Recall@K & Hit@K** ($K \in [1, 3, 5, 10, 20, 50]$): Tỷ lệ truy xuất đúng tài liệu đáp án trong top K kết quả.
* **MRR@K (Mean Reciprocal Rank):** Đánh giá vị trí xếp hạng của tài liệu đúng.
* **Latency Percentiles:** Phân tích chi tiết thời gian nhúng (Embedding) và thời gian tìm kiếm vector (Search) ở các mốc P50, P90, P95, P99.
* **Throughput (QPS):** Đo số lượng truy vấn hệ thống xử lý được trên mỗi giây (Single-stream & Batched QPS).

---

### Cách 1: Benchmark siêu tốc bằng Smart Pool (Khuyên dùng)

> **Giải pháp tối ưu thời gian:** Thay vì phải chờ nạp 2.68 triệu tài liệu NQ-VN mất 12 tiếng, phương pháp **Smart Pool** tự động trích xuất **100% tài liệu đáp án có trong `qrels`**, kết hợp với tài liệu gây nhiễu (distractors), cho phép đánh giá **100% câu hỏi** chỉ trong **2 - 4 phút**.

```bash
# 1. Benchmark NQ-VN với 100% câu hỏi (1.900 queries) trên GPU:
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --device cuda --fp16 --recreate

# 2. Benchmark Zalo Legal với 100% câu hỏi (818 queries) trên GPU:
python benchmark_smart_pool.py --dataset zalo --pool-size 5000 --device cuda --fp16 --recreate

# 3. Xuất kết quả ra file JSON để làm báo cáo:
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --output-json benchmark_nq_results.json
```

---

### Cách 2: Đánh giá trên dữ liệu hiện có (`evaluate_milvus.py`)

Đo lường trực tiếp trên collection hiện có trong `milvus_data.db`:
```bash
# Đánh giá 100 câu hỏi trên bộ Zalo:
python evaluate_milvus.py --dataset zalo --limit-queries 100

# Chỉ đánh giá các câu hỏi có đáp án nằm trong dữ liệu đã nạp (In-Corpus):
python evaluate_milvus.py --dataset zalo --in-corpus-only
```

---

### Cách 3: Tính toán & Dự báo quy mô toàn bộ Corpus (`benchmark_full_corpus.py`)

Dựa trên mô hình định lượng độ phức tạp thuật toán đồ thị HNSW $\mathcal{O}(\log N)$, dự phóng chính xác tài nguyên RAM, Disk và QPS khi tăng quy mô từ 5.000 docs $\rightarrow$ 61.425 docs $\rightarrow$ 2.680.000 docs:

```bash
# Dự báo lý thuyết quy mô cho Zalo Legal:
python benchmark_full_corpus.py --dataset zalo

# Dự báo lý thuyết quy mô cho NQ-VN (2.68 triệu docs):
python benchmark_full_corpus.py --dataset nq

# Tự động nạp nốt 100% văn bản Zalo Legal còn thiếu rồi đo kiểm thực tế:
python benchmark_full_corpus.py --dataset zalo --auto-index-full --device cuda --fp16
```

---

## 7. Triển khai Milvus Standalone Server với Docker (Tùy chọn)

Nếu bạn muốn chuyển từ **Milvus Lite** (file cục bộ) sang **Milvus Server Standalone** chuyên nghiệp có giao diện đồ họa **Attu Web GUI**:

1. **Khởi động cụm Milvus bằng Docker Compose:**
   ```bash
   docker compose up -d
   ```
2. **Truy cập các dịch vụ:**
   * **Attu Web GUI (Giao diện đồ họa Milvus):** Mở trình duyệt vào **`http://localhost:8000`**
   * **MinIO Storage Console:** Mở `http://localhost:9001` (Tài khoản/Mật khẩu: `minioadmin` / `minioadmin`)
   * **Milvus Port gRPC:** `localhost:19530`

3. **Kết nối mã nguồn tới Docker Milvus:**
   Chỉ cần truyền tham số `--db-path http://localhost:19530` vào bất kỳ lệnh nào:
   ```bash
   python search_milvus.py --db-path http://localhost:19530 --query "Luật an ninh mạng"
   ```

4. **Dừng cụm container khi hoàn thành:**
   ```bash
   docker compose down
   ```

---

## 8. Bảng tra cứu nhanh lệnh (Cheat Sheet)

| Nhu cầu tác vụ | Câu lệnh thực thi |
| :--- | :--- |
| **Cài đặt thư viện** | `pip install -r requirements.txt` |
| **Kiểm tra GPU CUDA** | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"` |
| **Nạp dữ liệu Zalo (GPU)** | `python index_milvus.py --dataset zalo --limit-zalo 5000 --device cuda --fp16 --recreate` |
| **Nạp dữ liệu NQ-VN (GPU)** | `python index_milvus.py --dataset nq --limit-nq 5000 --device cuda --fp16 --recreate` |
| **Tìm kiếm 1 câu (CLI)** | `python search_milvus.py --query "Nội dung cần tìm" --dataset both` |
| **Mở tìm kiếm tương tác** | `python search_milvus.py --interactive` |
| **Khởi chạy Web UI** | `python app.py` $\rightarrow$ Mở `http://localhost:5000` |
| **Benchmark NQ-VN siêu tốc** | `python benchmark_smart_pool.py --dataset nq --pool-size 5000 --device cuda --fp16 --recreate` |
| **Benchmark Zalo siêu tốc** | `python benchmark_smart_pool.py --dataset zalo --pool-size 5000 --device cuda --fp16 --recreate` |
| **Kiểm tra các collection trong DB** | `python -c "from pymilvus import MilvusClient; c=MilvusClient('milvus_data.db'); print([(x, c.get_collection_stats(x)) for x in c.list_collections()]); c.close()"` |

---

## 9. Xử lý sự cố thường gặp (Troubleshooting)

1. **Lỗi `Collection is in state released; call load() before search`:**
   * *Nguyên nhân:* Collection chưa được nạp vào bộ nhớ trước khi truy vấn.
   * *Khắc phục:* Toàn bộ script trong dự án đã tự động gọi `client.load_collection(name)`. Nếu gặp lỗi này, hãy chạy lại lệnh hoặc kiểm tra tên collection.
2. **Lỗi `UnicodeEncodeError: 'charmap' codec can't encode...` trên Windows:**
   * *Nguyên nhân:* Cửa sổ PowerShell hoặc Command Prompt mặc định dùng bảng mã cp1252.
   * *Khắc phục:* Gõ lệnh sau trong PowerShell trước khi chạy:
     ```powershell
     $env:PYTHONIOENCODING = "utf-8"
     [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
     ```
3. **GPU không nhận diện hoặc báo lỗi thiếu Kernel:**
   * *Nguyên nhân:* Bản PyTorch CUDA không khớp với Driver GPU (đặc biệt card RTX 50-series cần CUDA 13.x).
   * *Khắc phục:* Cài đặt đúng gói `pip install torch==2.14.1+cu132 --index-url https://download.pytorch.org/whl/cu132`. Ngoài ra, module `device_utils.py` sẽ tự động chuyển về chạy CPU an toàn nếu phát hiện lỗi driver.
