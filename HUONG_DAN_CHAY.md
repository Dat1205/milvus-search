# HƯỚNG DẪN CÀI ĐẶT, VẬN HÀNH & GIẢI THÍCH CHI TIẾT MÃ NGUỒN (A - Z)

> **Tài liệu toàn diện dự án Milvus:** Tổng hợp hướng dẫn cài đặt môi trường từ con số 0 trên máy mới, vận hành tìm kiếm ngữ nghĩa, phương pháp benchmark 100% full dữ liệu và **giải thích chi tiết cấu trúc, logic toán học của từng file mã nguồn**.

---

## 📑 Mục Lục
1. [Yêu cầu hệ thống (Prerequisites)](#1-yêu-cầu-hệ-thống-prerequisites)
2. [Hướng dẫn cài đặt trên máy mới](#2-hướng-dẫn-cài-đặt-trên-máy-mới)
   - [Bước 1: Khởi tạo môi trường ảo Python](#bước-1-khởi-tạo-môi-trường-ảo-python)
   - [Bước 2: Cài đặt thư viện phụ thuộc (`requirements.txt`)](#bước-2-cài-đặt-thư-viện-phụ-thuộc-requirementstxt)
   - [Bước 3: Cấu hình PyTorch tăng tốc GPU CUDA](#bước-3-cấu-hình-pytorch-tăng-tốc-gpu-cuda)
3. [Tải dữ liệu từ Hugging Face](#3-tải-dữ-liệu-từ-hugging-face)
4. [Nạp dữ liệu vào Vector Database (`index_milvus.py`)](#4-nạp-dữ-liệu-vào-vector-database-index_milvuspy)
5. [Tìm kiếm ngữ nghĩa (Semantic Search)](#5-tìm-kiếm-ngữ-nghĩa-semantic-search)
   - [Tìm kiếm dòng lệnh CLI (`search_milvus.py`)](#tìm-kiếm-dòng-lệnh-cli-search_milvuspy)
   - [Giao diện Web trực quan (`app.py`)](#giao-diện-web-trực-quan-apppy)
6. [Đánh giá hiệu năng & Benchmark chuẩn Qrels](#6-đánh-giá-hiệu-năng--benchmark-chuẩn-qrels)
   - [A. Benchmark 100% Full Corpus, Queries & Qrels (`benchmark_full.py`)](#a-benchmark-100-full-corpus-queries--qrels-benchmark_fullpy)
   - [B. Benchmark nhanh cả 2 bộ dữ liệu (`benchmark_all.py`)](#b-benchmark-nhanh-cả-2-bộ-dữ-liệu-benchmark_allpy)
   - [C. Benchmark siêu tốc bằng Smart Pool (`benchmark_smart_pool.py`)](#c-benchmark-siêu-tốc-bằng-smart-pool-benchmark_smart_poolpy)
7. [Giải thích chi tiết cấu trúc & Mã nguồn hệ thống](#7-giải-thích-chi-tiết-cấu-trúc--mã-nguồn-hệ-thống)
   - [1. `device_utils.py` - Quản lý phần cứng & Tăng tốc GPU](#1-device_utilspy---quản-lý-phần-cứng--tăng-tốc-gpu)
   - [2. `index_milvus.py` - Xây dựng Schema & Trích xuất Embeddings](#2-index_milvuspy---xây-dựng-schema--trích-xuất-embeddings)
   - [3. `search_milvus.py` - Cơ chế tìm kiếm & Xếp hạng Cosine](#3-search_milvuspy---cơ-chế-tìm-kiếm--xếp-hạng-cosine)
   - [4. `benchmark_full.py` - Đánh giá Full Corpus & Xuất JSON](#4-benchmark_fullpy---đánh-giá-full-corpus--xuất-json)
   - [5. `benchmark_smart_pool.py` - Kỹ thuật Smart Pool Guarantee](#5-benchmark_smart_poolpy---kỹ-thuật-smart-pool-guarantee)
   - [6. `app.py` - Kiến trúc Web REST API & Giao diện](#6-apppy---kiến-trúc-web-rest-api--giao-diện)
8. [Giải thích các chỉ số đánh giá Information Retrieval (IR)](#8-giải-thích-các-chỉ-số-đánh-giá-information-retrieval-ir)
9. [Bảng tra cứu nhanh lệnh (Cheat Sheet) & Xử lý lỗi](#9-bảng-tra-cứu-nhanh-lệnh-cheat-sheet--xử-lý-lỗi)

---

## 1. Yêu cầu hệ thống (Prerequisites)

* **Hệ điều hành:** Windows 10/11 (64-bit), Ubuntu/Debian Linux, hoặc macOS.
* **Python:** Phiên bản **3.10 trở lên** (Đã kiểm thử tối ưu trên Python 3.11, 3.12 và 3.14).
* **Phần cứng:**
  * **RAM tối thiểu:** 8 GB RAM (khuyến nghị 16 GB để nạp cache mượt mà).
  * **Ổ cứng:** Còn trống tối thiểu 5 – 10 GB cho database `milvus_data.db`.
  * **GPU (Tùy chọn):** Card NVIDIA (RTX 20/30/40/50 series hoặc GTX) có hỗ trợ CUDA. Nếu không có card rời, code tự động chạy trên CPU.

---

## 2. Hướng dẫn cài đặt trên máy mới

### Bước 1: Khởi tạo môi trường ảo Python
Mở Terminal / PowerShell tại thư mục dự án `d:\New\milvus`:

```bash
# Tạo môi trường ảo (Virtual Environment)
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source venv/bin/activate
```

### Bước 2: Cài đặt thư viện phụ thuộc (`requirements.txt`)
```bash
pip install -r requirements.txt
```

### Bước 3: Cấu hình PyTorch tăng tốc GPU CUDA
Nếu máy tính có card đồ họa NVIDIA, cài bản dựng PyTorch CUDA để tăng tốc tính embedding gấp **15x – 20x** (đạt 800 – 1.200 docs/s):

* **Dòng card NVIDIA RTX 50-series (RTX 5060, 5070, 5080, 5090) hoặc máy dùng CUDA 13.x:**
  ```bash
  pip install torch==2.14.1+cu132 --index-url https://download.pytorch.org/whl/cu132
  ```
* **Dòng card NVIDIA RTX 20/30/40-series hoặc GTX (CUDA 12.x/11.x phổ biến):**
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
* **Kiểm tra trạng thái nhận diện GPU:**
  ```bash
  python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
  ```
  *(Nếu hiện `CUDA Ready: True` kèm tên card đồ họa của bạn là thành công)*.

---

## 3. Tải dữ liệu từ Hugging Face

Dự án sử dụng 2 bộ dữ liệu tiếng Việt chuẩn của GreenNode:
1. `GreenNode/zalo-ai-legal-text-retrieval-vn`: 61.425 điều khoản pháp luật Việt Nam.
2. `GreenNode/nq-vn`: 2.681.468 đoạn trích bách khoa tri thức (bản dịch và gốc tiếng Anh từ Google Natural Questions).

Để tải trước toàn bộ dữ liệu vào cache máy cục bộ:
```bash
python download_data.py
```
*(Lưu ý: Nếu không chạy lệnh này, các script nạp dữ liệu bên dưới vẫn tự động tải dữ liệu khi cần)*.

---

## 4. Nạp dữ liệu vào Vector Database (`index_milvus.py`)

Dữ liệu vector được lưu vào cơ sở dữ liệu nhúng **`milvus_data.db`** (Milvus Lite).

```bash
# Nạp 5.000 điều khoản Zalo Legal (Chạy thử nghiệm nhanh trong 15s trên GPU):
python index_milvus.py --dataset zalo --limit-zalo 5000 --device cuda --fp16 --recreate

# Nạp TOÀN BỘ 61.425 điều khoản Zalo Legal vào Milvus:
python index_milvus.py --dataset zalo --device cuda --fp16 --recreate

# Nạp 5.000 bài viết NQ-VN (song ngữ title + og_title):
python index_milvus.py --dataset nq --limit-nq 5000 --device cuda --fp16 --recreate
```

---

## 5. Tìm kiếm ngữ nghĩa (Semantic Search)

### Tìm kiếm dòng lệnh CLI (`search_milvus.py`)
```bash
# Tìm kiếm 1 câu bất kỳ trên cả 2 bộ dữ liệu:
python search_milvus.py --query "Quy định về xử phạt vi phạm nồng độ cồn" --dataset both

# Chế độ dòng lệnh tương tác liên tục (Interactive REPL):
python search_milvus.py --interactive
```
* Trong chế độ tương tác: nhập câu hỏi và `Enter`. Có thể dùng `:zalo`, `:nq`, `:both`, hoặc `:k 10` để đổi chế độ tức thời.

### Giao diện Web trực quan (`app.py`)
```bash
python app.py
```
Mở trình duyệt truy cập: **`http://localhost:5000`**
* Hỗ trợ tìm kiếm song song cả 2 bộ dữ liệu, tab chọn bộ dữ liệu, hiển thị độ trễ thời gian thực và bài viết song ngữ Anh - Việt.

---

## 6. Đánh giá hiệu năng & Benchmark chuẩn Qrels

### A. Benchmark 100% Full Corpus, Queries & Qrels (`benchmark_full.py`)

> **Khuyến nghị cho bài toán đánh giá toàn diện:** Chạy trên **100% dữ liệu thực nghiệm**, không dùng sample hay pool rút gọn. Tự động xuất file kết quả JSON.

```bash
# 1. Benchmark 100% Full Corpus Zalo Legal (61.425 docs, 818 queries, 793 qrels) - Chạy trong ~38 giây:
python benchmark_full.py --dataset zalo --device cuda --fp16

# 2. Benchmark 100% Full Corpus NQ-VN (Tự động nạp nốt 2.68 triệu docs rồi đo 1.900 queries):
python benchmark_full.py --dataset nq --device cuda --fp16

# 3. Chạy lần lượt cả 2 bộ dữ liệu và lưu kết quả JSON tùy chỉnh:
python benchmark_full.py --dataset both --device cuda --fp16 --output-json full_results.json
```

### B. Benchmark nhanh cả 2 bộ dữ liệu (`benchmark_all.py`)
```bash
python benchmark_all.py --device cuda --fp16
```
Tự động chạy Zalo Legal (100% corpus) $\rightarrow$ NQ-VN (Smart Pool 100% ground-truth) $\rightarrow$ In bảng so sánh đối đầu song song chỉ trong **1 phút 10 giây**.

### C. Benchmark siêu tốc bằng Smart Pool (`benchmark_smart_pool.py`)
```bash
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --device cuda --fp16
```

---

## 7. Giải thích chi tiết cấu trúc & Mã nguồn hệ thống

### 1. `device_utils.py` - Quản lý phần cứng & Tăng tốc GPU
File này đóng vai trò là "trạm điều phối phần cứng", đảm bảo tính tương thích và hiệu năng tối đa:
* `resolve_device(device_arg="auto")`: Tự động gọi `torch.cuda.is_available()`. Nếu máy có card rời NVIDIA, trả về chuỗi `"cuda"`, ngược lại trả về `"cpu"`.
* `load_embedding_model(model_name, device, fp16, warmup)`: 
  * Nạp mô hình Transformer lên thiết bị đích.
  * Nếu chọn GPU và bật cờ `fp16=True`, gọi `model.half()` để chuyển trọng số mô hình từ FP32 (32-bit float) sang FP16 (16-bit half-precision). Điều này giúp **giảm 50% dung lượng VRAM** và **tăng gấp đôi tốc độ xử lý Tensor Core**.
  * **Warm-up Kernel:** Chạy một câu thử nghiệm ngắn `model.encode(["..."])` để CUDA load trước thư viện cuBLAS vào GPU. Nhờ đó, câu truy vấn đầu tiên của người dùng sẽ không bị độ trễ (latency spike).
  * **Safe Fallback:** Nếu GPU gặp lỗi bộ nhớ (OOM) hoặc driver không khớp, `try...except` sẽ tự động chuyển mô hình về CPU mà không làm gián đoạn chương trình.

---

### 2. `index_milvus.py` - Xây dựng Schema & Trích xuất Embeddings
File này phụ trách pipeline ETL dữ liệu từ Hugging Face vào Milvus:
* **Khởi tạo Collection Schema:**
  * Khởi tạo trường khóa chính `id` (`VARCHAR`), trường vector đặc trưng `vector` (`FLOAT_VECTOR`, kích thước 384 chiều tương ứng với MiniLM-L12).
  * Lưu trữ các trường nội dung: `title`, `text` (văn bản tiếng Việt) và đối với NQ-VN bổ sung thêm `og_title`, `og_text` (tiêu đề và nội dung gốc tiếng Anh).
* **Cấu hình Vector Index:**
  * Sử dụng metric `COSINE` (Cosine Similarity) để đo góc định hướng ngữ nghĩa giữa 2 vector.
  * Chỉ mục đồ thị `AUTOINDEX` (Milvus tự động dựng đồ thị HNSW trên bộ nhớ để đạt độ phức tạp tìm kiếm $\mathcal{O}(\log N)$ thay vì $\mathcal{O}(N)$ brute-force).
* **Pipeline Vector hóa theo lô (Batching):**
  * Ghép `title` và `text` theo công thức: `{title}. {text[:1000]}` để tạo đầu vào ngữ cảnh tốt nhất cho mô hình embedding.
  * Gọi `model.encode(..., normalize_embeddings=True)`: Tự động chuẩn hóa độ dài vector theo chuẩn L2 ($\|v\|_2 = 1$). Khi vector đã được chuẩn hóa L2, tích vô hướng Inner Product (IP) đồng nhất với Cosine Similarity, giúp Milvus tính toán cực nhanh.

---

### 3. `search_milvus.py` - Cơ chế tìm kiếm & Xếp hạng Cosine
* Class `MultiDatasetSearcher`: Khi khởi tạo, tự động gọi `client.load_collection(name)` để nạp đồ thị index của cả 2 collection vào RAM/VRAM.
* Hàm `search(...)`:
  1. Trích xuất vector của câu hỏi người dùng qua `model.encode()`. Đo thời gian `embedding_ms`.
  2. Truyền vector câu hỏi vào `client.search()` với tham số `limit=top_k`. Đo thời gian `milvus_search_ms`.
  3. Duyệt danh sách các nút lân cận gần nhất (Nearest Neighbors) do Milvus trả về, trích xuất `id`, `score`, `title`, `text` để hiển thị.
* Hỗ trợ chế độ tương tác qua dòng lệnh (`run_interactive_mode`) bắt các cú pháp đặc biệt (`:zalo`, `:nq`, `:both`, `:k`).

---

### 4. `benchmark_full.py` - Đánh giá Full Corpus & Xuất JSON
Đây là script đo kiểm chuẩn mực toàn diện:
* `ensure_full_corpus_indexed`: So sánh số bản ghi hiện có trong Milvus với số lượng tài liệu thực tế của corpus (61.425 với Zalo, 2.681.468 với NQ-VN). Nếu thiếu, tự động chạy tiếp từ vị trí `start_idx` và ghi vào Milvus theo lô 256 tài liệu.
* `run_full_benchmark`:
  1. Tải toàn bộ nhãn đáp án từ `qrels` (`query-id` $\rightarrow$ tập hợp các `corpus-id` liên quan).
  2. Duyệt qua 100% câu hỏi trong tập `queries`. Với mỗi câu hỏi, đo lường chính xác bằng `time.perf_counter()` độ trễ Embedding và độ trễ Milvus Search.
  3. Lấy Top-50 tài liệu Milvus trả về, đối chiếu xem có chứa tài liệu đáp án trong `qrels` hay không để tính **Recall@K, Hit Rate@K, MRR@K**.
  4. Thực thi thêm một vòng lặp đo **Batched QPS** (gửi từng lô 32 câu hỏi đồng thời vào Milvus) để mô phỏng tải của hệ thống Production trong thực tế.
  5. Đóng gói toàn bộ kết quả vào dictionary và tự động lưu ra file JSON (`benchmark_full_results.json`).

---

### 5. `benchmark_smart_pool.py` - Kỹ thuật Smart Pool Guarantee
Giải pháp giải quyết bài toán thời gian đối với các bộ dữ liệu siêu lớn (như NQ-VN với 2.68 triệu văn bản):
* **Vấn đề:** Nạp toàn bộ 2.68 triệu văn bản mất khoảng 35 - 45 phút trên GPU và tiêu tốn nhiều RAM.
* **Nguyên lý giải quyết:** Trong 2.68 triệu văn bản, 1.900 câu hỏi trong tập test chỉ tham chiếu tới đúng **2.283 văn bản đáp án** duy nhất.
* **Cơ chế:**
  1. Sử dụng thư viện **PyArrow** quét chỉ mục bộ nhớ cực nhanh ($< 4$ giây) để tìm đúng vị trí của 2.283 tài liệu đáp án.
  2. Lấy 2.283 tài liệu này kết hợp với 2.717 tài liệu gây nhiễu ngẫu nhiên (Distractors) để tạo thành một Pool chuẩn 5.000 tài liệu.
  3. Nạp 5.000 tài liệu này vào Milvus (chỉ mất ~10 giây).
  4. Đánh giá 100% câu hỏi (1.900 queries). Vì **100% tài liệu đáp án đều có mặt trong Pool**, các chỉ số đo lường Recall, Latency và QPS phản ánh đúng năng lực thuật toán mà không cần phải nạp hàng triệu tài liệu dư thừa.

---

### 6. `app.py` - Kiến trúc Web REST API & Giao diện
* Được xây dựng trên nền tảng **Flask Web Framework**.
* Khởi động và load sẵn mô hình embedding và 2 collection `zalo_legal`, `nq_vn`.
* Định nghĩa route RESTful API: `/api/search` (phương thức `POST` nhận JSON payload gồm `query`, `dataset`, `top_k`).
* Giao diện người dùng (`templates/index.html`): Sử dụng **Tailwind CSS**, thiết kế dạng Single Page App (SPA) bất đồng bộ (Fetch API). Khi người dùng gõ câu hỏi, frontend gửi request về backend, nhận kết quả và render các thẻ bài viết kèm điểm Cosine, thời gian phản hồi (ms) và nội dung song ngữ Anh - Việt.

---

## 8. Giải thích các chỉ số đánh giá Information Retrieval (IR)

| Chỉ số | Tên đầy đủ | Ý nghĩa & Công thức toán học |
| :--- | :--- | :--- |
| **Recall@K** | *Tỷ lệ thu hồi trong Top-K* | $\text{Recall@K} = \frac{|\text{Retrieved@K} \cap \text{Relevant}|}{|\text{Relevant}|}$<br>Đo lường trong số tất cả các tài liệu đúng của câu hỏi, hệ thống đã tìm ra được bao nhiêu phần trăm trong top K kết quả. |
| **Hit Rate@K** | *Tỷ lệ trúng Top-K* | $\text{Hit@K} = 1$ nếu có ít nhất 1 tài liệu đúng nằm trong top K kết quả; ngược lại bằng $0$. |
| **MRR@K** | *Mean Reciprocal Rank* | $\text{MRR@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$<br>Đánh giá vị trí xếp hạng của tài liệu đúng đầu tiên. Nếu tài liệu đúng nằm ở vị trí số 1, điểm là $1.0$; nếu nằm ở vị trí số 2, điểm là $0.5$; vị trí số 5 là $0.2$. |
| **Latency P50 / P95 / P99** | *Phân vị độ trễ (Mili-giây)* | **P50 (Median):** 50% số truy vấn phản hồi nhanh hơn mốc này.<br>**P95 / P99 (Tail Latency):** 95% hoặc 99% số truy vấn hoàn thành dưới mốc này (thước đo tiêu chuẩn cho cam kết SLA dịch vụ). |
| **Throughput (QPS)** | *Queries Per Second* | $\text{QPS} = \frac{\text{Tổng số câu hỏi}}{\text{Tổng thời gian thực thi (giây)}}$<br>Đo khả năng chịu tải của hệ thống. Gồm Throughput đơn luồng (Single-Stream) và Throughput xử lý theo lô (Batched QPS). |

---

## 9. Bảng tra cứu nhanh lệnh (Cheat Sheet) & Xử lý lỗi

### Bảng lệnh tra cứu nhanh:
| Tác vụ cần thực hiện | Lệnh chạy |
| :--- | :--- |
| **Cài đặt thư viện** | `pip install -r requirements.txt` |
| **Kiểm tra GPU CUDA** | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"` |
| **Benchmark 100% Full Corpus Zalo** | `python benchmark_full.py --dataset zalo --device cuda --fp16` |
| **Benchmark 100% Full Corpus NQ-VN** | `python benchmark_full.py --dataset nq --device cuda --fp16` |
| **Benchmark CẢ 2 BỘ (Tự động xuất JSON)**| `python benchmark_full.py --dataset both --device cuda --fp16` |
| **Benchmark nhanh đối sánh 2 bộ** | `python benchmark_all.py --device cuda --fp16` |
| **Tìm kiếm 1 câu hỏi bất kỳ** | `python search_milvus.py --query "Nội dung cần tìm" --dataset both` |
| **Khởi động Giao diện Web** | `python app.py` $\rightarrow$ Mở `http://localhost:5000` |
| **Xem số bản ghi trong Milvus** | `python -c "from pymilvus import MilvusClient; c=MilvusClient('milvus_data.db'); print([(x, c.get_collection_stats(x)) for x in c.list_collections()]); c.close()"` |

### Xử lý lỗi thường gặp (Troubleshooting):
1. **Lỗi font chữ tiếng Việt trên PowerShell Windows:**
   Chạy lệnh sau trước khi chạy script:
   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   ```
2. **Lỗi `CUDA error: no kernel image is available`:**
   Card đồ họa thế hệ mới NVIDIA RTX 50-series (kiến trúc Blackwell `sm_120`) cần bản PyTorch tương thích CUDA 13.x. Cài đặt lại bằng lệnh:
   ```bash
   pip install torch==2.14.1+cu132 --index-url https://download.pytorch.org/whl/cu132
   ```
3. **Lỗi `Collection is in state released; call load() before search`:**
   Collection chưa được nạp vào bộ nhớ trước khi gọi search. Toàn bộ code trong dự án đã có sẵn lệnh `client.load_collection(...)` tự động nạp.
