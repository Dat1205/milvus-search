# Hướng dẫn Tìm kiếm Dữ liệu với Milvus Vector Database

Dự án này sử dụng **Milvus** để lập chỉ mục vector (embedding) và tìm kiếm ngữ nghĩa (semantic search) trên các tập dữ liệu đã tải:
- **Zalo AI Legal Text Retrieval**: 61,425 điều khoản và văn bản pháp luật Việt Nam.
- **Natural Questions VN (NQ-VN)**: 2.68 triệu đoạn văn hỏi đáp tiếng Việt.

---

## 📁 Cấu trúc Thư mục

```text
milvus/
├── download_data.py   # Script tải dữ liệu từ Hugging Face
├── index_milvus.py    # Script tính vector embedding và nạp vào Milvus
├── search_milvus.py   # Script tìm kiếm (CLI một câu hỏi, REPL tương tác, đánh giá benchmark)
├── app.py             # Ứng dụng Web UI tìm kiếm bằng Flask
└── templates/
    └── index.html     # Giao diện Web trực quan (Tailwind CSS)
```

---

## 🚀 1. Lập chỉ mục dữ liệu vào Milvus (`index_milvus.py`)

Hệ thống sử dụng **Milvus Lite** (tự động tạo file database cục bộ `.db`, không cần cài đặt Docker) hoặc kết nối tới Milvus Server Standalone/Cluster.

Mô hình embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, tối ưu cho tiếng Việt).

### Ví dụ nạp dữ liệu:

- **Nạp thử nghiệm 5,000 tài liệu Zalo Legal:**
  ```bash
  python index_milvus.py --limit 5000 --recreate
  ```

- **Nạp toàn bộ 61,425 tài liệu Zalo Legal:**
  ```bash
  python index_milvus.py --recreate
  ```

- **Nạp tập dữ liệu NQ-VN (ví dụ 10,000 tài liệu):**
  ```bash
  python index_milvus.py --dataset nq --limit 10000 --collection-name nq_vn --recreate
  ```

- **Tùy chọn kết nối tới Milvus Docker/Server từ xa:**
  ```bash
  python index_milvus.py --db-path "http://localhost:19530"
  ```

---

## 🔍 2. Tìm kiếm (`search_milvus.py`)

### A. Tìm kiếm một câu hỏi trực tiếp từ dòng lệnh:
```bash
python search_milvus.py --query "Quy định về bảo vệ đê điều trong mùa lũ" --top-k 5
```

### B. Chế độ tương tác liên tục (Interactive CLI):
```bash
python search_milvus.py --interactive
```
*Nhập câu hỏi bằng tiếng Việt bất kỳ, gõ `exit` để thoát.*

### C. Đánh giá độ chính xác trên Benchmark (Zalo Legal Queries & Qrels):
```bash
python search_milvus.py --evaluate --eval-limit 100
```
*Tự động tính các chỉ số đo lường chuẩn: **Hit@1**, **Hit@5**, **Hit@10**, **MRR@10**.*

---

## 🌐 3. Giao diện Web Trực quan (`app.py`)

Khởi động giao diện Web để tìm kiếm trực tiếp trên trình duyệt:

```bash
python app.py
```

Mở trình duyệt truy cập: **[http://localhost:5000](http://localhost:5000)**
- Giao diện tra cứu hiện đại, hiển thị thời gian embedding và thời gian truy vấn Milvus (tính bằng mili-giây).
- Đi kèm sẵn các nút gợi ý câu hỏi mẫu để thử nghiệm ngay lập tức.

