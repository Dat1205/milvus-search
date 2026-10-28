# Hướng Dẫn Chi Tiết Triển Khai Milvus Vector Database

Tài liệu này cung cấp toàn bộ hướng dẫn từ cơ bản đến nâng cao để triển khai và vận hành **Milvus Vector Database** phục vụ bài toán tìm kiếm ngữ nghĩa (Semantic Search) và RAG (Retrieval-Augmented Generation).

---

## 📑 Mục lục
1. [Tổng quan & Kiến trúc Milvus](#1-tổng-quan--kiến-trúc-milvus)
2. [So sánh 3 mô hình triển khai](#2-so-sánh-3-mô-hình-triển-khai)
3. [Phương pháp 1: Triển khai Milvus Lite (Nhúng - Cục bộ)](#3-phương-pháp-1-triển-khai-milvus-lite-nhúng---cục-bộ)
4. [Phương pháp 2: Triển khai Milvus Standalone với Docker Compose (Kèm Web GUI Attu)](#4-phương-pháp-2-triển-khai-milvus-standalone-với-docker-compose-kèm-web-gui-attu)
5. [Phương pháp 3: Triển khai Milvus Cluster trên Kubernetes (Sản xuất quy mô lớn)](#5-phương-pháp-3-triển-khai-milvus-cluster-trên-kubernetes-sản-xuất-quy-mô-lớn)
6. [Tích hợp mã nguồn Python (Chuyển đổi linh hoạt)](#6-tích-hợp-mã-nguồn-python-chuyển-đổi-linh-hoạt)
7. [Tối ưu hóa Vector Index & Cấu hình Search](#7-tối-ưu-hóa-vector-index--cấu-hình-search)
8. [Dự toán tài nguyên phần cứng (Hardware Sizing)](#8-dự-toán-tài-nguyên-phần-cứng-hardware-sizing)
9. [Sao lưu (Backup), Phục hồi & Giám sát](#9-sao-lưu-backup-phục-hồi--giám-sát)

---

## 1. Tổng quan & Kiến trúc Milvus

Milvus là cơ sở dữ liệu vector mã nguồn mở hiệu năng cao, được thiết kế chuyên biệt để lưu trữ, lập chỉ mục và truy vấn hàng tỷ vector đặc trưng từ các mô hình AI/Deep Learning.

### Các thành phần cốt lõi trong hệ thống Milvus:
* **Proxy (Access Layer):** Cổng giao tiếp phân tán, xác thực truy vấn và định tuyến request từ client qua gRPC/REST.
* **Coordinator (Điều phối):** Quản lý topology cụm, phân chia metadata và giao việc cho các node:
  * `RootCoord`: Quản lý các lệnh DDL (tạo/xóa collection) và time tick toàn cục.
  * `QueryCoord`: Quản lý phân vùng và gán các segment vào bộ nhớ các QueryNode.
  * `DataCoord`: Quản lý các segment dữ liệu ghi và giao việc cho DataNode.
  * `IndexCoord`: Quản lý việc xây dựng chỉ mục (index building) trên IndexNode.
* **Worker Nodes (Xử lý tác vụ):**
  * `QueryNode`: Nạp vector vào RAM/VRAM và thực thi tìm kiếm ANN song song.
  * `DataNode`: Ghi nhận dữ liệu thô (insert/update) và gom thành các immutable segment.
  * `IndexNode`: Tính toán xây dựng chỉ mục vector (HNSW, IVF_FLAT, SCaNN).
* **Storage & Meta Dependencies:**
  * **etcd:** Lưu trữ metadata trạng thái cluster và service discovery.
  * **MinIO / Amazon S3 / GCS:** Lưu trữ dữ liệu vector dạng object storage bền vững.
  * **Apache Pulsar / Kafka:** Message log stream lưu trữ nhật ký ghi trước (WAL).

---

## 2. So sánh 3 mô hình triển khai

| Tiêu chí | Milvus Lite (Nhúng) | Milvus Standalone (Docker) | Milvus Cluster (Kubernetes) |
| :--- | :--- | :--- | :--- |
| **Bản chất** | Thư viện Python cục bộ (nhúng) | 1 Container Milvus + MinIO + etcd | Kiến trúc phân tán Microservices |
| **Yêu cầu cài đặt** | Chỉ cần `pip install pymilvus milvus-lite` | Docker & Docker Compose | Kubernetes Cluster, Helm |
| **Quy mô dữ liệu** | < 1 triệu vector | 1 - 50 triệu vector | Hàng trăm triệu đến hàng tỷ vector |
| **Tính sẵn sàng (HA)** | Không (Single process) | Không (Phụ thuộc 1 host) | Cao (High Availability, Auto-failover) |
| **Hỗ trợ GPU** | Không (Chạy CPU) | Có (Milvus GPU image) | Có (GPU Query/Index nodes) |
| **Phù hợp cho** | **Phát triển, PoC, ứng dụng Desktop, dự án vừa** | **Môi trường Server nội bộ, RAG Production vừa** | **Production Doanh nghiệp lớn, Multi-tenant** |

---

## 3. Phương pháp 1: Triển khai Milvus Lite (Nhúng - Cục bộ)

> Đây là phương pháp **đang được sử dụng trực tiếp trong thư mục làm việc hiện tại** của dự án (`d:\New\milvus\milvus_data.db`).

### Ưu điểm:
* **Zero Docker / Zero Setup:** Không cần cài đặt container hay cấu hình mạng.
* Dữ liệu gói gọn trong một thư mục/file `.db` cục bộ, dễ dàng copy, chia sẻ hoặc đưa vào Git LFS.
* Tương thích 100% với cú pháp API `MilvusClient` của phiên bản Server.

### Cách cài đặt & sử dụng:

1. **Cài đặt thư viện:**
   ```bash
   pip install pymilvus milvus-lite
   ```

2. **Khởi tạo và kết nối trong Python:**
   ```python
   from pymilvus import MilvusClient

   # Khởi tạo Milvus Lite bằng đường dẫn file cục bộ
   client = MilvusClient(uri="./milvus_data.db")

   # Kiểm tra danh sách collection
   print(client.list_collections())
   ```

---

## 4. Phương pháp 2: Triển khai Milvus Standalone với Docker Compose (Kèm Web GUI Attu)

Khi hệ thống cần phục vụ nhiều user đồng thời (API Server), hỗ trợ tăng tốc GPU hoặc muốn có giao diện đồ họa Web GUI để xem và quản lý collection, mô hình Docker Compose là lựa chọn tối ưu.

File cấu hình mẫu đã được tạo sẵn tại: **[docker-compose.yml](file:///d:/New/milvus/docker-compose.yml)**.

### Chi tiết các dịch vụ trong `docker-compose.yml`:
* **`milvus-standalone`** (Port `19530` gRPC, `9091` HTTP): Server Milvus 2.4.x.
* **`milvus-etcd`** (Port `2379`): Quản lý metadata và trạng thái.
* **`milvus-minio`** (Port `9000` API, `9001` Web Console): Lưu trữ snapshot vector và segment files.
* **`milvus-attu`** (Port `8000` Web GUI): Giao diện quản trị đồ họa trực quan do Zilliz phát triển.

### Các bước khởi chạy:

1. **Khởi động toàn bộ cụm Milvus:**
   ```bash
   docker compose up -d
   ```

2. **Kiểm tra trạng thái các container:**
   ```bash
   docker compose ps
   ```
   *Đảm bảo cả 4 service (`standalone`, `etcd`, `minio`, `attu`) đều ở trạng thái `Up (healthy)`.*

3. **Truy cập các cổng dịch vụ:**
   * **Milvus Service (gRPC):** `localhost:19530` (dùng để code Python kết nối)
   * **Attu Web Management GUI:** Mở trình duyệt truy cập **`http://localhost:8000`**
     * Kết nối tới: `milvus-standalone:19530` (không cần user/pass mặc định).
     * Cho phép xem danh sách collection, số lượng vector, schema, chạy thử câu truy vấn vector ngay trên giao diện web.
   * **MinIO Console:** `http://localhost:9001` (User: `minioadmin` / Pass: `minioadmin`).

4. **Dừng hệ thống:**
   ```bash
   docker compose down
   ```
   *(Dữ liệu vẫn được giữ nguyên an toàn trong thư mục `./volumes/`)*.

---

## 5. Phương pháp 3: Triển khai Milvus Cluster trên Kubernetes (Sản xuất quy mô lớn)

Phù hợp cho các doanh nghiệp cần khả năng co giãn linh hoạt (Elastic Scaling), tự động phục hồi (Self-healing) và lưu trữ từ hàng chục triệu đến hàng tỷ vector.

### Các bước cài đặt với Helm:

1. **Thêm Helm Repo của Milvus:**
   ```bash
   helm repo add milvus https://zilliztech.github.io/milvus-helm/
   helm repo update
   ```

2. **Cấu hình file `values.yaml` (Rút gọn):**
   ```yaml
   cluster:
     enabled: true

   etcd:
     replicaCount: 3
     persistence:
       size: 20Gi

   minio:
     mode: distributed
     replicas: 4
     persistence:
       size: 500Gi

   pulsar:
     enabled: true

   queryNode:
     replicas: 4
     resources:
       limits:
         cpu: "8"
         memory: "32Gi"
   ```

3. **Cài đặt cụm vào namespace `milvus`:**
   ```bash
   helm install my-milvus milvus/milvus -n milvus --create-namespace -f values.yaml
   ```

---

## 6. Tích hợp mã nguồn Python (Chuyển đổi linh hoạt)

Điểm mạnh lớn nhất của thư viện `pymilvus.MilvusClient` là **mã nguồn không cần thay đổi bất kỳ logic nào**, chỉ cần đổi địa chỉ `uri`:

### Ví dụ code chuyển đổi:

```python
from pymilvus import MilvusClient

# =======================================================
# LỰA CHỌN 1: Dùng Milvus Lite (Cục bộ không cần Docker)
# =======================================================
client = MilvusClient(uri="./milvus_data.db")

# =======================================================
# LỰA CHỌN 2: Dùng Milvus Server / Docker Standalone
# =======================================================
# client = MilvusClient(uri="http://localhost:19530")

# =======================================================
# LỰA CHỌN 3: Dùng Milvus Cluster (Có mật khẩu xác thực)
# =======================================================
# client = MilvusClient(
#     uri="http://milvus-proxy.internal:19530",
#     token="root:Milvus@2026"
# )
```

Các file trong dự án hiện tại ([search_milvus.py](file:///d:/New/milvus/search_milvus.py), [index_milvus.py](file:///d:/New/milvus/index_milvus.py), [app.py](file:///d:/New/milvus/app.py)) đều có cờ `--db-path`. Bạn chỉ cần truyền:
```bash
python search_milvus.py --db-path "http://localhost:19530" --query "Luật giao thông"
```

---

## 7. Tối ưu hóa Vector Index & Cấu hình Search

Lựa chọn chỉ mục (Index) quyết định sự cân bằng giữa **Độ chính xác (Recall)**, **Độ trễ (Latency)** và **Dung lượng RAM**.

### A. So sánh các loại Index phổ biến:

| Loại Index | Tốc độ Query | Độ chính xác (Recall) | Tiêu thụ RAM | Khuyên dùng |
| :--- | :--- | :--- | :--- | :--- |
| **HNSW** (Đồ thị Small World) | ⚡⚡⚡ Rất nhanh (5-10ms) | ⭐⭐⭐ Rất cao (>95%) | Cao (~1.3x vector thô) | **Khuyên dùng số 1 cho hầu hết bài toán RAG** |
| **IVF_FLAT** (Phân cụm Voronoi) | ⚡⚡ Nhanh | ⭐⭐ Khá (85-90%) | Thấp (~1.0x) | Dữ liệu lớn, tài nguyên RAM hạn chế |
| **AUTOINDEX** | ⚡⚡⚡ Tự tối ưu | ⭐⭐⭐ Tự điều chỉnh | Trung bình | Milvus tự động chọn tham số phù hợp |
| **SCaNN** | ⚡⚡⚡ Cực nhanh | ⭐⭐⭐ Rất cao | Trung bình | Tối ưu hóa tập lệnh AVX-512 CPU |

### B. Cấu hình Index HNSW chuẩn:

```python
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    metric_type="COSINE",         # COSINE (khuyên dùng cho text), L2 (Euclidean), IP (Dot Product)
    index_type="HNSW",            # Đồ thị phân tầng
    params={
        "M": 16,                  # Số liên kết tối đa của mỗi node trên đồ thị (8 - 64)
        "efConstruction": 200     # Độ sâu tìm kiếm khi xây dựng đồ thị (càng lớn build càng chuẩn nhưng lâu)
    }
)
```

### C. Tham số khi Search (`search_params`):
```python
search_params = {
    "metric_type": "COSINE",
    "params": {
        "ef": 64                  # Độ sâu tìm kiếm khi query (16 - 256). ef càng cao -> Recall càng cao, latency tăng nhẹ.
    }
}

results = client.search(
    collection_name="zalo_legal",
    data=[query_vector],
    limit=10,
    search_params=search_params,
    output_fields=["id", "title", "text"]
)
```

---

## 8. Dự toán tài nguyên phần cứng (Hardware Sizing)

Công thức tính dung lượng RAM cần thiết cho Milvus:
$$\text{RAM}_{\text{total}} = N \times d \times 4 \text{ bytes} \times \text{Overhead}_{\text{index}} \times 1.25 \text{ (OS/Buffer)}$$

Trong đó:
* $N$: Số lượng vector.
* $d$: Số chiều (dimension) của vector (mô hình `MiniLM-L12-v2` là $d=384$).
* $4 \text{ bytes}$: Kích thước kiểu dữ liệu `float32`.
* $\text{Overhead}_{\text{index}}$: Hệ số cấu trúc đồ thị (với HNSW khoảng $1.35 \sim 1.4$).

### Bảng dự toán cho 2 tập dữ liệu của bạn:

| Quy mô dữ liệu | Số lượng vector ($N$) | RAM chứa Vector | RAM chứa HNSW Index | Ổ cứng (SSD) tối thiểu | CPU đề xuất |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zalo AI Legal (Toàn bộ)** | **61.425** | ~90 MB | ~130 MB | ~1 GB | 2 - 4 Cores |
| **NQ-VN (Thử nghiệm)** | **50.000** | ~75 MB | ~110 MB | ~1 GB | 2 - 4 Cores |
| **NQ-VN (Toàn bộ)** | **2.681.468** | ~3.9 GB | ~5.5 GB | ~25 GB NVMe SSD | 8 - 16 Cores |
| **Hệ thống lớn Production** | **10.000.000** | ~14.6 GB | ~21.0 GB | ~100 GB NVMe SSD | 16 - 32 Cores |

---

## 9. Sao lưu (Backup), Phục hồi & Giám sát

### A. Sao lưu với Milvus Lite:
Chỉ cần copy thư mục database:
```bash
# Sao lưu
xcopy /E /I milvus_data.db milvus_data_backup_20261028.db

# Phục hồi
# Chỉ cần trỏ URI tới file backup
```

### B. Sao lưu với Milvus Standalone (Docker):
Dữ liệu nằm ở thư mục ánh xạ `./volumes/`:
* Thư mục `volumes/minio`: Chứa toàn bộ vector segments và index binary.
* Thư mục `volumes/etcd`: Chứa metadata cấu trúc collection.

Công cụ chính thức **`milvus-backup`**:
```bash
# Tải công cụ milvus-backup
wget https://github.com/zilliztech/milvus-backup/releases/download/v0.3.0/milvus-backup_v0.3.0_Linux_x86_64.tar.gz
tar -zxvf milvus-backup_v0.3.0_Linux_x86_64.tar.gz

# Tạo bản backup
./milvus-backup create --backup_name=my_backup_legal

# Khôi phục từ backup
./milvus-backup restore --backup_name=my_backup_legal --restore_index=true
```

### C. Giám sát hệ thống (Monitoring):
* **Attu GUI:** Truy cập `http://localhost:8000` để xem trực tiếp số bản ghi, tỷ lệ nạp memory và trạng thái index.
* **Prometheus & Grafana:** Milvus Standalone cung cấp sẵn metrics endpoint tại:
  `http://localhost:9091/metrics`
  Tích hợp trực tiếp với Grafana Dashboard chính thức của Milvus (Dashboard ID: `19036`).

