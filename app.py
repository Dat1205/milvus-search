"""
Flask Web UI for Multi-Dataset Semantic Search with Milvus
Supports searching across:
  1. Zalo AI Legal (Pháp luật Việt Nam)
  2. NQ-VN (Bách khoa toàn thư & Tri thức mở)
"""

import sys
import time
from flask import Flask, render_template, request, jsonify
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from device_utils import load_embedding_model, get_device_info, resolve_device

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)

DB_PATH = "milvus_data.db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"[*] Đang nạp mô hình embedding: {MODEL_NAME}...")
model = load_embedding_model(MODEL_NAME, device="auto", fp16=True)

print(f"[*] Kết nối Milvus tại: {DB_PATH}...")
client = MilvusClient(uri=DB_PATH)

# Load các collections sẵn sàng
collections = client.list_collections()
print(f"[+] Các collections hiện có: {collections}")
for col in ["zalo_legal", "nq_vn"]:
    if col in collections:
        try:
            client.load_collection(col)
            print(f"    • Loaded {col}")
        except Exception as e:
            print(f"    • Error loading {col}: {e}")


@app.route("/")
def index():
    cols = client.list_collections()
    zalo_count = client.get_collection_stats("zalo_legal").get("row_count", 0) if "zalo_legal" in cols else 0
    nq_count = client.get_collection_stats("nq_vn").get("row_count", 0) if "nq_vn" in cols else 0
    return render_template(
        "index.html",
        db_path=DB_PATH,
        zalo_count=zalo_count,
        nq_count=nq_count,
    )


@app.route("/api/search", methods=["POST"])
def search_api():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    dataset_choice = data.get("dataset", "both")
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "Vui lòng nhập câu hỏi tìm kiếm"}), 400

    cols = client.list_collections()
    t0 = time.time()
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    t_embed = time.time() - t0

    t1 = time.time()
    results = {}

    # 1. Tìm kiếm trên Zalo Legal
    if dataset_choice in ["zalo", "both"] and "zalo_legal" in cols:
        try:
            z_res = client.search(
                collection_name="zalo_legal",
                data=[query_vector],
                limit=top_k,
                output_fields=["id", "title", "text"]
            )
            hits = []
            if z_res and len(z_res) > 0:
                for hit in z_res[0]:
                    entity = hit.get("entity", {})
                    hits.append({
                        "id": hit.get("id"),
                        "score": round(float(hit.get("distance", 0.0)), 4),
                        "title": entity.get("title", ""),
                        "text": entity.get("text", "")
                    })
            results["zalo_legal"] = hits
        except Exception as e:
            results["zalo_legal_error"] = str(e)

    # 2. Tìm kiếm trên NQ-VN
    if dataset_choice in ["nq", "both"] and "nq_vn" in cols:
        try:
            nq_res = client.search(
                collection_name="nq_vn",
                data=[query_vector],
                limit=top_k,
                output_fields=["id", "title", "text", "og_title", "og_text"]
            )
            hits = []
            if nq_res and len(nq_res) > 0:
                for hit in nq_res[0]:
                    entity = hit.get("entity", {})
                    hits.append({
                        "id": hit.get("id"),
                        "score": round(float(hit.get("distance", 0.0)), 4),
                        "title": entity.get("title", ""),
                        "text": entity.get("text", ""),
                        "og_title": entity.get("og_title", ""),
                        "og_text": entity.get("og_text", "")
                    })
            results["nq_vn"] = hits
        except Exception as e:
            results["nq_vn_error"] = str(e)

    t_search = time.time() - t1

    return jsonify({
        "query": query,
        "dataset": dataset_choice,
        "results": results,
        "metrics": {
            "embed_ms": round(t_embed * 1000, 2),
            "search_ms": round(t_search * 1000, 2),
            "total_ms": round((t_embed + t_search) * 1000, 2),
        }
    })


if __name__ == "__main__":
    print("[+] Khởi động Web Server tại http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)

