"""
Flask Web UI for Searching Documents with Milvus
Run: python app.py
Open: http://localhost:5000
"""

import sys
import time
from flask import Flask, render_template, request, jsonify
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)

DB_PATH = "milvus_legal.db"
COLLECTION_NAME = "zalo_legal"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"[*] Đang khởi tạo mô hình embedding: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print(f"[*] Kết nối Milvus tại: {DB_PATH}...")
client = MilvusClient(uri=DB_PATH)

if client.has_collection(COLLECTION_NAME):
    client.load_collection(COLLECTION_NAME)
    print(f"[+] Đã load collection '{COLLECTION_NAME}' sẵn sàng!")
else:
    print(f"[!] Cảnh báo: Collection '{COLLECTION_NAME}' chưa được khởi tạo. Hãy chạy index_milvus.py trước.")


@app.route("/")
def index():
    has_col = client.has_collection(COLLECTION_NAME)
    stats = client.get_collection_stats(COLLECTION_NAME) if has_col else {"row_count": 0}
    return render_template(
        "index.html",
        collection_name=COLLECTION_NAME,
        db_path=DB_PATH,
        doc_count=stats.get("row_count", 0)
    )


@app.route("/api/search", methods=["POST"])
def search_api():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "Vui lòng nhập câu hỏi tìm kiếm"}), 400

    if not client.has_collection(COLLECTION_NAME):
        return jsonify({"error": f"Collection '{COLLECTION_NAME}' chưa tồn tại trong Milvus"}), 400

    t0 = time.time()
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    t_embed = time.time() - t0

    t1 = time.time()
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["id", "title", "text"]
    )
    t_search = time.time() - t1

    hits = []
    if search_res and len(search_res) > 0:
        for hit in search_res[0]:
            entity = hit.get("entity", {})
            hits.append({
                "id": hit.get("id"),
                "score": round(float(hit.get("distance", 0.0)), 4),
                "title": entity.get("title", ""),
                "text": entity.get("text", "")
            })

    return jsonify({
        "query": query,
        "hits": hits,
        "metrics": {
            "embed_ms": round(t_embed * 1000, 2),
            "search_ms": round(t_search * 1000, 2),
            "total_ms": round((t_embed + t_search) * 1000, 2),
            "total_results": len(hits)
        }
    })


if __name__ == "__main__":
    print("[+] Khởi động Web Server tại http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)

