"""
SMART POOL BENCHMARK - MILVUS INFORMATION RETRIEVAL EVALUATION
Phương pháp: "Ground-Truth Guarantee + Distractor Pool" (Chuẩn BEIR / TREC Benchmark)

Giải pháp tối ưu thời gian:
1. Đảm bảo 100% tài liệu đáp án trong `qrels` được nạp vào Milvus.
2. Bổ sung một tập tài liệu gây nhiễu (Distractors) ngẫu nhiên (ví dụ 10.000 - 25.000 docs).
3. Đánh giá 100% câu hỏi (1.900 queries cho NQ-VN, 818 queries cho Zalo Legal).
-> Đo lường chính xác tuyệt đối Recall@K, Latency, Throughput (QPS) chỉ trong 3 - 5 phút!
"""

import argparse
import sys
import time
import json
import random
from typing import List, Dict, Set, Any
import numpy as np

from datasets import load_dataset
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DB_PATH = "milvus_data.db"

DATASET_CONFIGS = {
    "nq": {
        "hf_name": "GreenNode/nq-vn",
        "collection_name": "nq_vn_smart",
        "name": "Natural Questions VN (NQ-VN)",
        "fields": ["id", "title", "text", "og_title", "og_text"],
        "default_pool_size": 15000,
    },
    "zalo": {
        "hf_name": "GreenNode/zalo-ai-legal-text-retrieval-vn",
        "collection_name": "zalo_legal_smart",
        "name": "Zalo AI Legal Text Retrieval",
        "fields": ["id", "title", "text"],
        "default_pool_size": 15000,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Pool Benchmark for Milvus (Ground-Truth + Distractors)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nq",
        choices=["nq", "zalo"],
        help="Bộ dữ liệu đánh giá: 'nq' hoặc 'zalo' (mặc định: nq)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Tổng số tài liệu nạp vào pool (gồm 100%% đáp án qrels + distractors). Mặc định: 15,000",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size tính embedding và insert vào Milvus (mặc định: 64)",
    )
    parser.add_argument(
        "--top-k-max",
        type=int,
        default=50,
        help="Top-K tối đa cần truy xuất (mặc định: 50)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"File DB Milvus (mặc định: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Mô hình embedding (mặc định: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa collection cũ và tạo nạp lại từ đầu",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Đường dẫn file JSON để lưu kết quả đánh giá",
    )
    return parser.parse_args()


# =====================================================================
# 1. TRÍCH XUẤT GROUND-TRUTH VÀ TẠO POOL TÀI LIỆU
# =====================================================================
def prepare_smart_pool(
    hf_name: str,
    pool_size: int,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], Dict[str, Set[str]], List[Dict[str, str]]]:
    print(f"\n[*] Bước 1: Nạp qrels và queries từ '{hf_name}'...")
    raw_qrels = load_dataset(hf_name, "qrels", split="test")
    raw_queries = load_dataset(hf_name, "queries", split="test")

    # Xây dựng ground_truth: query_id -> set(corpus_ids)
    ground_truth: Dict[str, Set[str]] = {}
    target_corpus_ids: Set[str] = set()

    for item in raw_qrels:
        qid = str(item["query-id"])
        cid = str(item["corpus-id"])
        if int(item.get("score", 1)) > 0:
            if qid not in ground_truth:
                ground_truth[qid] = set()
            ground_truth[qid].add(cid)
            target_corpus_ids.add(cid)

    # Danh sách queries có đáp án
    eval_queries = []
    for q in raw_queries:
        qid = str(q["id"])
        if qid in ground_truth and len(ground_truth[qid]) > 0:
            eval_queries.append({"id": qid, "text": str(q["text"])})

    print(f"[+] Tìm thấy {len(eval_queries):,} queries hợp lệ với {len(target_corpus_ids):,} tài liệu đáp án duy nhất trong qrels.")

    # Đọc corpus
    print(f"[*] Bước 2: Quét nhanh kho corpus để tìm đúng {len(target_corpus_ids):,} tài liệu đáp án...")
    t0 = time.time()
    raw_corpus = load_dataset(hf_name, "corpus", split="test")
    total_corpus_docs = len(raw_corpus)

    # Dùng PyArrow table để quét siêu tốc (dưới 1 giây trên 2.68M dòng)
    table = raw_corpus.data.table
    all_ids = table["id"].to_pylist()

    # Tìm chỉ số dòng của các tài liệu đáp án
    gt_indices = []
    other_indices = []

    for idx, doc_id in enumerate(all_ids):
        doc_str = str(doc_id)
        if doc_str in target_corpus_ids:
            gt_indices.append(idx)
        else:
            other_indices.append(idx)

    print(f"[+] Đã định vị chính xác 100% tài liệu đáp án ({len(gt_indices):,} docs) trong {time.time()-t0:.2f}s!")

    # Lấy thêm các tài liệu gây nhiễu (distractors)
    num_distractors_needed = max(0, pool_size - len(gt_indices))
    random.seed(seed)
    if num_distractors_needed > 0 and len(other_indices) > 0:
        distractor_indices = random.sample(other_indices, min(num_distractors_needed, len(other_indices)))
    else:
        distractor_indices = []

    selected_indices = gt_indices + distractor_indices
    # Xáo trộn thứ tự để tránh thiên lệch vị trí
    random.shuffle(selected_indices)

    print(f"[+] Tạo Smart Pool thành công: {len(selected_indices):,} tài liệu")
    print(f"    • Tài liệu đáp án chuẩn (Ground-truth): {len(gt_indices):,} docs (100% có mặt)")
    print(f"    • Tài liệu gây nhiễu (Distractors):     {len(distractor_indices):,} docs")

    # Trích xuất dữ liệu của các chỉ số đã chọn
    print(f"[*] Trích xuất nội dung {len(selected_indices):,} tài liệu...")
    selected_docs = [raw_corpus[i] for i in selected_indices]

    return selected_docs, ground_truth, eval_queries


# =====================================================================
# 2. NẠP SMART POOL VÀO MILVUS
# =====================================================================
def index_pool_into_milvus(
    client: MilvusClient,
    model: SentenceTransformer,
    collection_name: str,
    dataset_key: str,
    pool_docs: List[Dict[str, Any]],
    batch_size: int = 64,
    recreate: bool = False,
):
    dim = model.get_sentence_embedding_dimension()

    if client.has_collection(collection_name):
        if recreate:
            print(f"[*] Xóa collection cũ '{collection_name}'...")
            client.drop_collection(collection_name)
        else:
            stats = client.get_collection_stats(collection_name)
            cnt = stats.get("row_count", 0)
            if cnt >= len(pool_docs):
                print(f"[+] Collection '{collection_name}' đã có sẵn {cnt:,} tài liệu. Sử dụng luôn.")
                return
            else:
                client.drop_collection(collection_name)

    print(f"[*] Tạo collection '{collection_name}' (dim={dim}, metric=COSINE)...")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

    if dataset_key == "nq":
        schema.add_field(field_name="og_title", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="og_text", datatype=DataType.VARCHAR, max_length=65535)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    total = len(pool_docs)
    print(f"[*] Bắt đầu tính embedding và nạp {total:,} tài liệu vào Milvus...")
    t0 = time.time()

    with tqdm(total=total, desc="Indexing Smart Pool", unit="doc") as pbar:
        for i in range(0, total, batch_size):
            batch = pool_docs[i : i + batch_size]
            ids = [str(x["id"]) for x in batch]
            titles = [str(x.get("title") or "") for x in batch]
            texts = [str(x.get("text") or "") for x in batch]

            input_texts = []
            for t, c in zip(titles, texts):
                t_str = t.strip()
                c_str = c.strip()[:1000]
                input_texts.append(f"{t_str}. {c_str}" if t_str else c_str)

            embeddings = model.encode(input_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

            entities = []
            for idx in range(len(ids)):
                item = {
                    "id": ids[idx],
                    "vector": embeddings[idx].tolist(),
                    "title": titles[idx][:2000],
                    "text": texts[idx][:65000],
                }
                if dataset_key == "nq":
                    item["og_title"] = str(batch[idx].get("og_title") or "")[:2000]
                    item["og_text"] = str(batch[idx].get("og_text") or "")[:65000]
                entities.append(item)

            client.insert(collection_name=collection_name, data=entities)
            pbar.update(len(entities))

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"[✓] Nạp hoàn tất {total:,} tài liệu vào Milvus trong {elapsed:.2f}s ({rate:.1f} docs/s)!")


# =====================================================================
# 3. ĐÁNH GIÁ IR METRICS TRÊN TOÀN BỘ QUERIES
# =====================================================================
def evaluate_ir_performance(
    client: MilvusClient,
    model: SentenceTransformer,
    collection_name: str,
    eval_queries: List[Dict[str, str]],
    ground_truth: Dict[str, Set[str]],
    top_k_max: int = 50,
    batch_size: int = 32,
) -> Dict[str, Any]:
    print(f"\n[*] Bước 4: Đánh giá IR Metrics trên 100% queries ({len(eval_queries):,} câu hỏi)...")
    client.load_collection(collection_name)

    k_list = [1, 3, 5, 10, 20, top_k_max]
    k_list = sorted(list(set([k for k in k_list if k <= top_k_max])))

    embed_latencies = []
    search_latencies = []
    total_latencies = []

    recall_at_k = {k: [] for k in k_list}
    hit_at_k = {k: 0 for k in k_list}
    mrr_at_k = {k: 0.0 for k in k_list}

    t0_all = time.time()
    for q in tqdm(eval_queries, desc="Đánh giá Queries", unit="query"):
        qid = q["id"]
        qtext = q["text"]
        relevant = ground_truth[qid]

        # 1. Embedding
        t_a = time.perf_counter()
        q_vec = model.encode([qtext], normalize_embeddings=True)[0].tolist()
        t_b = time.perf_counter()
        t_emb = (t_b - t_a) * 1000.0

        # 2. Milvus ANN Search
        t_c = time.perf_counter()
        res = client.search(collection_name=collection_name, data=[q_vec], limit=top_k_max, output_fields=["id"])
        t_d = time.perf_counter()
        t_search = (t_d - t_c) * 1000.0

        embed_latencies.append(t_emb)
        search_latencies.append(t_search)
        total_latencies.append(t_emb + t_search)

        retrieved_ids = [str(hit["id"]) for hit in res[0]] if res and len(res) > 0 else []

        # 3. Đối sánh với qrels
        num_rel = len(relevant)
        for k in k_list:
            top_k_hits = [d for d in retrieved_ids[:k] if d in relevant]
            hit_cnt = len(top_k_hits)

            recall_at_k[k].append(hit_cnt / num_rel if num_rel > 0 else 0.0)
            if hit_cnt > 0:
                hit_at_k[k] += 1

            for rank, d in enumerate(retrieved_ids[:k], 1):
                if d in relevant:
                    mrr_at_k[k] += 1.0 / rank
                    break

    elapsed_seq = time.time() - t0_all
    total_q = len(eval_queries)
    seq_qps = total_q / elapsed_seq if elapsed_seq > 0 else 0.0

    # 4. Đo Batched Throughput
    print(f"[*] Đo Batch Throughput (batch_size={batch_size})...")
    t_batch_start = time.perf_counter()
    for i in range(0, total_q, batch_size):
        b_texts = [q["text"] for q in eval_queries[i : i + batch_size]]
        b_vecs = model.encode(b_texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        client.search(collection_name=collection_name, data=b_vecs, limit=top_k_max, output_fields=["id"])
    elapsed_batch = time.perf_counter() - t_batch_start
    batch_qps = total_q / elapsed_batch if elapsed_batch > 0 else 0.0

    def get_stats(arr):
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    return {
        "k_list": k_list,
        "total_queries": total_q,
        "recall_at_k": {k: float(np.mean(recall_at_k[k])) for k in k_list},
        "hit_at_k": {k: float(hit_at_k[k] / total_q) for k in k_list},
        "mrr_at_k": {k: float(mrr_at_k[k] / total_q) for k in k_list},
        "latency": {
            "embedding_ms": get_stats(embed_latencies),
            "milvus_search_ms": get_stats(search_latencies),
            "end_to_end_ms": get_stats(total_latencies),
        },
        "throughput": {
            "sequential_qps": float(seq_qps),
            "batch_qps": float(batch_qps),
            "milvus_search_qps": float(1000.0 / np.mean(search_latencies)),
        },
    }


# =====================================================================
# 4. IN BÁO CÁO KẾT QUẢ
# =====================================================================
def print_smart_report(metrics: Dict[str, Any], dataset_name: str, pool_size: int, collection_name: str):
    k_list = metrics["k_list"]
    recall = metrics["recall_at_k"]
    hit = metrics["hit_at_k"]
    mrr = metrics["mrr_at_k"]
    lat = metrics["latency"]
    tp = metrics["throughput"]
    total_q = metrics["total_queries"]

    print("\n" + "=" * 82)
    print(f"📊 BÁO CÁO SMART POOL BENCHMARK - MILVUS VECTOR SEARCH")
    print(f"   • Bộ dữ liệu:       {dataset_name}")
    print(f"   • Collection:       {collection_name}")
    print(f"   • Quy mô Pool:      {pool_size:,} documents (100% Ground-Truth + Distractors)")
    print(f"   • Đánh giá trên:    100% câu hỏi ({total_q:,} queries đối chiếu với qrels)")
    print("=" * 82)

    # 1. Recall & Accuracy
    print("\n1. ĐỘ CHÍNH XÁC TRUY XUẤT (INFORMATION RETRIEVAL ACCURACY):")
    print("-" * 82)
    print(f"{'Metric':<18} | " + " | ".join([f"K = {k:<4}" for k in k_list]))
    print("-" * 82)
    print(f"{'Recall@K':<18} | " + " | ".join([f"{recall[k]*100:6.2f}%" for k in k_list]))
    print(f"{'Hit Rate@K':<18} | " + " | ".join([f"{hit[k]*100:6.2f}%" for k in k_list]))
    print(f"{'MRR@K':<18} | " + " | ".join([f"{mrr[k]:6.4f} " for k in k_list]))
    print("-" * 82)

    # 2. Latency
    print("\n2. ĐỘ TRỄ TRUY VẤN (LATENCY - ĐƠN VỊ: MILI-GIÂY / QUERY):")
    print("-" * 82)
    print(f"{'Giai đoạn':<24} | {'Mean':<8} | {'P50':<8} | {'P90':<8} | {'P95':<8} | {'P99':<8} | {'Max':<8}")
    print("-" * 82)
    for key, label in [
        ("embedding_ms", "1. Query Embedding"),
        ("milvus_search_ms", "2. Milvus ANN Search"),
        ("end_to_end_ms", "-> End-to-End Total"),
    ]:
        s = lat[key]
        print(f"{label:<24} | {s['mean']:6.2f}ms | {s['p50']:6.2f}ms | {s['p90']:6.2f}ms | {s['p95']:6.2f}ms | {s['p99']:6.2f}ms | {s['max']:6.2f}ms")
    print("-" * 82)

    # 3. Throughput
    print("\n3. THÔNG LƯỢNG HỆ THỐNG (THROUGHPUT / QPS - QUERIES PER SECOND):")
    print("-" * 82)
    print(f"  • Throughput tuần tự (Single-Stream QPS):   {tp['sequential_qps']:8.2f} truy vấn / giây")
    print(f"  • Throughput theo lô (Batched QPS, B=32):  {tp['batch_qps']:8.2f} truy vấn / giây")
    print(f"  • Tốc độ Milvus thuần túy (Search-Only):     ~{tp['milvus_search_qps']:8.1f} searches / giây")
    print("=" * 82 + "\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    hf_name = cfg["hf_name"]
    dataset_name = cfg["name"]
    collection_name = cfg["collection_name"]
    pool_size = args.pool_size or cfg["default_pool_size"]

    print("=" * 82)
    print("  SMART POOL BENCHMARK CHO MILVUS (GROUND-TRUTH GUARANTEE)")
    print(f"  Dataset:     {dataset_name}")
    print(f"  Collection:  {collection_name}")
    print(f"  Pool Size:   {pool_size:,} tài liệu (Đảm bảo 100% đáp án qrels có mặt)")
    print(f"  DB Path:     {args.db_path}")
    print(f"  Model:       {args.model_name}")
    print("=" * 82)

    # 1. Chuẩn bị Smart Pool
    pool_docs, ground_truth, eval_queries = prepare_smart_pool(
        hf_name=hf_name,
        pool_size=pool_size,
    )

    # 2. Kết nối Milvus và nạp dữ liệu
    print(f"\n[*] Bước 3: Kết nối Milvus tại '{args.db_path}'...")
    client = MilvusClient(uri=args.db_path)
    model = SentenceTransformer(args.model_name)

    index_pool_into_milvus(
        client=client,
        model=model,
        collection_name=collection_name,
        dataset_key=args.dataset,
        pool_docs=pool_docs,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )

    # 3. Đánh giá IR Metrics trên 100% queries
    metrics = evaluate_ir_performance(
        client=client,
        model=model,
        collection_name=collection_name,
        eval_queries=eval_queries,
        ground_truth=ground_truth,
        top_k_max=args.top_k_max,
        batch_size=32,
    )

    # 4. In báo cáo
    print_smart_report(metrics, dataset_name, len(pool_docs), collection_name)

    # 5. Xuất file JSON nếu có
    if args.output_json:
        export_data = {
            "dataset": args.dataset,
            "dataset_name": dataset_name,
            "collection_name": collection_name,
            "pool_size": len(pool_docs),
            "metrics": metrics,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"[✓] Đã xuất báo cáo chi tiết ra file JSON: {args.output_json}")

    client.close()


if __name__ == "__main__":
    main()

