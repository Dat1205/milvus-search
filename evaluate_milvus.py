"""
Milvus Benchmark & Evaluation Engine
Evaluates Information Retrieval (IR) performance using qrels ground truth:
  1. Recall@K (K = 1, 3, 5, 10, 20, 50, 100) & Hit@K, MRR@K
  2. Latency (Mean, P50, P90, P95, P99, Min, Max) - broken down by Embedding vs Milvus Search
  3. Throughput / QPS (Single-query sequential QPS vs Batched QPS)
"""

import argparse
import sys
import time
import json
from typing import List, Dict, Set, Any
import numpy as np

from datasets import load_dataset
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DB_PATH = "milvus_data.db"

DATASET_CONFIGS = {
    "zalo": {
        "hf_name": "GreenNode/zalo-ai-legal-text-retrieval-vn",
        "collection_name": "zalo_legal",
        "name": "Zalo AI Legal Text Retrieval",
    },
    "nq": {
        "hf_name": "GreenNode/nq-vn",
        "collection_name": "nq_vn",
        "name": "Natural Questions VN",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Milvus IR metrics using qrels")
    parser.add_argument(
        "--dataset",
        type=str,
        default="zalo",
        choices=["zalo", "nq"],
        help="Dataset to evaluate: 'zalo' (Zalo Legal) or 'nq' (NQ-VN) (default: zalo)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Milvus database path or URI (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Milvus collection name (default based on dataset: zalo_legal or nq_vn)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (e.g. 200). Default is all eligible queries.",
    )
    parser.add_argument(
        "--top-k-max",
        type=int,
        default=50,
        help="Maximum Top-K documents to retrieve for evaluation (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for batch throughput (QPS) benchmark (default: 32)",
    )
    parser.add_argument(
        "--in-corpus-only",
        action="store_true",
        help="Evaluate only on queries whose ground truth document exists in the indexed collection",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save evaluation metrics as a JSON file",
    )
    return parser.parse_args()


def load_ground_truth(hf_name: str) -> Dict[str, Set[str]]:
    """Loads qrels and maps query_id -> set of relevant doc_ids"""
    print(f"[*] Đang tải nhãn qrels từ '{hf_name}'...")
    raw_qrels = load_dataset(hf_name, "qrels", split="test")
    ground_truth: Dict[str, Set[str]] = {}
    for item in raw_qrels:
        qid = str(item["query-id"])
        cid = str(item["corpus-id"])
        score = int(item.get("score", 1))
        if score > 0:
            if qid not in ground_truth:
                ground_truth[qid] = set()
            ground_truth[qid].add(cid)
    return ground_truth


def get_indexed_document_ids(client: MilvusClient, collection_name: str) -> Set[str]:
    """Retrieves all indexed primary key IDs in the collection"""
    print(f"[*] Đang kiểm tra các tài liệu đã lập chỉ mục trong '{collection_name}'...")
    try:
        # Milvus client query to get IDs
        records = client.query(
            collection_name=collection_name,
            filter='id != ""',
            output_fields=["id"],
            limit=16384,
        )
        return set(str(r["id"]) for r in records)
    except Exception as e:
        print(f"[!] Warning khi lấy danh sách ID: {e}")
        return set()


def run_benchmark(
    client: MilvusClient,
    model: SentenceTransformer,
    collection_name: str,
    eval_queries: List[Dict[str, str]],
    ground_truth: Dict[str, Set[str]],
    top_k_max: int = 50,
    k_list: List[int] = [1, 3, 5, 10, 20, 50],
    batch_size: int = 32,
):
    total_queries = len(eval_queries)
    print(f"\n[*] Bắt đầu đo lường IR Metrics trên {total_queries} câu hỏi...")
    print(f"[*] K-values: {k_list} (Retrieve Top-{top_k_max})")

    # Arrays for metrics
    embed_latencies_ms = []
    search_latencies_ms = []
    total_latencies_ms = []

    # Per-K accumulators
    recall_at_k = {k: [] for k in k_list}
    hit_at_k = {k: 0 for k in k_list}
    mrr_at_k = {k: 0.0 for k in k_list}

    t_start_total = time.time()

    # 1. Sequential Evaluation (Single query at a time)
    for q in tqdm(eval_queries, desc="Đang đánh giá", unit="query"):
        qid = q["id"]
        qtext = q["text"]
        relevant_docs = ground_truth.get(qid, set())
        if not relevant_docs:
            continue

        # Đo latency Embedding
        t0 = time.perf_counter()
        query_vector = model.encode([qtext], normalize_embeddings=True)[0].tolist()
        t1 = time.perf_counter()
        t_embed_ms = (t1 - t0) * 1000.0

        # Đo latency Milvus Search
        t2 = time.perf_counter()
        res = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k_max,
            output_fields=["id"],
        )
        t3 = time.perf_counter()
        t_search_ms = (t3 - t2) * 1000.0
        t_total_ms = t_embed_ms + t_search_ms

        embed_latencies_ms.append(t_embed_ms)
        search_latencies_ms.append(t_search_ms)
        total_latencies_ms.append(t_total_ms)

        retrieved_ids = []
        if res and len(res) > 0:
            retrieved_ids = [str(hit["id"]) for hit in res[0]]

        # Tính Recall@K, Hit@K, MRR@K
        num_relevant = len(relevant_docs)
        for k in k_list:
            top_k_docs = retrieved_ids[:k]
            hits = [doc for doc in top_k_docs if doc in relevant_docs]
            hit_count = len(hits)

            # Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
            recall = hit_count / num_relevant if num_relevant > 0 else 0.0
            recall_at_k[k].append(recall)

            # Hit@K = 1 if at least 1 relevant doc retrieved
            if hit_count > 0:
                hit_at_k[k] += 1

            # MRR@K = 1 / first_rank
            for rank, doc in enumerate(top_k_docs, 1):
                if doc in relevant_docs:
                    mrr_at_k[k] += 1.0 / rank
                    break

    elapsed_seq_total = time.time() - t_start_total
    seq_qps = total_queries / elapsed_seq_total if elapsed_seq_total > 0 else 0.0

    # 2. Batch Throughput Evaluation (Max QPS)
    print(f"\n[*] Đo lường Throughput xử lý theo Batch (batch_size={batch_size})...")
    batch_latencies_sec = []
    num_batches = (total_queries + batch_size - 1) // batch_size
    t_start_batch = time.perf_counter()

    for i in range(0, total_queries, batch_size):
        batch_queries = [q["text"] for q in eval_queries[i : i + batch_size]]
        t_b0 = time.perf_counter()
        batch_embs = model.encode(batch_queries, normalize_embeddings=True, show_progress_bar=False).tolist()
        client.search(
            collection_name=collection_name,
            data=batch_embs,
            limit=top_k_max,
            output_fields=["id"],
        )
        t_b1 = time.perf_counter()
        batch_latencies_sec.append(t_b1 - t_b0)

    elapsed_batch_total = time.perf_counter() - t_start_batch
    batch_qps = total_queries / elapsed_batch_total if elapsed_batch_total > 0 else 0.0

    # Tính toán các chỉ số thống kê Latency
    def calc_stats(lat_list):
        arr = np.array(lat_list)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    embed_stats = calc_stats(embed_latencies_ms)
    search_stats = calc_stats(search_latencies_ms)
    total_stats = calc_stats(total_latencies_ms)

    # Average metrics
    avg_recall = {k: float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0 for k in k_list}
    hit_rate = {k: float(hit_at_k[k] / total_queries) if total_queries > 0 else 0.0 for k in k_list}
    avg_mrr = {k: float(mrr_at_k[k] / total_queries) if total_queries > 0 else 0.0 for k in k_list}

    return {
        "total_queries": total_queries,
        "k_list": k_list,
        "recall_at_k": avg_recall,
        "hit_at_k": hit_rate,
        "mrr_at_k": avg_mrr,
        "latency": {
            "embedding_ms": embed_stats,
            "milvus_search_ms": search_stats,
            "end_to_end_ms": total_stats,
        },
        "throughput": {
            "sequential_qps": float(seq_qps),
            "batch_qps": float(batch_qps),
            "batch_size": batch_size,
            "sequential_time_s": float(elapsed_seq_total),
            "batch_time_s": float(elapsed_batch_total),
        },
    }


def print_evaluation_report(results: Dict[str, Any], dataset_name: str, collection_name: str, mode_desc: str):
    k_list = results["k_list"]
    recall = results["recall_at_k"]
    hit = results["hit_at_k"]
    mrr = results["mrr_at_k"]
    lat = results["latency"]
    tp = results["throughput"]
    n = results["total_queries"]

    print("\n" + "=" * 78)
    print(f"📊 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG TÌM KIẾM TRÊN MILVUS")
    print(f"   • Dataset:     {dataset_name}")
    print(f"   • Collection:  {collection_name}")
    print(f"   • Chế độ:      {mode_desc}")
    print(f"   • Số câu hỏi:  {n:,} queries đối sánh với qrels ground-truth")
    print("=" * 78)

    # 1. BẢNG RECALL@K, HIT@K, MRR@K
    print("\n1. ĐỘ CHÍNH XÁC TRUY XUẤT (INFORMATION RETRIEVAL ACCURACY):")
    print("-" * 78)
    print(f"{'Metric':<18} | " + " | ".join([f"K = {k:<4}" for k in k_list]))
    print("-" * 78)
    print(f"{'Recall@K':<18} | " + " | ".join([f"{recall[k]*100:6.2f}%" for k in k_list]))
    print(f"{'Hit Rate@K':<18} | " + " | ".join([f"{hit[k]*100:6.2f}%" for k in k_list]))
    print(f"{'MRR@K':<18} | " + " | ".join([f"{mrr[k]:6.4f} " for k in k_list]))
    print("-" * 78)

    # 2. BẢNG ĐỘ TRỄ (LATENCY)
    print("\n2. ĐỘ TRỄ TRUY VẤN (LATENCY BREAKDOWN - ĐƠN VỊ: MILI-GIÂY):")
    print("-" * 78)
    print(f"{'Giai đoạn':<24} | {'Mean':<8} | {'P50':<8} | {'P90':<8} | {'P95':<8} | {'P99':<8} | {'Max':<8}")
    print("-" * 78)
    for key, name in [
        ("embedding_ms", "1. Query Embedding"),
        ("milvus_search_ms", "2. Milvus ANN Search"),
        ("end_to_end_ms", "-> End-to-End Total"),
    ]:
        s = lat[key]
        print(
            f"{name:<24} | {s['mean']:6.2f}ms | {s['p50']:6.2f}ms | {s['p90']:6.2f}ms | {s['p95']:6.2f}ms | {s['p99']:6.2f}ms | {s['max']:6.2f}ms"
        )
    print("-" * 78)

    # 3. BẢNG THROUGHPUT (QPS)
    print("\n3. THÔNG LƯỢNG HỆ THỐNG (THROUGHPUT / QPS - QUERIES PER SECOND):")
    print("-" * 78)
    print(f"  • Throughput tuần tự (Single-Stream QPS):   {tp['sequential_qps']:8.2f} truy vấn / giây")
    print(f"  • Throughput theo lô (Batched QPS, B={tp['batch_size']}):  {tp['batch_qps']:8.2f} truy vấn / giây")
    print(f"  • Tốc độ Milvus thuần túy (Search-Only):     ~{1000.0 / lat['milvus_search_ms']['mean']:8.1f} searches / giây")
    print("=" * 78 + "\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    hf_name = cfg["hf_name"]
    collection_name = args.collection_name or cfg["collection_name"]
    dataset_name = cfg["name"]

    print("=" * 78)
    print("  KHỞI TẠO HỆ THỐNG ĐÁNH GIÁ MILVUS")
    print(f"  Dataset:     {dataset_name} ({hf_name})")
    print(f"  Collection:  {collection_name}")
    print(f"  DB Path:     {args.db_path}")
    print(f"  Model:       {args.model_name}")
    print("=" * 78)

    # 1. Kết nối Milvus
    client = MilvusClient(uri=args.db_path)
    if not client.has_collection(collection_name):
        print(f"[!] Lỗi: Collection '{collection_name}' không tồn tại trong {args.db_path}!")
        print("    Vui lòng chạy 'python index_milvus.py' trước để tạo dữ liệu.")
        sys.exit(1)

    client.load_collection(collection_name)
    stats = client.get_collection_stats(collection_name)
    total_indexed_rows = stats.get("row_count", 0)
    print(f"[+] Đã kết nối collection '{collection_name}' (Hiện có: {total_indexed_rows:,} tài liệu indexed)")

    # 2. Tải mô hình embedding
    print(f"[*] Đang tải mô hình: {args.model_name}...")
    model = SentenceTransformer(args.model_name)

    # 3. Tải queries và qrels
    ground_truth = load_ground_truth(hf_name)
    print(f"[+] Tổng số query có ground-truth trong qrels: {len(ground_truth):,}")

    raw_queries = load_dataset(hf_name, "queries", split="test")

    # 4. Xác định tập query cần đánh giá
    indexed_doc_ids = get_indexed_document_ids(client, collection_name)
    print(f"[+] Số ID tài liệu lấy được từ collection: {len(indexed_doc_ids):,}")

    in_corpus_queries = []
    all_valid_queries = []

    for q in raw_queries:
        qid = str(q["id"])
        qtext = str(q["text"])
        if qid in ground_truth and len(ground_truth[qid]) > 0:
            item = {"id": qid, "text": qtext}
            all_valid_queries.append(item)
            # Kiểm tra xem có ít nhất 1 tài liệu đáp án nằm trong collection đã index không
            if any(doc_id in indexed_doc_ids for doc_id in ground_truth[qid]):
                in_corpus_queries.append(item)

    print(f"[+] Số queries có đáp án nằm trong {total_indexed_rows:,} docs đã index: {len(in_corpus_queries):,}")
    print(f"[+] Tổng số queries hợp lệ trong toàn bộ benchmark: {len(all_valid_queries):,}")

    # Lựa chọn danh sách query
    if args.in_corpus_only or len(in_corpus_queries) == len(all_valid_queries):
        eval_list = in_corpus_queries
        mode_desc = f"In-Corpus Benchmark (Chỉ đánh giá các câu hỏi có tài liệu mục tiêu nằm trong {total_indexed_rows:,} docs đã index)"
    else:
        eval_list = all_valid_queries
        mode_desc = f"Toàn bộ Benchmark ({total_indexed_rows:,} docs đã index / toàn tập)"

    if args.limit_queries and args.limit_queries < len(eval_list):
        eval_list = eval_list[: args.limit_queries]

    # 5. Chạy đánh giá
    k_list = [1, 3, 5, 10, 20, args.top_k_max]
    k_list = sorted(list(set([k for k in k_list if k <= args.top_k_max])))

    results = run_benchmark(
        client=client,
        model=model,
        collection_name=collection_name,
        eval_queries=eval_list,
        ground_truth=ground_truth,
        top_k_max=args.top_k_max,
        k_list=k_list,
        batch_size=args.batch_size,
    )

    # 6. In báo cáo
    print_evaluation_report(results, dataset_name, collection_name, mode_desc)

    # 7. Xuất file JSON nếu có yêu cầu
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[✓] Đã lưu kết quả chi tiết vào file: {args.output_json}")

    client.close()


if __name__ == "__main__":
    main()

