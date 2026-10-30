"""
BENCHMARK FULL CORPUS, QUERIES & QRELS - MILVUS VECTOR SEARCH
Đo lường chính xác tuyệt đối trên 100% dữ liệu thực nghiệm:
  • 100% Corpus:   Zalo Legal (61.425 docs) & NQ-VN (2.681.468 docs)
  • 100% Queries:  Zalo Legal (818 queries) & NQ-VN (1.900 queries)
  • 100% Qrels:    Đối chiếu toàn bộ nhãn ground-truth chuẩn
"""

import argparse
import sys
import time
import json
from typing import List, Dict, Set, Any, Optional
import numpy as np

import torch
from datasets import load_dataset
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from device_utils import add_device_args, load_embedding_model, get_device_info

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DB_PATH = "milvus_data.db"

DATASET_CONFIGS = {
    "zalo": {
        "hf_name": "GreenNode/zalo-ai-legal-text-retrieval-vn",
        "collection_name": "zalo_legal",
        "name": "Zalo AI Legal Text Retrieval (Pháp luật Việt Nam)",
        "total_corpus": 61425,
        "total_queries": 818,
        "fields": ["id", "title", "text"],
    },
    "nq": {
        "hf_name": "GreenNode/nq-vn",
        "collection_name": "nq_vn",
        "name": "Natural Questions VN (Bách khoa Tri thức)",
        "total_corpus": 2681468,
        "total_queries": 1900,
        "fields": ["id", "title", "text", "og_title", "og_text"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Milvus trên 100% Full Corpus, Queries và Qrels")
    parser.add_argument(
        "--dataset",
        type=str,
        default="zalo",
        choices=["zalo", "nq", "both"],
        help="Bộ dữ liệu benchmark: 'zalo' (61.4k docs), 'nq' (2.68M docs), hoặc 'both' (mặc định: zalo)",
    )
    parser.add_argument(
        "--auto-index-full",
        action="store_true",
        default=True,
        help="Tự động nạp nốt các tài liệu còn thiếu vào Milvus để đạt 100%% corpus (mặc định: True)",
    )
    parser.add_argument(
        "--batch-size-index",
        type=int,
        default=256,
        help="Kích thước batch tính embedding và nạp vào Milvus (GPU khuyến nghị: 256)",
    )
    parser.add_argument(
        "--batch-size-eval",
        type=int,
        default=32,
        help="Kích thước batch đo batched QPS (mặc định: 32)",
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
        "--output-json",
        type=str,
        default="benchmark_full_results.json",
        help="Đường dẫn lưu kết quả ra file JSON (mặc định: benchmark_full_results.json)",
    )
    add_device_args(parser)
    return parser.parse_args()


# =====================================================================
# 1. HÀM NẠP ĐẦY ĐỦ 100% CORPUS VÀO MILVUS (HỖ TRỢ RESUME TỰ ĐỘNG)
# =====================================================================
def ensure_full_corpus_indexed(
    client: MilvusClient,
    model: SentenceTransformer,
    dataset_key: str,
    batch_size: int = 256,
):
    cfg = DATASET_CONFIGS[dataset_key]
    col_name = cfg["collection_name"]
    hf_name = cfg["hf_name"]
    total_docs = cfg["total_corpus"]
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()

    current_rows = 0
    if client.has_collection(col_name):
        client.load_collection(col_name)
        current_rows = client.get_collection_stats(col_name).get("row_count", 0)

    if current_rows >= total_docs:
        print(f"[✓] Collection '{col_name}' đã có đủ 100% corpus ({current_rows:,}/{total_docs:,} tài liệu). Sẵn sàng benchmark!")
        return

    print(f"\n[*] Tiến hành nạp dữ liệu vào '{col_name}' ({current_rows:,} -> {total_docs:,} tài liệu)...")

    # Tạo schema nếu collection chưa tồn tại
    if not client.has_collection(col_name):
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
        client.create_collection(collection_name=col_name, schema=schema, index_params=index_params)

    # Đọc corpus từ Hugging Face
    print(f"[*] Đang tải corpus từ '{hf_name}'...")
    raw_ds = load_dataset(hf_name, "corpus")
    split = "test" if "test" in raw_ds else list(raw_ds.keys())[0]
    corpus = raw_ds[split]

    start_idx = current_rows
    remaining = total_docs - start_idx
    print(f"[*] Bắt đầu tính embedding GPU và nạp {remaining:,} tài liệu còn lại (batch_size={batch_size})...")

    t0 = time.time()
    with tqdm(total=total_docs, initial=start_idx, desc=f"Nạp {col_name}", unit="doc") as pbar:
        for i in range(start_idx, total_docs, batch_size):
            batch = corpus[i : i + batch_size]
            ids = [str(x) for x in batch["id"]]
            titles = [str(x or "") for x in batch["title"]]
            texts = [str(x or "") for x in batch["text"]]

            input_texts = []
            for t, c in zip(titles, texts):
                t_str = t.strip()
                c_str = c.strip()[:1000]
                input_texts.append(f"{t_str}. {c_str}" if t_str else c_str)

            embeddings = model.encode(
                input_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            entities = []
            for idx in range(len(ids)):
                item = {
                    "id": ids[idx],
                    "vector": embeddings[idx].tolist(),
                    "title": titles[idx][:2000],
                    "text": texts[idx][:65000],
                }
                if dataset_key == "nq":
                    item["og_title"] = str(batch["og_title"][idx] or "")[:2000]
                    item["og_text"] = str(batch["og_text"][idx] or "")[:65000]
                entities.append(item)

            client.insert(collection_name=col_name, data=entities)
            pbar.update(len(entities))

    elapsed = time.time() - t0
    rate = remaining / elapsed if elapsed > 0 else 0
    print(f"[✓] Nạp hoàn tất 100% corpus ({total_docs:,} docs) trong {elapsed:.2f}s ({rate:.1f} docs/s)!")


# =====================================================================
# 2. ĐÁNH GIÁ TRÊN 100% QUERIES VÀ 100% QRELS
# =====================================================================
def run_full_benchmark(
    client: MilvusClient,
    model: SentenceTransformer,
    dataset_key: str,
    top_k_max: int = 50,
    batch_size: int = 32,
) -> Dict[str, Any]:
    cfg = DATASET_CONFIGS[dataset_key]
    col_name = cfg["collection_name"]
    hf_name = cfg["hf_name"]
    dataset_name = cfg["name"]

    print("\n" + "=" * 80)
    print(f"📊 BẮT ĐẦU BENCHMARK 100% FULL CORPUS: {dataset_name.upper()}")
    print("=" * 80)

    client.load_collection(col_name)
    stats = client.get_collection_stats(col_name)
    total_indexed_docs = stats.get("row_count", 0)
    print(f"[+] Collection: '{col_name}' | Tổng số tài liệu trong cơ sở dữ liệu: {total_indexed_docs:,} docs")

    # 1. Tải nhãn qrels và queries
    print(f"[*] Đang tải nhãn qrels từ '{hf_name}'...")
    raw_qrels = load_dataset(hf_name, "qrels", split="test")
    raw_queries = load_dataset(hf_name, "queries", split="test")

    ground_truth: Dict[str, Set[str]] = {}
    for item in raw_qrels:
        qid = str(item["query-id"])
        cid = str(item["corpus-id"])
        score = int(item.get("score", 1))
        if score > 0:
            if qid not in ground_truth:
                ground_truth[qid] = set()
            ground_truth[qid].add(cid)

    # Lọc danh sách câu hỏi hợp lệ có trong qrels
    eval_queries = []
    for q in raw_queries:
        qid = str(q["id"])
        if qid in ground_truth and len(ground_truth[qid]) > 0:
            eval_queries.append({"id": qid, "text": str(q["text"])})

    total_q = len(eval_queries)
    k_list = [1, 3, 5, 10, 20, top_k_max]
    k_list = sorted(list(set([k for k in k_list if k <= top_k_max])))

    print(f"[+] Số câu hỏi đánh giá: {total_q:,} queries đối chiếu với {len(ground_truth):,} qrels mappings.")

    # 2. Thực thi đánh giá từng câu hỏi (Single-Stream Latency & Recall)
    embed_latencies = []
    search_latencies = []
    total_latencies = []

    recall_at_k = {k: [] for k in k_list}
    hit_at_k = {k: 0 for k in k_list}
    mrr_at_k = {k: 0.0 for k in k_list}

    t0_eval = time.time()
    for q in tqdm(eval_queries, desc=f"Benchmark 100% Queries ({col_name})", unit="query"):
        qid = q["id"]
        qtext = q["text"]
        relevant_docs = ground_truth[qid]

        # A. Đo thời gian Embedding câu hỏi
        t0_emb = time.perf_counter()
        q_vec = model.encode([qtext], normalize_embeddings=True)[0].tolist()
        t1_emb = time.perf_counter()
        t_emb = (t1_emb - t0_emb) * 1000.0

        # B. Đo thời gian tìm kiếm ANN trong Milvus
        t0_sea = time.perf_counter()
        res = client.search(
            collection_name=col_name,
            data=[q_vec],
            limit=top_k_max,
            output_fields=["id"]
        )
        t1_sea = time.perf_counter()
        t_sea = (t1_sea - t0_sea) * 1000.0

        embed_latencies.append(t_emb)
        search_latencies.append(t_sea)
        total_latencies.append(t_emb + t_sea)

        retrieved_ids = [str(hit["id"]) for hit in res[0]] if res and len(res) > 0 else []

        # C. Tính toán Recall@K, Hit@K, MRR@K
        num_rel = len(relevant_docs)
        for k in k_list:
            top_k_ids = retrieved_ids[:k]
            hits = [d for d in top_k_ids if d in relevant_docs]
            hit_count = len(hits)
            recall_at_k[k].append(hit_count / num_rel if num_rel > 0 else 0.0)
            if hit_count > 0:
                hit_at_k[k] += 1
            for rank, doc_id in enumerate(top_k_ids, start=1):
                if doc_id in relevant_docs:
                    mrr_at_k[k] += 1.0 / rank
                    break

    elapsed_seq = time.time() - t0_eval
    sequential_qps = total_q / elapsed_seq if elapsed_seq > 0 else 0.0

    # 3. Đo Batched Throughput (QPS)
    print(f"[*] Đo Throughput xử lý theo lô (Batched QPS, batch_size={batch_size})...")
    t0_batch = time.perf_counter()
    for i in range(0, total_q, batch_size):
        b_texts = [q["text"] for q in eval_queries[i : i + batch_size]]
        b_vecs = model.encode(b_texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        client.search(
            collection_name=col_name,
            data=b_vecs,
            limit=top_k_max,
            output_fields=["id"]
        )
    elapsed_batch = time.perf_counter() - t0_batch
    batch_qps = total_q / elapsed_batch if elapsed_batch > 0 else 0.0

    def calc_stats(arr):
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

    metrics = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "collection_name": col_name,
        "total_corpus_indexed": total_indexed_docs,
        "total_queries_evaluated": total_q,
        "k_list": k_list,
        "recall_at_k": {k: float(np.mean(recall_at_k[k])) for k in k_list},
        "hit_at_k": {k: float(hit_at_k[k] / total_q) for k in k_list},
        "mrr_at_k": {k: float(mrr_at_k[k] / total_q) for k in k_list},
        "latency": {
            "embedding_ms": calc_stats(embed_latencies),
            "milvus_search_ms": calc_stats(search_latencies),
            "end_to_end_ms": calc_stats(total_latencies),
        },
        "throughput": {
            "sequential_qps": float(sequential_qps),
            "batch_qps": float(batch_qps),
            "milvus_search_qps": float(1000.0 / np.mean(search_latencies)),
        },
    }

    print_single_full_report(metrics)
    return metrics


def print_single_full_report(m: Dict[str, Any]):
    k_list = m["k_list"]
    recall = m["recall_at_k"]
    hit = m["hit_at_k"]
    mrr = m["mrr_at_k"]
    lat = m["latency"]
    tp = m["throughput"]

    print("\n" + "=" * 82)
    print(f"📋 KẾT QUẢ BENCHMARK 100% FULL CORPUS: {m['dataset_name']}")
    print(f"   • Collection:       {m['collection_name']}")
    print(f"   • Tổng tài liệu:    {m['total_corpus_indexed']:,} documents (100% toàn bộ)")
    print(f"   • Tổng câu hỏi:     {m['total_queries_evaluated']:,} queries (100% đối chiếu qrels)")
    print("=" * 82)

    # 1. Bảng Accuracy
    print("\n1. ĐỘ CHÍNH XÁC TRUY XUẤT (ACCURACY):")
    print("-" * 82)
    print(f"{'Metric':<18} | " + " | ".join([f"K = {k:<4}" for k in k_list]))
    print("-" * 82)
    print(f"{'Recall@K':<18} | " + " | ".join([f"{recall[k]*100:6.2f}%" for k in k_list]))
    print(f"{'Hit Rate@K':<18} | " + " | ".join([f"{hit[k]*100:6.2f}%" for k in k_list]))
    print(f"{'MRR@K':<18} | " + " | ".join([f"{mrr[k]:6.4f} " for k in k_list]))
    print("-" * 82)

    # 2. Bảng Latency
    print("\n2. ĐỘ TRỄ TRUY VẤN (LATENCY - ĐƠN VỊ: MILI-GIÂY):")
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

    # 3. Bảng Throughput
    print("\n3. THÔNG LƯỢNG HỆ THỐNG (THROUGHPUT / QPS):")
    print("-" * 82)
    print(f"  • Throughput tuần tự (Single-Stream QPS):   {tp['sequential_qps']:8.2f} queries / giây")
    print(f"  • Throughput theo lô (Batched QPS, B=32):  {tp['batch_qps']:8.2f} queries / giây")
    print(f"  • Tốc độ Milvus thuần túy (Search-Only):     ~{tp['milvus_search_qps']:8.1f} searches / giây")
    print("=" * 82 + "\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    print("=" * 82)
    print("  🚀 FULL CORPUS BENCHMARK ENGINE CHO MILVUS")
    print(f"  • DB Path:    {args.db_path}")
    print(f"  • Model:      {args.model_name}")
    print(f"  • Device:     {args.device.upper()}")
    print(f"  • Target:     {args.dataset.upper()}")
    print("=" * 82)

    model = load_embedding_model(args.model_name, device=args.device, fp16=args.fp16)
    client = MilvusClient(uri=args.db_path)

    targets = ["zalo", "nq"] if args.dataset == "both" else [args.dataset]
    all_results = {}

    for target in targets:
        # Bước 1: Nạp 100% corpus nếu cần
        if args.auto_index_full:
            ensure_full_corpus_indexed(
                client=client,
                model=model,
                dataset_key=target,
                batch_size=args.batch_size_index,
            )

        # Bước 2: Chạy benchmark trên 100% queries và qrels
        metrics = run_full_benchmark(
            client=client,
            model=model,
            dataset_key=target,
            top_k_max=args.top_k_max,
            batch_size=args.batch_size_eval,
        )
        all_results[target] = metrics

    # Bước 3: Xuất file JSON chi tiết
    json_path = args.output_json or f"benchmark_full_{args.dataset}.json"
    
    export_payload = {
        "benchmark_type": "Full Corpus, Queries & Qrels",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": args.device,
            "device_info": get_device_info(args.device),
            "fp16": args.fp16,
        },
        "model": args.model_name,
        "datasets_evaluated": list(all_results.keys()),
        "results": all_results,
    }

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, ensure_ascii=False, indent=2)
        print(f"\n[✓] ĐÃ XUẤT KẾT QUẢ BENCHMARK FULL RA FILE JSON:")
        print(f"    👉 Đường dẫn: {json_path}")
    except Exception as e:
        print(f"[!] Lỗi khi ghi file JSON: {e}")

    client.close()


if __name__ == "__main__":
    main()

