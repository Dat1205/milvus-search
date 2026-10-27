"""
Milvus Search & Evaluation Script
Allows searching documents in Milvus using semantic vector similarity (Cosine).
Supports:
1. Direct command-line query: --query "câu hỏi"
2. Interactive search CLI:    --interactive
3. Benchmark evaluation:      --evaluate (using dataset queries & qrels)
"""

import argparse
import sys
import time
from typing import List, Dict, Any

from datasets import load_dataset
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DB_PATH = "milvus_legal.db"
DEFAULT_COLLECTION = "zalo_legal"


def parse_args():
    parser = argparse.ArgumentParser(description="Search documents in Milvus")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Search query text in Vietnamese",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top search results to return (default: 5)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Milvus Lite db file or Milvus Server URI (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Milvus collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive search prompt in terminal",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run benchmark evaluation on dataset queries and qrels",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=100,
        help="Number of queries to evaluate (default: 100)",
    )
    return parser.parse_args()


class MilvusSearcher:
    def __init__(self, db_path: str, collection_name: str, model_name: str):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name

        print(f"[*] Đang tải mô hình embedding: {model_name}...")
        self.model = SentenceTransformer(model_name)

        print(f"[*] Kết nối Milvus tại: {db_path}...")
        self.client = MilvusClient(uri=db_path)

        if not self.client.has_collection(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' không tồn tại trong Milvus! "
                f"Hãy chạy 'python index_milvus.py' trước để nạp dữ liệu."
            )
        
        # Load collection vào bộ nhớ để tìm kiếm
        print(f"[*] Đang load collection '{collection_name}' vào bộ nhớ...")
        self.client.load_collection(collection_name)
        print(f"[+] Sẵn sàng tìm kiếm trên collection '{collection_name}'!\n")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        t0 = time.time()
        # Tạo embedding cho query (normalize vector để dùng Cosine Similarity)
        query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        t_embed = time.time() - t0

        t1 = time.time()
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["id", "title", "text"],
        )
        t_search = time.time() - t1

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                hits.append({
                    "id": hit.get("id"),
                    "score": hit.get("distance", 0.0),
                    "title": entity.get("title", ""),
                    "text": entity.get("text", ""),
                })

        return hits, t_embed, t_search

    def close(self):
        self.client.close()


def print_results(query: str, hits: List[Dict[str, Any]], t_embed: float, t_search: float):
    print("=" * 70)
    print(f"🔍 CÂU TRUY VẤN: \"{query}\"")
    print(f"⏱️  Thời gian: Embedding {t_embed*1000:.1f}ms | Milvus Search: {t_search*1000:.1f}ms | Tổng: {(t_embed + t_search)*1000:.1f}ms")
    print(f"📊 Tìm thấy: {len(hits)} kết quả phù hợp nhất")
    print("=" * 70)

    if not hits:
        print("  Không tìm thấy kết quả nào phù hợp.")
        return

    for rank, hit in enumerate(hits, 1):
        doc_id = hit["id"]
        score = hit["score"]
        title = hit["title"].strip()
        text = hit["text"].strip().replace("\n", " ")
        snippet = text[:200] + ("..." if len(text) > 200 else "")

        print(f"\n[{rank}] Độ tương đồng: {score:.4f} | ID: {doc_id}")
        if title:
            print(f"    Tiêu đề: {title}")
        print(f"    Trích dẫn: {snippet}")
    print("-" * 70)


def run_interactive_mode(searcher: MilvusSearcher, top_k: int):
    print("\n" + "=" * 60)
    print("  CHẾ ĐỘ TÌM KIẾM TƯƠNG TÁC (INTERACTIVE CLI)")
    print("  Nhập câu hỏi bằng tiếng Việt để tìm kiếm tài liệu.")
    print("  Gõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("❓ Nhập câu hỏi tìm kiếm: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Tạm biệt!")
                break

            hits, t_embed, t_search = searcher.search(query, top_k=top_k)
            print_results(query, hits, t_embed, t_search)
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break


def run_evaluation(searcher: MilvusSearcher, limit: int = 100):
    print("=" * 60)
    print("  ĐÁNH GIÁ CHẤT LƯỢNG TÌM KIẾM TRÊN BENCHMARK (Zalo Legal)")
    print(f"  Số lượng queries đánh giá: {limit}")
    print("=" * 60)

    # 1. Nạp queries và qrels
    print("[*] Nạp queries và qrels từ Hugging Face...")
    raw_queries = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", "queries", split="test")
    raw_qrels = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", "qrels", split="test")

    # Map qrels: query_id -> set of relevant doc_ids
    ground_truth = {}
    for item in raw_qrels:
        qid = str(item["query-id"])
        cid = str(item["corpus-id"])
        if qid not in ground_truth:
            ground_truth[qid] = set()
        ground_truth[qid].add(cid)

    # Lọc các query có ground truth
    eval_queries = []
    for item in raw_queries:
        qid = str(item["id"])
        if qid in ground_truth and len(ground_truth[qid]) > 0:
            eval_queries.append({"id": qid, "text": item["text"]})
        if len(eval_queries) >= limit:
            break

    print(f"[+] Số query hợp lệ để đánh giá: {len(eval_queries)}")

    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    mrr_at_10 = 0.0

    start_time = time.time()
    for idx, q in enumerate(eval_queries):
        qid = q["id"]
        qtext = q["text"]
        relevant_docs = ground_truth[qid]

        hits, _, _ = searcher.search(qtext, top_k=10)
        retrieved_ids = [str(h["id"]) for h in hits]

        # Kiểm tra Hit@1, Hit@5, Hit@10
        if any(doc in relevant_docs for doc in retrieved_ids[:1]):
            hit_at_1 += 1
        if any(doc in relevant_docs for doc in retrieved_ids[:5]):
            hit_at_5 += 1
        if any(doc in relevant_docs for doc in retrieved_ids[:10]):
            hit_at_10 += 1

        # MRR@10
        for rank, doc_id in enumerate(retrieved_ids[:10], 1):
            if doc_id in relevant_docs:
                mrr_at_10 += 1.0 / rank
                break

    n = len(eval_queries)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ (N = {n} queries | Thời gian: {elapsed:.2f}s):")
    print(f"  • Hit@1:   {hit_at_1}/{n} ({hit_at_1/n*100:.2f}%)")
    print(f"  • Hit@5:   {hit_at_5}/{n} ({hit_at_5/n*100:.2f}%)")
    print(f"  • Hit@10:  {hit_at_10}/{n} ({hit_at_10/n*100:.2f}%)")
    print(f"  • MRR@10:  {mrr_at_10/n:.4f}")
    print("=" * 60)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    searcher = MilvusSearcher(
        db_path=args.db_path,
        collection_name=args.collection_name,
        model_name=args.model_name,
    )

    try:
        if args.evaluate:
            run_evaluation(searcher, limit=args.eval_limit)
        elif args.query:
            hits, t_embed, t_search = searcher.search(args.query, top_k=args.top_k)
            print_results(args.query, hits, t_embed, t_search)
        else:
            run_interactive_mode(searcher, top_k=args.top_k)
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
