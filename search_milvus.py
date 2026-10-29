"""
Milvus Multi-Dataset Search & Evaluation Engine
Supports searching across:
  - Zalo AI Legal (zalo_legal)
  - Natural Questions VN (nq_vn)
  - Both datasets simultaneously (cross-collection search)
"""

import argparse
import sys
import time
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from device_utils import add_device_args, load_embedding_model

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DB_PATH = "milvus_data.db"


def parse_args():
    parser = argparse.ArgumentParser(description="Search documents across Milvus collections")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Search query text in Vietnamese",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="both",
        choices=["zalo", "nq", "both"],
        help="Target dataset to search: 'zalo', 'nq', or 'both' (default: both)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top search results per collection (default: 5)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Milvus Lite db file or Milvus Server URI (default: {DEFAULT_DB_PATH})",
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
        "--eval-dataset",
        type=str,
        default="zalo",
        choices=["zalo", "nq"],
        help="Dataset to evaluate: 'zalo' or 'nq' (default: zalo)",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=100,
        help="Number of queries to evaluate (default: 100)",
    )
    add_device_args(parser)
    return parser.parse_args()


class MultiDatasetSearcher:
    def __init__(self, db_path: str, model_name: str, device: str = "auto", fp16: bool = False):
        self.db_path = db_path
        self.model_name = model_name

        self.model = load_embedding_model(model_name, device=device, fp16=fp16)

        print(f"[*] Kết nối Milvus tại: {db_path}...")
        self.client = MilvusClient(uri=db_path)

        self.available_collections = self.client.list_collections()
        print(f"[+] Các collections hiện có trong Milvus: {self.available_collections}")

        # Load collections vào bộ nhớ nếu có
        for col in ["zalo_legal", "nq_vn"]:
            if col in self.available_collections:
                try:
                    self.client.load_collection(col)
                    stats = self.client.get_collection_stats(col)
                    print(f"    • Loaded '{col}' ({stats.get('row_count', 0):,} bản ghi)")
                except Exception as e:
                    print(f"    • Warning loading '{col}': {e}")
        print()

    def search_collection(self, collection_name: str, query_vector: list, top_k: int = 5) -> List[Dict[str, Any]]:
        if collection_name not in self.available_collections:
            return []

        output_fields = ["id", "title", "text"]
        if collection_name == "nq_vn":
            output_fields.extend(["og_title", "og_text"])

        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=output_fields,
            )
        except Exception as e:
            print(f"[!] Lỗi truy vấn '{collection_name}': {e}")
            return []

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                item = {
                    "collection": collection_name,
                    "id": hit.get("id"),
                    "score": round(float(hit.get("distance", 0.0)), 4),
                    "title": entity.get("title", ""),
                    "text": entity.get("text", ""),
                }
                if "og_title" in entity:
                    item["og_title"] = entity.get("og_title", "")
                if "og_text" in entity:
                    item["og_text"] = entity.get("og_text", "")
                hits.append(item)
        return hits

    def search(self, query: str, dataset_choice: str = "both", top_k: int = 5):
        t0 = time.time()
        query_vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        t_embed = time.time() - t0

        t1 = time.time()
        results_by_source = {}

        if dataset_choice in ["zalo", "both"]:
            results_by_source["zalo_legal"] = self.search_collection("zalo_legal", query_vector, top_k)

        if dataset_choice in ["nq", "both"]:
            results_by_source["nq_vn"] = self.search_collection("nq_vn", query_vector, top_k)

        t_search = time.time() - t1
        return results_by_source, t_embed, t_search

    def close(self):
        self.client.close()


def print_results(query: str, results_by_source: Dict[str, List[Dict[str, Any]]], t_embed: float, t_search: float):
    print("=" * 75)
    print(f"🔍 CÂU TRUY VẤN: \"{query}\"")
    print(f"⏱️  Thời gian: Embedding: {t_embed*1000:.1f}ms | Milvus Search: {t_search*1000:.1f}ms | Tổng: {(t_embed + t_search)*1000:.1f}ms")
    print("=" * 75)

    source_names = {
        "zalo_legal": "🏛️  VĂN BẢN PHÁP LUẬT VIỆT NAM (Zalo Legal)",
        "nq_vn": "📚 BÁCH KHOA TOÀN THƯ & TRI THỨC MỞ (NQ-VN)",
    }

    total_hits = sum(len(hits) for hits in results_by_source.values())
    if total_hits == 0:
        print("  Không tìm thấy kết quả nào phù hợp trong các collection được chọn.")
        return

    for col_name, hits in results_by_source.items():
        label = source_names.get(col_name, col_name)
        print(f"\n{label} ({len(hits)} kết quả)")
        print("-" * 75)

        if not hits:
            print("  (Không có dữ liệu trong collection này)")
            continue

        for rank, hit in enumerate(hits, 1):
            doc_id = hit["id"]
            score = hit["score"]
            title = hit["title"].strip()
            text = hit["text"].strip().replace("\n", " ")
            snippet = text[:220] + ("..." if len(text) > 220 else "")

            print(f"[{rank}] Score: {score:.4f} | ID: {doc_id}")
            if title:
                print(f"    Tiêu đề: {title}")
            print(f"    Nội dung: {snippet}")

            if "og_title" in hit and hit["og_title"]:
                og_t = hit["og_title"].strip()
                og_snippet = hit.get("og_text", "").strip()[:150]
                print(f"    Bản gốc EN: {og_t} - \"{og_snippet}...\"")
            print()
    print("=" * 75)


def run_interactive_mode(searcher: MultiDatasetSearcher, top_k: int):
    print("\n" + "=" * 65)
    print("  CHẾ ĐỘ TÌM KIẾM TƯƠNG TÁC (INTERACTIVE MULTI-DATASET CLI)")
    print("  Tìm kiếm ngữ nghĩa đồng thời trên Pháp luật & Bách khoa tri thức.")
    print("  • Gõ ':mode all'  để tìm kiếm cả 2 bộ (mặc định)")
    print("  • Gõ ':mode zalo' để chỉ tìm văn bản pháp luật")
    print("  • Gõ ':mode nq'   để chỉ tìm bách khoa tri thức NQ-VN")
    print("  • Gõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 65 + "\n")

    current_mode = "both"

    while True:
        try:
            mode_tag = "[ALL]" if current_mode == "both" else f"[{current_mode.upper()}]"
            query = input(f"❓ {mode_tag} Nhập câu hỏi: ").strip()
            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("Tạm biệt!")
                break

            if query.lower() == ":mode all":
                current_mode = "both"
                print("-> Chế độ: Tìm kiếm cả 2 bộ dữ liệu\n")
                continue
            elif query.lower() == ":mode zalo":
                current_mode = "zalo"
                print("-> Chế độ: Chỉ tìm kiếm văn bản Pháp luật (Zalo Legal)\n")
                continue
            elif query.lower() == ":mode nq":
                current_mode = "nq"
                print("-> Chế độ: Chỉ tìm kiếm Bách khoa tri thức (NQ-VN)\n")
                continue

            results, t_embed, t_search = searcher.search(query, dataset_choice=current_mode, top_k=top_k)
            print_results(query, results, t_embed, t_search)

        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break


def run_evaluation(searcher: MultiDatasetSearcher, eval_dataset: str, limit: int = 100):
    if eval_dataset == "zalo":
        hf_dataset = "GreenNode/zalo-ai-legal-text-retrieval-vn"
        col_name = "zalo_legal"
        desc = "Zalo AI Legal"
    else:
        hf_dataset = "GreenNode/nq-vn"
        col_name = "nq_vn"
        desc = "Natural Questions VN"

    print("=" * 65)
    print(f"  BENCHMARK EVALUATION: {desc.upper()}")
    print(f"  Collection: {col_name} | Số câu hỏi đánh giá: {limit}")
    print("=" * 65)

    print(f"[*] Đang nạp queries và qrels từ {hf_dataset}...")
    raw_queries = load_dataset(hf_dataset, "queries", split="test")
    raw_qrels = load_dataset(hf_dataset, "qrels", split="test")

    ground_truth = {}
    for item in raw_qrels:
        qid = str(item["query-id"])
        cid = str(item["corpus-id"])
        if qid not in ground_truth:
            ground_truth[qid] = set()
        ground_truth[qid].add(cid)

    eval_queries = []
    for item in raw_queries:
        qid = str(item["id"])
        if qid in ground_truth and len(ground_truth[qid]) > 0:
            eval_queries.append({"id": qid, "text": item["text"]})
        if len(eval_queries) >= limit:
            break

    print(f"[+] Tìm thấy {len(eval_queries)} queries hợp lệ để đánh giá.")

    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    mrr_at_10 = 0.0

    start_time = time.time()
    for q in eval_queries:
        qid = q["id"]
        qtext = q["text"]
        relevant_docs = ground_truth[qid]

        res, _, _ = searcher.search(qtext, dataset_choice=eval_dataset, top_k=10)
        hits = res.get(col_name, [])
        retrieved_ids = [str(h["id"]) for h in hits]

        if any(doc in relevant_docs for doc in retrieved_ids[:1]):
            hit_at_1 += 1
        if any(doc in relevant_docs for doc in retrieved_ids[:5]):
            hit_at_5 += 1
        if any(doc in relevant_docs for doc in retrieved_ids[:10]):
            hit_at_10 += 1

        for rank, doc_id in enumerate(retrieved_ids[:10], 1):
            if doc_id in relevant_docs:
                mrr_at_10 += 1.0 / rank
                break

    n = len(eval_queries)
    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print(f"📊 KẾT QUẢ BENCHMARK ({desc} | N = {n} | Thời gian: {elapsed:.2f}s):")
    print(f"  • Hit@1:   {hit_at_1}/{n} ({hit_at_1/n*100:.2f}%)")
    print(f"  • Hit@5:   {hit_at_5}/{n} ({hit_at_5/n*100:.2f}%)")
    print(f"  • Hit@10:  {hit_at_10}/{n} ({hit_at_10/n*100:.2f}%)")
    print(f"  • MRR@10:  {mrr_at_10/n:.4f}")
    print("=" * 65)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    searcher = MultiDatasetSearcher(
        db_path=args.db_path,
        model_name=args.model_name,
        device=args.device,
        fp16=args.fp16,
    )

    try:
        if args.evaluate:
            run_evaluation(searcher, eval_dataset=args.eval_dataset, limit=args.eval_limit)
        elif args.query:
            results, t_embed, t_search = searcher.search(
                query=args.query,
                dataset_choice=args.dataset,
                top_k=args.top_k,
            )
            print_results(args.query, results, t_embed, t_search)
        else:
            run_interactive_mode(searcher, top_k=args.top_k)
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
