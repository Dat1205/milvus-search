"""
Milvus Multi-Dataset Indexing Script
Indexes text embeddings for:
  1. Zalo AI Legal (GreenNode/zalo-ai-legal-text-retrieval-vn)
  2. NQ-VN (GreenNode/nq-vn)
into Milvus Vector Database (Milvus Lite or Milvus Standalone)
"""

import argparse
import sys
import os
import time
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DB_PATH = "milvus_data.db"

DATASETS_CONFIG = {
    "zalo": {
        "hf_name": "GreenNode/zalo-ai-legal-text-retrieval-vn",
        "collection_name": "zalo_legal",
        "description": "Zalo AI Legal Text Retrieval (Pháp luật Việt Nam)",
        "fields": ["id", "title", "text"],
        "default_limit": 5000,
    },
    "nq": {
        "hf_name": "GreenNode/nq-vn",
        "collection_name": "nq_vn",
        "description": "Natural Questions VN (Tri thức bách khoa mở)",
        "fields": ["id", "title", "text", "og_title", "og_text"],
        "default_limit": 5000,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Index datasets into Milvus")
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["zalo", "nq", "both"],
        help="Dataset to index: 'zalo', 'nq', or 'both' (default: both)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Global limit for number of documents per dataset (overrides per-dataset limit). None = index all or use defaults.",
    )
    parser.add_argument(
        "--limit-zalo",
        type=int,
        default=None,
        help="Limit number of documents for Zalo Legal dataset (e.g., 5000 or 61425 for all)",
    )
    parser.add_argument(
        "--limit-nq",
        type=int,
        default=5000,
        help="Limit number of documents for NQ-VN dataset (default: 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation and insertion (default: 64)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Milvus Lite db file path or Milvus Server URI (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model for embeddings (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop existing collection and recreate from scratch",
    )
    return parser.parse_args()


def create_collection_for_dataset(
    client: MilvusClient,
    collection_name: str,
    dataset_key: str,
    dimension: int,
    recreate: bool = False,
):
    if client.has_collection(collection_name):
        if recreate:
            print(f"[*] Đang xóa collection cũ '{collection_name}'...")
            client.drop_collection(collection_name)
        else:
            stats = client.get_collection_stats(collection_name)
            row_count = stats.get("row_count", 0)
            if row_count > 0:
                print(f"[+] Collection '{collection_name}' đã có sẵn {row_count:,} bản ghi. (Dùng --recreate nếu muốn nạp lại từ đầu).")
                return False
            else:
                print(f"[+] Collection '{collection_name}' đã tồn tại (chưa có bản ghi). Tiếp tục nạp dữ liệu...")
                return True

    print(f"[*] Đang tạo collection '{collection_name}' (dim={dimension}, metric=COSINE)...")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

    if dataset_key == "nq":
        schema.add_field(field_name="og_title", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="og_text", datatype=DataType.VARCHAR, max_length=65535)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="AUTOINDEX"
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print(f"[+] Đã tạo collection '{collection_name}' thành công!")
    return True


def index_single_dataset(
    client: MilvusClient,
    model: SentenceTransformer,
    dataset_key: str,
    limit: Optional[int],
    batch_size: int,
    recreate: bool = False,
):
    config = DATASETS_CONFIG[dataset_key]
    hf_name = config["hf_name"]
    collection_name = config["collection_name"]
    desc = config["description"]
    dimension = model.get_sentence_embedding_dimension()

    print("\n" + "=" * 65)
    print(f"  BẮT ĐẦU NẠP DỮ LIỆU: {desc.upper()}")
    print(f"  • Hugging Face Hub: {hf_name}")
    print(f"  • Collection Milvus: {collection_name}")
    print("=" * 65)

    should_index = create_collection_for_dataset(
        client, collection_name, dataset_key, dimension, recreate=recreate
    )
    if not should_index:
        return

    # Tải dữ liệu từ cache Hugging Face
    print(f"[*] Đang đọc corpus từ '{hf_name}'...")
    raw_ds = load_dataset(hf_name, "corpus")
    split_name = "test" if "test" in raw_ds else list(raw_ds.keys())[0]
    corpus = raw_ds[split_name]
    total_docs = len(corpus)
    print(f"[+] Tổng số tài liệu trong tập dữ liệu gốc: {total_docs:,}")

    if limit and limit < total_docs:
        docs_to_index = limit
        corpus_subset = corpus.select(range(docs_to_index))
    else:
        docs_to_index = total_docs
        corpus_subset = corpus

    print(f"[*] Đang xử lý embedding và nạp {docs_to_index:,} tài liệu vào '{collection_name}'...")
    start_time = time.time()

    with tqdm(total=docs_to_index, desc=f"Indexing {collection_name}", unit="doc") as pbar:
        for i in range(0, docs_to_index, batch_size):
            batch = corpus_subset[i : i + batch_size]
            ids = [str(x) for x in batch["id"]]
            titles = [str(x or "") for x in batch["title"]]
            texts = [str(x or "") for x in batch["text"]]

            if dataset_key == "nq":
                og_titles = [str(x or "") for x in batch.get("og_title", [""] * len(ids))]
                og_texts = [str(x or "") for x in batch.get("og_text", [""] * len(ids))]

            # Chuẩn bị input text để tính embedding (kết hợp title + text)
            input_texts = []
            for t, c in zip(titles, texts):
                t_str = t.strip()
                c_str = c.strip()[:1000]
                if t_str:
                    input_texts.append(f"{t_str}. {c_str}")
                else:
                    input_texts.append(c_str)

            embeddings = model.encode(
                input_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            entities = []
            for idx in range(len(ids)):
                entity = {
                    "id": ids[idx],
                    "vector": embeddings[idx].tolist(),
                    "title": titles[idx][:2000],
                    "text": texts[idx][:65000],
                }
                if dataset_key == "nq":
                    entity["og_title"] = og_titles[idx][:2000]
                    entity["og_text"] = og_texts[idx][:65000]
                entities.append(entity)

            client.insert(collection_name=collection_name, data=entities)
            pbar.update(len(entities))

    elapsed = time.time() - start_time
    rate = docs_to_index / elapsed if elapsed > 0 else 0
    print(f"[✓] Hoàn thành nạp '{collection_name}': {docs_to_index:,} docs trong {elapsed:.2f}s ({rate:.1f} docs/s)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    print("=" * 65)
    print(f"  MILVUS DUAL DATASET INDEXER")
    print(f"  DB Path:    {args.db_path}")
    print(f"  Model:      {args.model_name}")
    print(f"  Target:     {args.dataset.upper()}")
    print("=" * 65)

    print(f"[*] Đang tải mô hình SentenceTransformer: {args.model_name}...")
    model = SentenceTransformer(args.model_name)
    print(f"[+] Mô hình sẵn sàng (Vector Dim = {model.get_sentence_embedding_dimension()})")

    print(f"[*] Kết nối Milvus tại: {args.db_path}...")
    client = MilvusClient(uri=args.db_path)

    # Xác định danh sách dataset cần nạp
    if args.dataset == "both":
        targets = ["zalo", "nq"]
    else:
        targets = [args.dataset]

    for target in targets:
        # Xác định số lượng tài liệu cần nạp
        if args.limit is not None:
            doc_limit = args.limit
        elif target == "zalo":
            doc_limit = args.limit_zalo
        elif target == "nq":
            doc_limit = args.limit_nq
        else:
            doc_limit = None

        index_single_dataset(
            client=client,
            model=model,
            dataset_key=target,
            limit=doc_limit,
            batch_size=args.batch_size,
            recreate=args.recreate,
        )

    print("\n" + "=" * 65)
    print("📊 TỔNG KẾT TRẠNG THÁI CÁC COLLECTIONS TRONG MILVUS:")
    available_cols = client.list_collections()
    for col in available_cols:
        stats = client.get_collection_stats(col)
        print(f"  • {col:15s}: {stats.get('row_count', 0):,} bản ghi")
    print("=" * 65)

    client.close()


if __name__ == "__main__":
    main()
