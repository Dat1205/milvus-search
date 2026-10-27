"""
Milvus Indexing Script for Datasets (Zalo Legal / NQ-VN)
Indexes text embeddings into Milvus (Milvus Lite or Milvus Standalone)
"""

import argparse
import sys
import os
import time
from typing import List, Dict, Any

from datasets import load_dataset
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DB_PATH = "milvus_legal.db"


def parse_args():
    parser = argparse.ArgumentParser(description="Index dataset into Milvus")
    parser.add_argument(
        "--dataset",
        type=str,
        default="zalo",
        choices=["zalo", "nq"],
        help="Dataset to index: 'zalo' (Zalo AI Legal) or 'nq' (Natural Questions VN)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to index (e.g., 5000). None = index all.",
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
        "--collection-name",
        type=str,
        default=None,
        help="Collection name in Milvus. Default is 'zalo_legal' or 'nq_vn'",
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


def get_dataset_info(dataset_choice: str):
    if dataset_choice == "zalo":
        dataset_name = "GreenNode/zalo-ai-legal-text-retrieval-vn"
        default_collection = "zalo_legal"
    else:
        dataset_name = "GreenNode/nq-vn"
        default_collection = "nq_vn"
    return dataset_name, default_collection


def init_milvus_collection(
    client: MilvusClient,
    collection_name: str,
    dimension: int,
    recreate: bool = False,
):
    if client.has_collection(collection_name):
        if recreate:
            print(f"[*] Đang xóa collection cũ '{collection_name}'...")
            client.drop_collection(collection_name)
        else:
            print(f"[+] Collection '{collection_name}' đã tồn tại.")
            return

    print(f"[*] Đang tạo collection '{collection_name}' (dimension={dimension}, metric=COSINE)...")
    
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

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
    print(f"[+] Tạo collection '{collection_name}' thành công!")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    dataset_name, default_collection = get_dataset_info(args.dataset)
    collection_name = args.collection_name or default_collection

    print("=" * 60)
    print(f"  MILVUS DATA INDEXING")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Collection: {collection_name}")
    print(f"  DB Path:    {args.db_path}")
    print(f"  Model:      {args.model_name}")
    print(f"  Batch size: {args.batch_size}")
    if args.limit:
        print(f"  Limit:      {args.limit} docs")
    else:
        print(f"  Limit:      Tất cả dữ liệu")
    print("=" * 60)

    # 1. Tải mô hình embedding
    print(f"[*] Đang tải mô hình embedding: {args.model_name}...")
    model = SentenceTransformer(args.model_name)
    dimension = model.get_sentence_embedding_dimension()
    print(f"[+] Mô hình sẵn sàng! Kích thước vector: {dimension}")

    # 2. Khởi tạo Milvus Client
    print(f"[*] Kết nối Milvus tại: {args.db_path}...")
    client = MilvusClient(uri=args.db_path)
    init_milvus_collection(client, collection_name, dimension, recreate=args.recreate)

    # 3. Tải corpus từ HuggingFace cache
    print(f"[*] Đang nạp dữ liệu corpus từ '{dataset_name}'...")
    raw_dataset = load_dataset(dataset_name, "corpus")
    # lấy split test (hoặc train nếu có)
    split_name = "test" if "test" in raw_dataset else list(raw_dataset.keys())[0]
    corpus = raw_dataset[split_name]
    total_docs = len(corpus)
    print(f"[+] Tổng số tài liệu trong corpus: {total_docs:,}")

    if args.limit and args.limit < total_docs:
        docs_to_index = args.limit
        corpus_subset = corpus.select(range(docs_to_index))
    else:
        docs_to_index = total_docs
        corpus_subset = corpus

    print(f"[*] Bắt đầu tạo embedding và nạp {docs_to_index:,} tài liệu vào Milvus...")
    batch_size = args.batch_size
    start_time = time.time()

    # Xử lý theo từng batch
    total_batches = (docs_to_index + batch_size - 1) // batch_size
    with tqdm(total=docs_to_index, desc="Indexing", unit="doc") as pbar:
        for i in range(0, docs_to_index, batch_size):
            batch_slice = corpus_subset[i : i + batch_size]
            ids = [str(x) for x in batch_slice["id"]]
            titles = [str(x or "") for x in batch_slice["title"]]
            texts = [str(x or "") for x in batch_slice["text"]]

            # Chuẩn bị nội dung để encode: kết hợp title + text để semantic search tốt hơn
            input_texts = []
            for t, c in zip(titles, texts):
                if t and t.strip():
                    input_texts.append(f"{t.strip()}. {c.strip()[:1000]}")
                else:
                    input_texts.append(c.strip()[:1000])

            # Tính vector embedding
            embeddings = model.encode(
                input_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            # Tạo danh sách entities nạp vào Milvus
            entities = []
            for doc_id, title, text, emb in zip(ids, titles, texts, embeddings):
                # Cắt ngắn title và text nếu vượt quá max_length của VARCHAR
                safe_title = title[:2000]
                safe_text = text[:65000]
                entities.append({
                    "id": doc_id,
                    "vector": emb.tolist(),
                    "title": safe_title,
                    "text": safe_text,
                })

            client.insert(collection_name=collection_name, data=entities)
            pbar.update(len(entities))

    elapsed = time.time() - start_time
    rate = docs_to_index / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print(f"[✓] ĐÃ NẠP THÀNH CÔNG VÀO MILVUS!")
    print(f"    - Tổng số documents: {docs_to_index:,}")
    print(f"    - Thời gian thực thi: {elapsed:.2f} giây")
    print(f"    - Tốc độ trung bình:   {rate:.1f} docs/giây")
    print(f"    - File lưu trữ:        {args.db_path}")
    print(f"    - Collection:          {collection_name}")
    print("=" * 60)

    # Hiển thị số lượng records trong collection
    stats = client.get_collection_stats(collection_name=collection_name)
    print(f"[i] Collection stats: {stats}")
    client.close()


if __name__ == "__main__":
    main()

