from datasets import load_dataset

# 1. Tải bộ dữ liệu zalo-ai-legal-text-retrieval-vn
zalo_corpus = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", "corpus")
zalo_queries = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", "queries")
zalo_qrels = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", "qrels")

# 2. Tải bộ dữ liệu nq-vn
nq_corpus = load_dataset("GreenNode/nq-vn", "corpus")
nq_queries = load_dataset("GreenNode/nq-vn", "queries")
nq_qrels = load_dataset("GreenNode/nq-vn", "qrels")

print(zalo_corpus)