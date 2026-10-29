"""
Milvus & PyTorch Device Management Utilities
Auto-detects CUDA / GPU, configures precision (FP32/FP16), and provides device diagnostics.
"""

import argparse
import sys
from typing import Optional
import torch
from sentence_transformers import SentenceTransformer


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def resolve_device(device_arg: str = "auto") -> str:
    """
    Xác định thiết bị chạy ('cuda' hoặc 'cpu').
    Nếu device_arg là 'auto', tự động chọn 'cuda' nếu có GPU NVIDIA khả dụng.
    """
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    device_arg = device_arg.lower()
    if device_arg in ["cuda", "gpu"]:
        if not torch.cuda.is_available():
            print("[!] CẢNH BÁO: Bạn yêu cầu CUDA nhưng hệ thống chưa phát hiện GPU hỗ trợ PyTorch CUDA. Chuyển về CPU.")
            return "cpu"
        return "cuda"
    
    return "cpu"


def get_device_info(device: str) -> str:
    """Trả về chuỗi thông tin chi tiết về thiết bị đang sử dụng."""
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        cuda_ver = torch.version.cuda or "N/A"
        return f"CUDA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM, CUDA {cuda_ver})"
    else:
        return "CPU"


def log_device_info(device: str):
    """In thông tin phần cứng thiết bị được sử dụng."""
    info = get_device_info(device)
    if "CUDA" in info:
        print(f"[+] [GPU] Thiết bị tăng tốc: {info}")
    else:
        print(f"[i] Thiết bị tính toán: {info}")


def add_device_args(parser: argparse.ArgumentParser):
    """Bổ sung các tham số --device và --fp16 vào argparse."""
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị tính toán embedding: 'auto' (ưu tiên GPU nếu có), 'cuda', hoặc 'cpu' (mặc định: auto)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Sử dụng nửa độ chính xác (Half Precision - FP16) trên GPU để tăng tốc độ và tiết kiệm VRAM",
    )


def load_embedding_model(
    model_name: str,
    device: str = "auto",
    fp16: bool = False,
    warmup: bool = True
) -> SentenceTransformer:
    """
    Tải mô hình SentenceTransformer với thiết bị tối ưu (GPU/CPU) và FP16 (nếu được kích hoạt).
    Tự động fallback an toàn về CPU nếu GPU gặp lỗi phần cứng/driver.
    """
    target_device = resolve_device(device)
    log_device_info(target_device)

    print(f"[*] Đang tải mô hình: {model_name}...")
    try:
        model = SentenceTransformer(model_name, device=target_device)
        if target_device == "cuda":
            if fp16:
                print("[+] Kích hoạt chế độ FP16 (Half Precision) để tối ưu tốc độ inference GPU.")
                model.half()
            
            # Warm-up GPU kernel để kiểm tra thực thi & tránh overhead trễ ở lần suy luận đầu tiên
            if warmup:
                _ = model.encode(["Khởi động CUDA kernel."], show_progress_bar=False, normalize_embeddings=True)
                print("[+] CUDA kernel khởi động thành công!")
    except Exception as e:
        if target_device == "cuda":
            print(f"[!] CẢNH BÁO: Không thể khởi chạy mô hình trên CUDA ({e}).")
            print("[*] Đang tự động chuyển sang chế độ CPU an toàn...")
            target_device = "cpu"
            model = SentenceTransformer(model_name, device="cpu")
        else:
            raise e

    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    print(f"[+] Mô hình sẵn sàng trên {target_device.upper()} (Vector Dim = {dim})")
    return model

