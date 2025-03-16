#!/usr/bin/env python3
"""
Script để tải mô hình DeepSeek R1 từ Hugging Face Hub.
"""

import os
import sys
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("download-model")

def download_model(model_name: str = None, output_dir: str = None, use_cuda: bool = True):
    """
    Tải mô hình từ Hugging Face Hub.
    
    Args:
        model_name: Tên mô hình trên Hugging Face Hub
        output_dir: Thư mục đầu ra để lưu mô hình
        use_cuda: Sử dụng CUDA/MPS nếu có
    """
    # Sử dụng giá trị mặc định từ config nếu không được cung cấp
    model_name = model_name or config.LLM_MODEL_NAME
    output_dir = output_dir or config.LLM_MODEL_PATH
    
    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Kiểm tra xem mô hình đã tồn tại chưa
    if os.path.exists(output_dir) and os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        logger.info(f"Model already exists at {output_dir}. Skipping download.")
        return
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Tải tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tải mô hình
        logger.info("Downloading model...")
        
        # Detect the available device (support for M1 Mac)
        if torch.backends.mps.is_available() and use_cuda:
            device = "mps"
            logger.info("Using Apple Silicon (M1/M2) GPU via MPS")
        elif torch.cuda.is_available() and use_cuda:
            device = "cuda"
            logger.info("Using NVIDIA GPU via CUDA")
        else:
            device = "cpu"
            logger.info("Using CPU (no GPU acceleration)")
        
        # Use half-precision for GPU, full precision for CPU
        dtype = torch.float16 if device != "cpu" else torch.float32
        
        # Load the model
        offload_folder = os.path.join(output_dir, "offload")
        os.makedirs(offload_folder, exist_ok=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device != "cpu" else None,
            offload_folder=offload_folder if device != "cpu" else None,
            trust_remote_code=True
        )
        
        # Lưu mô hình và tokenizer
        logger.info(f"Saving model to {output_dir}...")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        
        logger.info("Model downloaded and saved successfully!")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download LLM model from Hugging Face Hub")
    parser.add_argument("--model", type=str, help="Model name on Hugging Face Hub")
    parser.add_argument("--output", type=str, help="Output directory to save the model")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    download_model(
        model_name=args.model,
        output_dir=args.output,
        use_cuda=not args.cpu
    )

if __name__ == "__main__":
    main() 