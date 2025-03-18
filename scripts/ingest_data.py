#!/usr/bin/env python3
"""
Script để nhập dữ liệu vào vector database.
"""

import os
import sys
import logging
import argparse
import json
from typing import List, Dict, Any
from langchain.schema import Document

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.database.vector_store import VectorStore

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ingest-data")

def load_text_files(directory: str, intent: str) -> List[Document]:
    """
    Tải các tệp văn bản từ thư mục và chuyển đổi thành Document.
    
    Args:
        directory: Đường dẫn đến thư mục chứa tệp văn bản
        intent: Intent liên quan đến dữ liệu
        
    Returns:
        Danh sách các Document
    """
    documents = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return documents
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Tạo Document
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "filename": filename,
                        "intent": intent
                    }
                )
                documents.append(doc)
                logger.info(f"Loaded document: {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
    
    return documents

def load_inventory_data(file_path: str) -> List[Document]:
    """
    Tải dữ liệu tồn kho từ tệp JSON và chuyển đổi thành Document.
    
    Args:
        file_path: Đường dẫn đến tệp JSON
        
    Returns:
        Danh sách các Document
    """
    documents = []
    
    if not os.path.exists(file_path):
        logger.warning(f"Inventory file not found: {file_path}")
        return documents
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tạo Document tổng quan
        overview_doc = Document(
            page_content=f"Thông tin tồn kho cập nhật ngày: {data.get('last_updated', 'N/A')}",
            metadata={
                "source": file_path,
                "filename": os.path.basename(file_path),
                "intent": "inventory",
                "type": "overview"
            }
        )
        documents.append(overview_doc)
        
        # Tạo Document cho từng sản phẩm
        for product in data.get("products", []):
            product_info = f"Sản phẩm: {product.get('name')} (ID: {product.get('id')})\n"
            product_info += f"Loại: {product.get('category')}\n"
            product_info += f"Giá: {product.get('price')} VND\n"
            product_info += "Kích cỡ và màu sắc có sẵn:\n"
            
            for size, colors in product.get("sizes", {}).items():
                for color, quantity in colors.items():
                    if quantity > 0:
                        product_info += f"- Size {size}, màu {color}: {quantity} sản phẩm\n"
                    else:
                        product_info += f"- Size {size}, màu {color}: Hết hàng\n"
            
            doc = Document(
                page_content=product_info,
                metadata={
                    "source": file_path,
                    "product_id": product.get("id"),
                    "product_name": product.get("name"),
                    "intent": "inventory",
                    "type": "product"
                }
            )
            documents.append(doc)
            logger.info(f"Loaded inventory data for product: {product.get('id')}")
            
    except Exception as e:
        logger.error(f"Error loading inventory data from {file_path}: {str(e)}")
    
    return documents

def ingest_data(clear_existing: bool = False):
    """
    Nhập dữ liệu vào vector database.
    
    Args:
        clear_existing: Xóa dữ liệu hiện có trước khi nhập
    """
    # Khởi tạo vector store
    vector_store = VectorStore()
    
    # Xóa dữ liệu hiện có nếu được yêu cầu
    if clear_existing:
        logger.info("Clearing existing data from vector store...")
        vector_store.clear()
    
    # Tải dữ liệu từ các thư mục
    all_documents = []
    
    # Dữ liệu chính sách
    policy_docs = load_text_files(os.path.join(config.DATA_DIR, "policies"), "policy")
    all_documents.extend(policy_docs)
    logger.info(f"Loaded {len(policy_docs)} policy documents")
    
    # Dữ liệu sản phẩm
    product_docs = load_text_files(os.path.join(config.DATA_DIR, "products"), "product")
    all_documents.extend(product_docs)
    logger.info(f"Loaded {len(product_docs)} product documents")
    
    # Dữ liệu cửa hàng
    store_docs = load_text_files(os.path.join(config.DATA_DIR, "stores"), "store")
    all_documents.extend(store_docs)
    logger.info(f"Loaded {len(store_docs)} store documents")
    
    # Dữ liệu tồn kho
    inventory_docs = load_inventory_data(os.path.join(config.DATA_DIR, "inventory", "inventory.json"))
    all_documents.extend(inventory_docs)
    logger.info(f"Loaded {len(inventory_docs)} inventory documents")
    
    # Thêm dữ liệu vào vector store
    if all_documents:
        logger.info(f"Adding {len(all_documents)} documents to vector store...")
        vector_store.add_documents_with_chunking(all_documents)
        logger.info("Data ingestion completed successfully!")
    else:
        logger.warning("No documents found to ingest.")

def main():
    parser = argparse.ArgumentParser(description="Ingest data into vector database")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingestion")
    
    args = parser.parse_args()
    
    ingest_data(clear_existing=args.clear)

if __name__ == "__main__":
    main() 