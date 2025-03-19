#!/usr/bin/env python3
"""
Script để kiểm thử RAG trên một sản phẩm cụ thể.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.database.vector_store import VectorStore
from src.llm.model import LLMModel

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test-rag")

async def test_rag_for_product(product_id: str):
    """
    Kiểm thử RAG trực tiếp với một mã sản phẩm cụ thể.
    
    Args:
        product_id: Mã sản phẩm cần kiểm thử
    """
    print(f"\n=== Testing RAG for product ID: {product_id} ===\n")
    
    # Khởi tạo vector store
    vector_store = VectorStore()
    llm_model = LLMModel()
    
    # Tạo truy vấn
    query = f"Tìm kiếm sản phẩm có mã {product_id}"
    print(f"Query: {query}\n")
    
    # Tìm kiếm trực tiếp với product ID
    search_query = f"product {product_id} information details"
    filter_dict = {"product_id": {"$eq": product_id}}
    
    print(f"Search query: {search_query}")
    print(f"Filter: {filter_dict}\n")
    
    try:
        # Thực hiện tìm kiếm
        results = vector_store.db.similarity_search(
            search_query,
            k=5,
            filter=filter_dict
        )
        
        # Hiển thị kết quả
        print(f"Found {len(results)} results:\n")
        for i, doc in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}\n")
        
        # Nếu có kết quả, tạo câu trả lời
        if results:
            context_texts = [doc.page_content for doc in results]
            response = await llm_model.generate_rag_response(query, context_texts)
            print(f"\n=== Generated Response ===\n{response}")
        else:
            print("No results found in vector store.")
            
            # Fallback: search in all documents without filter
            print("\n=== Fallback: Searching without filter ===")
            all_results = vector_store.db.similarity_search(
                search_query,
                k=5
            )
            
            print(f"Found {len(all_results)} results without filter:\n")
            for i, doc in enumerate(all_results):
                print(f"--- Result {i+1} ---")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}\n")
    
    except Exception as e:
        print(f"Error during search: {str(e)}")
        
        # Try a simpler query without filter
        print("\n=== Trying a simpler query without filter ===")
        try:
            simple_results = vector_store.db.similarity_search(
                product_id,
                k=5
            )
            
            print(f"Found {len(simple_results)} results with simple query:\n")
            for i, doc in enumerate(simple_results):
                print(f"--- Result {i+1} ---")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}\n")
                
        except Exception as e2:
            print(f"Error during simple search: {str(e2)}")

async def main():
    # Kiểm thử một số mã sản phẩm
    await test_rag_for_product("AK017")
    
    print("\n" + "="*50 + "\n")
    
    # Thử với một sản phẩm khác để so sánh
    await test_rag_for_product("ASM001")

if __name__ == "__main__":
    asyncio.run(main()) 