import os
import logging
import sys
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger("clothing-chatbot")

class VectorStore:
    """
    Quản lý vector database để lưu trữ và truy xuất dữ liệu cho RAG.
    Sử dụng Chroma DB và mô hình embedding từ HuggingFace.
    """
    
    def __init__(self):
        """Khởi tạo vector store với cấu hình từ tệp config."""
        self.vector_db_path = config.VECTOR_DB_PATH
        self.embedding_model_name = config.EMBEDDING_MODEL_NAME
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Khởi tạo embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Kết nối đến Chroma DB hoặc tạo mới nếu chưa tồn tại
        self.db = None
        try:
            self.db = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embedding_model
            )
            logger.info(f"Vector store loaded from {self.vector_db_path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            # Tạo mới nếu không thể load
            self.db = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.vector_db_path
            )
            logger.info(f"Created new vector store at {self.vector_db_path}")
    
    async def search(self, query: str, intent: str = None, limit: int = 5) -> List[Document]:
        """
        Tìm kiếm các tài liệu liên quan trong vector store.
        
        Args:
            query: Câu truy vấn tìm kiếm
            intent: Ý định để lọc kết quả (tùy chọn)
            limit: Số lượng kết quả tối đa
            
        Returns:
            Danh sách các tài liệu liên quan
        """
        try:
            # Điều chỉnh truy vấn dựa trên intent
            search_query = query
            filter_dict = None
            
            # Extract product ID from query if it exists
            product_id_match = re.search(r'[A-Z]{2,3}\d{3,4}', query)
            product_id = product_id_match.group(0) if product_id_match else None
            
            if product_id:
                logger.info(f"Detected product ID in query: {product_id}")
                # Fix: Use correct ChromaDB filter format with $eq operator
                filter_dict = {"product_id": {"$eq": product_id}}
                search_query = f"product {product_id} information details"
            
            # If intent is provided, include it in the search query
            if intent and intent not in ["greeting", "general"]:
                search_query = f"{intent}: {search_query}"
                
                # Add intent to filter if no product ID filter
                if not product_id:
                    filter_dict = {"intent": {"$eq": intent}}
            
            logger.info(f"Searching with query: '{search_query}', filters: {filter_dict}, limit: {limit}")
            
            # Thực hiện tìm kiếm
            results = self.db.similarity_search(
                search_query,
                k=limit,
                filter=filter_dict
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Thêm tài liệu vào vector store.
        
        Args:
            documents: Danh sách tài liệu cần thêm
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            self.db.add_documents(documents)
            self.db.persist()
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Xóa toàn bộ dữ liệu trong vector store.
        
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            self.db._collection.delete(where={})
            self.db.persist()
            logger.info("Cleared vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def add_documents_with_chunking(self, documents: List[Document]) -> bool:
        """
        Thêm tài liệu vào vector store với chunking để tối ưu hóa.
        
        Args:
            documents: Danh sách tài liệu cần thêm
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            # Choose the appropriate text splitter based on your content
            # For general text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,           # Increased from 500 to 1200 characters per chunk
                chunk_overlap=200,        # Increased from 100 to 200 characters overlap
                length_function=len,
                separators=["\n\n", ". ", " ", ""]  # Priority order of separators
            )
            
            # For Markdown content:
            # text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100)
            
            chunked_documents = []
            
            # Process each document
            for doc in documents:
                # Extract product ID if it exists in content
                product_id = None
                product_name = None
                
                # Check for product ID in the content (format: Mã sản phẩm: XXXXX)
                id_match = re.search(r'Mã\s+sản\s+phẩm:\s+([A-Z0-9]+)', doc.page_content)
                if id_match:
                    product_id = id_match.group(1)
                    
                # Check for product name in the content
                name_match = re.search(r'Tên\s+sản\s+phẩm:\s+(.*?)(?:\n|$|-)', doc.page_content)
                if name_match:
                    product_name = name_match.group(1).strip()
                
                # Split the text into chunks
                chunks = text_splitter.split_text(doc.page_content)
                
                # Create new documents for each chunk
                for i, chunk in enumerate(chunks):
                    # Create a copy of the original metadata
                    chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                    
                    # Add chunk information
                    chunk_metadata["chunk"] = i
                    chunk_metadata["chunk_total"] = len(chunks)
                    
                    # Preserve product ID and name in each chunk's metadata if found
                    if product_id:
                        chunk_metadata["product_id"] = product_id
                    if product_name:
                        chunk_metadata["product_name"] = product_name
                    
                    # Create a new document with the chunk and metadata
                    chunk_doc = Document(page_content=chunk, metadata=chunk_metadata)
                    chunked_documents.append(chunk_doc)
            
            # Log the total number of chunked documents
            logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
            
            # Add the chunked documents to the vector store
            self.add_documents(chunked_documents)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in add_documents_with_chunking: {str(e)}")
            return False 