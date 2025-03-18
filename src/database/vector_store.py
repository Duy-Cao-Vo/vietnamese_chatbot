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
            filter_dict = {}
            
            # Nếu có intent, thêm vào truy vấn và bộ lọc
            if intent and intent not in ["greeting", "general"]:
                search_query = f"{intent}: {query}"
                filter_dict = {"metadata_field": {"$eq": intent}}
            
            # Thực hiện tìm kiếm
            results = self.db.similarity_search(
                search_query,
                k=limit,
                filter=filter_dict if filter_dict else None
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
                chunk_size=500,           # Characters per chunk
                chunk_overlap=100,        # Characters overlap between chunks
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Priority order of separators
            )
            
            # For Markdown content:
            # text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100)
            
            chunked_documents = []
            
            # Process each document
            for doc in documents:
                # Get the text and metadata
                content = doc.page_content
                metadata = doc.metadata.copy()
                
                # Split text into chunks
                chunks = text_splitter.split_text(content)
                
                # Create new documents for each chunk
                for i, chunk in enumerate(chunks):
                    # Preserve original metadata and add chunk information
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk": i,
                        "total_chunks": len(chunks),
                        "document_id": metadata.get("source", "") + f"-chunk-{i}"
                    })
                    
                    chunked_doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    chunked_documents.append(chunked_doc)
            
            # Add the chunked documents to the vector store
            self.db.add_documents(chunked_documents)
            self.db.persist()
            logger.info(f"Added {len(chunked_documents)} chunked documents from {len(documents)} original documents")
            return True
        
        except Exception as e:
            logger.error(f"Error adding chunked documents to vector store: {str(e)}")
            return False 