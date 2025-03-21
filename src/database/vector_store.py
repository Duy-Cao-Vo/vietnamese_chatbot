import os
import asyncio
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
import tiktoken

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
    
    def _extract_product_attributes(self, query: str) -> Dict[str, str]:
        """
        Extract important product attributes from the query.
        """
        attributes = {}
        
        # Extract product type
        type_patterns = [
            (r'áo\s+sơ\s+mi', 'áo sơ mi'),
            (r'quần\s+jean', 'quần jean'),
            (r'quần\s+kaki', 'quần kaki'),
            (r'quần\s+âu', 'quần âu'),
            (r'áo\s+thun', 'áo thun'),
            (r'áo\s+khoác', 'áo khoác')
        ]
        
        for pattern, value in type_patterns:
            if re.search(pattern, query.lower()):
                attributes['type'] = value
                break
        
        # Extract color
        color_patterns = [
            (r'trắng', 'trắng'),
            (r'đen', 'đen'),
            (r'xanh', 'xanh'),
            (r'đỏ', 'đỏ'),
            (r'vàng', 'vàng'),
            (r'xám', 'xám'),
            (r'hồng', 'hồng'),
            (r'tím', 'tím'),
            (r'be', 'be'),
            (r'nâu', 'nâu')
        ]
        
        for pattern, value in color_patterns:
            if re.search(pattern, query.lower()):
                attributes['color'] = value
                break
        
        # Extract style
        style_patterns = [
            (r'slim\s*fit', 'Slim fit'),
            (r'regular\s*fit', 'Regular fit'),
            (r'oversized', 'Oversized'),
            (r'loose\s*fit', 'Loose fit'),
            (r'skinny', 'Skinny'),
            (r'ôm', 'ôm'),
            (r'rộng', 'rộng')
        ]
        
        for pattern, value in style_patterns:
            if re.search(pattern, query.lower()):
                attributes['style'] = value
                break
        
        return attributes
    
    def _rerank_product_results(self, results: List[Document], attributes: Dict[str, str]) -> List[Document]:
        """
        Rerank product search results based on attribute matching.
        """
        if not attributes or not results:
            return results
        
        scored_results = []
        
        # Log the total results before reranking
        logger.info(f"Reranking {len(results)} results for attributes: {attributes}")
        
        # Check if we have a specific product type we're looking for
        product_type = attributes.get('type', '').lower()
        product_color = attributes.get('color', '').lower()
        product_style = attributes.get('style', '').lower()
        
        # Special handling for specific types
        is_special_type = product_type in ['quần âu', 'quần jean', 'quần kaki']
        
        for doc in results:
            score = 0
            doc_content = doc.page_content.lower()
            doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            doc_format = doc_metadata.get('format', '')
            
            # Basic scoring for format matching
            if is_special_type and doc_format == 'name_first' and product_type:
                # For specific types like "quần âu", prefer name_first format
                score += 2
                logger.info(f"Format bonus for name_first format with special type: +2")
            elif doc_format == 'original' and 'product_id' in attributes:
                # For product ID searches, prefer original format
                score += 2
                logger.info(f"Format bonus for original format with product ID: +2")
            
            # Add boost for documents that start with the product type
            if product_type and doc_content.startswith(product_type):
                score += 4
                logger.info(f"Boost for document starting with '{product_type}': +4")
            
            # Check for product type match in content
            if product_type and product_type in doc_content:
                base_score = 3
                
                # Special handling for specific product types
                if is_special_type:
                    # Look for exact matches at beginning of lines or after punctuation
                    type_patterns = [
                        fr'^{re.escape(product_type)}',  # At start of line
                        fr'[.,:;]\s*{re.escape(product_type)}',  # After punctuation
                        fr'tên\s+sản\s+phẩm:.*{re.escape(product_type)}',  # In product name
                        fr'mã\s+sản\s+phẩm:.*\s+{re.escape(product_type)}'  # Near product ID
                    ]
                    
                    # Add bonus points for each pattern found
                    for pattern in type_patterns:
                        if re.search(pattern, doc_content, re.MULTILINE | re.IGNORECASE):
                            base_score += 2
                            logger.info(f"Found strong match pattern for '{product_type}' (+2 points)")
                            
                    # Super high priority for "quần âu" specifically as requested
                    if product_type == 'quần âu' and "quần âu" in doc_content:
                        base_score += 5
                        logger.info(f"Added bonus points for 'quần âu' document")
                
                score += base_score
                logger.info(f"Product type match score: +{base_score} for '{product_type}'")
                
            # Check product_name metadata
            if 'product_name' in doc_metadata:
                meta_product_name = str(doc_metadata['product_name']).lower()
                
                # Bonus if metadata product name matches query type
                if product_type and product_type in meta_product_name:
                    score += 3
                    logger.info(f"Product type in metadata match: +3 for '{product_type}'")
                
                # Additional bonus for name-first format if product type is in metadata
                if doc_format == 'name_first' and product_type and product_type in meta_product_name:
                    score += 2
                    logger.info(f"Additional bonus for name-first format with matching meta: +2")
            
            # Check for color match
            if product_color and product_color in doc_content:
                score += 2
                logger.info(f"Color match: +2 for '{product_color}'")
            
            # Check for style match
            if product_style and product_style in doc_content:
                score += 2
                logger.info(f"Style match: +2 for '{product_style}'")
            
            # Penalize general sections
            if "THÔNG TIN CHUNG" in doc_content:
                score -= 5
                logger.info("Penalized general section: -5")
            
            # Log individual document scores for debugging
            logger.info(f"Final document score: {score} for {doc_metadata.get('product_id', 'unknown')} - {doc_content[:50]}...")
            
            scored_results.append((doc, score))
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Log reranking results
        logger.info(f"Reranked results (showing top {min(5, len(scored_results))}):")
        for i, (doc, score) in enumerate(scored_results[:5]):
            metadata_id = doc.metadata.get('product_id', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
            format_type = doc.metadata.get('format', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
            logger.info(f"  {i+1}. Score {score}: {metadata_id} (format: {format_type}) - {doc.page_content[:50]}...")
        
        # Remove duplicates based on product_id to avoid showing the same product twice
        seen_product_ids = set()
        unique_docs = []
        
        for doc, _ in scored_results:
            product_id = doc.metadata.get('product_id', '') if hasattr(doc, 'metadata') else ''
            if product_id and product_id not in seen_product_ids:
                seen_product_ids.add(product_id)
                unique_docs.append(doc)
            elif not product_id:
                # If there's no product_id, always include
                unique_docs.append(doc)
        
        logger.info(f"Removed duplicates: {len(scored_results)} -> {len(unique_docs)} unique documents")
        
        return unique_docs
    
    def _limit_context_by_tokens(self, documents: List[Document], max_tokens: int = 4000) -> List[Document]:
        """Limit the number of documents to avoid exceeding token limits."""
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        token_count = 0
        limited_docs = []
        
        for doc in documents:
            doc_tokens = len(enc.encode(doc.page_content))
            if token_count + doc_tokens <= max_tokens:
                limited_docs.append(doc)
                token_count += doc_tokens
            else:
                break
            
        logger.info(f"Limited context from {len(documents)} docs to {len(limited_docs)} docs ({token_count} tokens)")
        return limited_docs
    
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
            chunked_documents = []
            
            # Process each document
            for doc in documents:
                # Split the document into product sections first
                product_sections = re.split(r'(\d+)\.\s+Mã\s+sản\s+phẩm:', doc.page_content)
                
                # Skip the first empty section if it exists
                if product_sections and not product_sections[0].strip():
                    product_sections = product_sections[1:]
                    
                
                # Group sections by pairs (index, content)
                product_data = []
                for i in range(0, len(product_sections), 2):
                    if i+1 < len(product_sections):
                        product_data.append((product_sections[i], product_sections[i+1]))
                
                # Process each product section
                for index, section in product_data:
                    # Add back the "Mã sản phẩm:" prefix that was removed by the split
                    full_section = f"{index}. Mã sản phẩm:{section}"
                    
                    # Extract product ID - try both formats with and without "Mã sản phẩm:" prefix
                    id_match = re.search(r'Mã\s+sản\s+phẩm:\s+([A-Z]{2,3}\d{3,4})', full_section)
                    if not id_match:
                        # Try direct pattern for cases where the prefix might be missing
                        id_match = re.search(r'^\s*([A-Z]{2,3}\d{3,4})\s', full_section)

                    product_id = id_match.group(1) if id_match else None
                    
                    # Extract product name
                    name_match = re.search(r'Tên\s+sản\s+phẩm:\s+(.*?)(?:\n|$|-)', full_section)
                    product_name = name_match.group(1).strip() if name_match else None
                    
                    if not product_id or not product_name:
                        logger.warning(f"Could not extract product ID or name from section: {full_section[:100]}...")
                        continue
                    
                    logger.info(f"Processing product section: ID={product_id}, Name={product_name}")
                    
                    # Create text splitter with appropriate settings
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500 if product_id else 1200,
                        chunk_overlap=100 if product_id else 200,
                        length_function=len,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    
                    # ---- ORIGINAL DOCUMENT (starts with product_id) ----
                    # Split this product section into chunks
                    original_chunks = text_splitter.split_text(full_section)
                    
                    # Create new documents for each chunk with CORRECT product metadata
                    for j, chunk in enumerate(original_chunks):
                        chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                        chunk_metadata.update({
                            "chunk": j,
                            "chunk_total": len(original_chunks),
                            "product_id": product_id,
                            "product_name": product_name,
                            "category": doc.metadata.get("category", ""),
                            "format": "original"  # Mark this as the original format
                        })
                        
                        chunk_doc = Document(page_content=chunk, metadata=chunk_metadata)
                        chunked_documents.append(chunk_doc)
                    
                    # ---- ADDITIONAL DOCUMENT (starts with product_name) ----
                    # Create a version of the section that starts with the product name
                    product_name_first = f"{product_name} {product_id}\n"
                    
                    # Add the rest of the content, skipping the original product name line
                    content_lines = full_section.split('\n')
                    for line in content_lines:
                        if "Tên sản phẩm:" not in line and line.strip():
                            product_name_first += line + "\n"
                    
                    # Split this alternative format into chunks
                    name_first_chunks = text_splitter.split_text(product_name_first)
                    
                    # Create new documents for each chunk with the same metadata but marked as name-first format
                    for j, chunk in enumerate(name_first_chunks):
                        chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                        chunk_metadata.update({
                            "chunk": j,
                            "chunk_total": len(name_first_chunks),
                            "product_id": product_id,
                            "product_name": product_name,
                            "category": doc.metadata.get("category", ""),
                            "format": "name_first"  # Mark this as the name-first format
                        })
                        logger.info(f"Created alternative format chunk with name first: {chunk[:50]}...")
                        
                        chunk_doc = Document(page_content=chunk, metadata=chunk_metadata)
                        chunked_documents.append(chunk_doc)
            
            # Log the total number of chunked documents
            logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents (including alternative formats)")
            
            # Add the chunked documents to the vector store
            self.add_documents(chunked_documents)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in add_documents_with_chunking: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False




        
def test():        
    # Sample document to test
    doc_content = """
    20. Mã sản phẩm: AT020  
    Tên sản phẩm: Áo Thun Graphic Art Thập Niên 70 
    - Chất liệu: Cotton pha spandex mềm mịn  
    - Kiểu dáng: Regular fit (đồ họa nghệ thuật vintage)  
    - Giá: 460.000 VNĐ  
    - Màu sắc: Xanh teal, Đỏ rượu, Nâu đất  
    """
    
    # Create a Document object with metadata
    doc = Document(
        page_content=doc_content, 
        metadata={"filename": "test_doc.txt", "category": "ao thun"}
    )
    
    try:
        chunked_documents = []
        
        # Split the document into product sections first
        product_sections = re.split(r'\d+\.\s+Mã\s+sản\s+phẩm:', doc.page_content)
        
        # Skip the first empty section if it exists
        if product_sections and not product_sections[0].strip():
            product_sections = product_sections[1:]
        
        # Process each product section
        for i, section in enumerate(product_sections):
            if not section.strip():
                continue
                
            # Add back the "Mã sản phẩm:" prefix that was removed by the split
            full_section = f"Mã sản phẩm:{section}"
            
            # Extract product ID
            id_match = re.search(r'Mã\s+sản\s+phẩm:\s+([A-Z0-9]+)', full_section)
            product_id = id_match.group(1) if id_match else None
            
            # Extract product name
            name_match = re.search(r'Tên\s+sản\s+phẩm:\s+(.*?)(?:\n|$|-)', full_section)
            product_name = name_match.group(1).strip() if name_match else None
            
            if not product_id or not product_name:
                logger.warning(f"Could not extract product ID or name from section: {full_section[:100]}...")
                continue
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Increased to 1000 characters
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split this product section into chunks
            chunks = text_splitter.split_text(full_section)
            
            # Create new documents for each chunk with CORRECT product metadata
            for j, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                chunk_metadata.update({
                    "chunk": j,
                    "chunk_total": len(chunks),
                    "product_id": product_id,
                    "product_name": product_name,
                    "category": doc.metadata.get("category", "")
                })
                
                chunk_doc = Document(page_content=chunk, metadata=chunk_metadata)
                chunked_documents.append(chunk_doc)
        
        # Print results for testing
        logger.info(f"Created {len(chunked_documents)} chunks from test document")
        for i, chunk_doc in enumerate(chunked_documents):
            logger.info(f"Chunk {i}:")
            logger.info(f"  Content: {chunk_doc.page_content[:50]}...")
            logger.info(f"  Metadata: {chunk_doc.metadata}")
        
        return chunked_documents
            
    except Exception as e:
        logger.error(f"Error in test function: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def test2():
    """Test the search functionality with a sample query."""
    vector_store = VectorStore()
    query = "Tìm kiếm sản phẩm: AT020"
    intent = "product"
    
    # Now we can use await since we're in an async function
    relevant_docs = await vector_store.search(query, intent, 5)
    
    # Print the results for debugging
    logger.info(f"Found {len(relevant_docs)} relevant documents for query: '{query}'")
    for i, doc in enumerate(relevant_docs):
        logger.info(f"Result {i+1}:")
        logger.info(f"  Content: {doc.page_content[:200]}...")
        logger.info(f"  Metadata: {doc.metadata}")
    
    return relevant_docs

async def test3():
    """Test the search functionality with a sample query."""
    vector_store = VectorStore()
    query = "Quần Jean Skinny Wax"
    intent = "product"
    
    # Now we can use await since we're in an async function
    relevant_docs = await vector_store.search(query, intent, 5)
    
    # Print the results for debugging
    logger.info(f"Found {len(relevant_docs)} relevant documents for query: '{query}'")
    for i, doc in enumerate(relevant_docs):
        logger.info(f"Result {i+1}:")
        logger.info(f"  Content: {doc.page_content[:200]}...")
        logger.info(f"  Metadata: {doc.metadata}")
    
    return relevant_docs

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test3())