import os
import sys
import logging
from typing import Dict, Any, List, Optional
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.llm.model import LLMModel
from src.database.vector_store import VectorStore
from src.database.inventory_service import InventoryService  # New import for inventory service

logger = logging.getLogger("clothing-chatbot")

class ResponseGenerator:
    """
    Tạo câu trả lời dựa trên ý định của người dùng và sử dụng RAG để lấy thông tin liên quan.
    """
    
    def __init__(self, llm_model: LLMModel, vector_store: VectorStore):
        """
        Khởi tạo response generator.
        
        Args:
            llm_model: Đối tượng mô hình LLM
            vector_store: Đối tượng vector store
        """
        self.llm = llm_model
        self.vector_store = vector_store
        self.data_dir = config.DATA_DIR
        self.inventory_service = InventoryService()  # Initialize inventory service
        
    async def generate_response(self, query: str, intent: str, session_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Tạo câu trả lời cho câu hỏi của người dùng dựa trên intent.
        
        Args:
            query: Câu hỏi của người dùng
            intent: Ý định của người dùng
            session_data: Dữ liệu phiên của người dùng (tùy chọn)
            
        Returns:
            Câu trả lời cho người dùng
        """
        try:
            # Xử lý dựa trên loại intent
            if intent == "greeting":
                return config.DEFAULT_RESPONSES["greeting"]
            
            # Special handling for inventory queries to get real-time data
            if intent == "inventory":
                return await self._handle_inventory_query(query, session_data)
                
            # For other intents, use the RAG approach with vector store
            # Truy xuất thông tin liên quan từ vector store
            logger.info(f"RAG search query: {query}")
            relevant_docs = await self.vector_store.search(query, intent, 10)
            
            # DEBUG: Print retrieved documents
            logger.info(f"Retrieved {len(relevant_docs)} documents from vector store")
            for i, doc in enumerate(relevant_docs):
                logger.info(f"Doc {i+1}: {doc.page_content[:100]}... (metadata: {doc.metadata})")
            
            context_texts = [doc.page_content for doc in relevant_docs]
            
            # Nếu không tìm thấy thông tin liên quan, thử tìm kiếm từ tệp
            if not context_texts:
                logger.info("No documents retrieved from vector store, searching from files")
                context_texts = await self._get_context_from_files(query, intent)
                
                # DEBUG: Print context from files
                logger.info(f"Retrieved {len(context_texts)} context texts from files")
                for i, ctx in enumerate(context_texts):
                    logger.info(f"Context {i+1} from files: {ctx[:100]}...")
            
            # Sinh câu trả lời sử dụng RAG
            if context_texts:
                logger.info(f"Generating RAG response with {len(context_texts)} context texts")
                response = await self.llm.generate_rag_response(query, context_texts)
            else:
                # Sử dụng phản hồi chung nếu không có ngữ cảnh
                logger.info("No context found, generating response without RAG")
                response = await self.llm.generate(
                    f"Người dùng hỏi: {query}. Hãy trả lời dựa trên kiến thức của bạn.",
                    config.SYSTEM_PROMPT.format(context="")
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return config.DEFAULT_RESPONSES["fallback"]
    
    async def _handle_inventory_query(self, query: str, session_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Xử lý truy vấn về tồn kho bằng cách gọi trực tiếp đến API/database.
        
        Args:
            query: Câu hỏi của người dùng
            session_data: Dữ liệu phiên của người dùng (tùy chọn)
            
        Returns:
            Câu trả lời về thông tin tồn kho
        """
        try:
            # Extract product details from the query
            product_details = await self._extract_product_details(query)
            
            # Call inventory database/API directly
            inventory_data = await self.inventory_service.check_inventory(
                product_id=product_details.get("product_id"),
                product_name=product_details.get("product_name"),
                size=product_details.get("size"),
                color=product_details.get("color")
            )
            
            # If no specific product was identified, return general inventory
            if not product_details.get("product_id") and not product_details.get("product_name"):
                inventory_data = await self.inventory_service.get_all_inventory()
                
                # Format response for general inventory
                response = "Đây là thông tin tồn kho hiện tại:\n\n"
                for item in inventory_data:
                    response += f"Sản phẩm: {item['name']} (ID: {item['id']})\n"
                    response += f"Loại: {item['category']}\n"
                    response += f"Giá: {item['price']} VND\n"
                    response += "Tồn kho: "
                    
                    available_sizes = []
                    for size, colors in item['sizes'].items():
                        for color, quantity in colors.items():
                            if quantity > 0:
                                available_sizes.append(f"Size {size}, màu {color}: {quantity} sản phẩm")
                    
                    response += ", ".join(available_sizes) + "\n\n"
                
                return response
            
            # Format response for specific product
            if inventory_data:
                response = f"Thông tin tồn kho cho sản phẩm {inventory_data['name']}:\n\n"
                response += f"ID: {inventory_data['id']}\n"
                response += f"Loại: {inventory_data['category']}\n"
                response += f"Giá: {inventory_data['price']} VND\n"
                response += "Số lượng tồn kho theo kích cỡ và màu sắc:\n"
                
                for size, colors in inventory_data['sizes'].items():
                    for color, quantity in colors.items():
                        if quantity > 0:
                            response += f"- Size {size}, màu {color}: {quantity} sản phẩm\n"
                        else:
                            response += f"- Size {size}, màu {color}: Hết hàng\n"
                
                return response
            else:
                return "Xin lỗi, không tìm thấy thông tin tồn kho cho sản phẩm bạn yêu cầu."
                
        except Exception as e:
            logger.error(f"Error handling inventory query: {str(e)}")
            return f"Xin lỗi, không thể truy vấn thông tin tồn kho lúc này. Lỗi: {str(e)}"
    
    async def _extract_product_details(self, query: str) -> Dict[str, Any]:
        """
        Trích xuất thông tin sản phẩm từ câu truy vấn của người dùng.
        
        Args:
            query: Câu hỏi của người dùng
            
        Returns:
            Thông tin chi tiết về sản phẩm
        """
        try:
            # Skip LLM extraction and use regex patterns directly
            details = {}
            query_lower = query.lower()
            
            # Try to extract product ID - improved to catch product codes without ID: prefix
            import re
            # First look for explicit ID pattern like "ID: QK008"
            id_match = re.search(r'ID[:\s]+([A-Za-z0-9]+)', query)
            if id_match:
                details["product_id"] = id_match.group(1)
            else:
                # Look for alphanumeric codes that look like product IDs (2-3 letters followed by 3-4 numbers)
                # This will catch codes like QK008, ASM001, etc.
                id_pattern = r'\b([A-Za-z]{2,3}\d{3,4})\b'
                direct_id_match = re.search(id_pattern, query.upper())
                if direct_id_match:
                    details["product_id"] = direct_id_match.group(1)
            
            # Extract potential size - improved to catch numeric sizes like "40"
            sizes = ["XS", "S", "M", "L", "XL", "XXL"]
            numeric_sizes = ["36", "37", "38", "39", "40", "41", "42", "43", "44", "45"]
            
            # Check for explicit size mentions
            size_pattern = r'\b(size\s+([SMLX]{1,3}|\d{2})|(cỡ|kích\s*(cỡ|thước)?)?\s+([SMLX]{1,3}|\d{2}))\b'
            size_match = re.search(size_pattern, query, re.IGNORECASE)
            if size_match:
                size_value = size_match.group(2) or size_match.group(5)
                if size_value:
                    details["size"] = size_value.upper()
            else:
                # Check for direct size mentions
                for size in sizes:
                    if f" {size} " in f" {query} " or f" {size.lower()} " in f" {query_lower} ":
                        details["size"] = size
                        break
                
                # Check for numeric sizes
                for size in numeric_sizes:
                    if size in query:
                        details["size"] = size
                        break
            
            # Extract potential product names using common clothing terms
            product_terms = ["áo", "quần", "váy", "đầm", "giày", "dép", "túi", "nón", "mũ", "jacket", "hoodie", "sơ mi"]
            for term in product_terms:
                if term in query_lower:
                    # Try to find the whole product name (e.g., "áo sơ mi", "quần jean")
                    # Look for 2-3 word phrases containing the term
                    pattern = fr'({term}(\s+\w+){{0,2}})'
                    name_match = re.search(pattern, query_lower)
                    if name_match:
                        details["product_name"] = name_match.group(0)
                        break
            
            # Extract potential color
            common_colors = ["đen", "trắng", "đỏ", "xanh", "vàng", "hồng", "tím", "cam", "xám", "nâu", "be"]
            color_pattern = r'(màu\s+(\w+)|(màu sắc|color)?\s+(\w+))'
            color_match = re.search(color_pattern, query_lower)
            if color_match:
                color_value = color_match.group(2) or color_match.group(4)
                if color_value in common_colors:
                    details["color"] = color_value
            else:
                # Direct color mention
                for color in common_colors:
                    if color in query_lower:
                        details["color"] = color
                        break
            
            logger.info(f"Extracted product details: {details}")
            return details
            
        except Exception as e:
            logger.error(f"Error extracting product details: {str(e)}")
            return {}
    
    async def _get_context_from_files(self, query: str, intent: str) -> List[str]:
        """
        Tìm thông tin liên quan từ các tệp dữ liệu dựa trên intent.
        
        Args:
            query: Câu hỏi của người dùng
            intent: Ý định của người dùng
            
        Returns:
            Danh sách các đoạn văn bản liên quan
        """
        context_texts = []
        
        # Xác định thư mục dữ liệu dựa trên intent
        if intent == "policy":
            data_subdir = os.path.join(self.data_dir, "policies")
        elif intent == "product":
            data_subdir = os.path.join(self.data_dir, "products")
        elif intent == "store":
            data_subdir = os.path.join(self.data_dir, "stores")
        elif intent == "inventory":
            # Đối với inventory, chúng ta sẽ sử dụng tệp JSON
            inventory_path = os.path.join(self.data_dir, "inventory", "inventory.json")
            if os.path.exists(inventory_path):
                try:
                    with open(inventory_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Chuyển đổi dữ liệu thành văn bản
                    context_texts.append(f"Thông tin tồn kho cập nhật ngày: {data.get('last_updated', 'N/A')}")
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
                        
                        context_texts.append(product_info)
                except Exception as e:
                    logger.error(f"Error reading inventory data: {str(e)}")
            return context_texts
        else:
            # Đối với các intent khác, không có dữ liệu cụ thể
            return []
        
        # Đọc các tệp trong thư mục dữ liệu
        if os.path.exists(data_subdir):
            for filename in os.listdir(data_subdir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(data_subdir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        context_texts.append(content)
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {str(e)}")
        
        return context_texts 