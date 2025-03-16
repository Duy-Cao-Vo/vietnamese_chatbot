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
            
            # Truy xuất thông tin liên quan từ vector store
            relevant_docs = await self.vector_store.search(query, intent)
            context_texts = [doc.page_content for doc in relevant_docs]
            
            # Nếu không tìm thấy thông tin liên quan, thử tìm kiếm từ tệp
            if not context_texts:
                context_texts = await self._get_context_from_files(query, intent)
            
            # Sinh câu trả lời sử dụng RAG
            if context_texts:
                response = await self.llm.generate_rag_response(query, context_texts)
            else:
                # Sử dụng phản hồi chung nếu không có ngữ cảnh
                response = await self.llm.generate(
                    f"Người dùng hỏi: {query}. Hãy trả lời dựa trên kiến thức của bạn.",
                    config.SYSTEM_PROMPT.format(context="")
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return config.DEFAULT_RESPONSES["fallback"]
    
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