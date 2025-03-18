import re
import logging
from typing import List, Dict, Any, Optional
import sys
import os
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.llm.model import LLMModel  # Import the LLM model

logger = logging.getLogger("clothing-chatbot")

class IntentDetector:
    """
    Lớp phát hiện ý định (intent) từ câu hỏi của người dùng.
    Sử dụng các mẫu regex và từ khóa để xác định ý định.
    """
    
    def __init__(self, llm_model: Optional[LLMModel] = None):
        # Initialize with an optional LLM model
        self.llm = llm_model if llm_model is not None else LLMModel()
        
        # Các từ khóa và pattern cho mỗi loại intent
        self.intent_patterns = {
            "policy": [
                r"chính\s*sách",
                r"đổi\s*trả",
                r"hoàn\s*tiền",
                r"bảo\s*hành",
                r"bao\s*lâu",
                r"thời\s*hạn",
                r"điều\s*kiện",
                r"quy\s*định",
                r"hỏng",
                r"lỗi",
                r"đổi\s*size",
                r"đổi\s*màu"
            ],
            "product": [
                r"sản\s*phẩm",
                r"áo",
                r"quần",
                r"đầm",
                r"váy",
                r"giày",
                r"mũ",
                r"khăn",
                r"phụ\s*kiện",
                r"chất\s*liệu",
                r"màu\s*sắc",
                r"size",
                r"kích\s*cỡ",
                r"mẫu",
                r"kiểu\s*dáng",
                r"phối\s*đồ",
                r"mặc\s*với",
                r"đeo\s*với",
                r"đẹp\s*không",
                r"giặt",
                r"bảo\s*quản"
            ],
            "inventory": [
                r"còn\s*hàng",
                r"hết\s*hàng",
                r"còn\s*size",
                r"còn\s*màu",
                r"tồn\s*kho",
                r"kiểm\s*tra",
                r"có\s*sẵn",
                r"bao\s*giờ\s*về\s*hàng",
                r"mấy\s*giờ",
                r"khi\s*nào\s*có\s*hàng",
                r"còn\s*không",
                r"còn\s*hàng\s*không"
            ],
            "store": [
                r"cửa\s*hàng",
                r"chi\s*nhánh",
                r"địa\s*chỉ",
                r"địa\s*điểm",
                r"ở\s*đâu",
                r"chỗ\s*nào",
                r"mở\s*cửa",
                r"đóng\s*cửa",
                r"giờ\s*mở\s*cửa",
                r"thành\s*phố",
                r"tỉnh",
                r"quận",
                r"huyện"
            ],
            "purchase": [
                r"mua",
                r"đặt\s*hàng",
                r"đặt\s*mua",
                r"thanh\s*toán",
                r"giá",
                r"bao\s*nhiêu\s*tiền",
                r"phí\s*ship",
                r"phí\s*vận\s*chuyển",
                r"giao\s*hàng",
                r"bao\s*lâu",
                r"nhận\s*hàng",
                r"mã\s*giảm\s*giá",
                r"voucher",
                r"khuyến\s*mãi",
                r"sale",
                r"giảm\s*giá"
            ],
            "greeting": [
                r"^xin\s*chào",
                r"^chào\s*bạn",
                r"^hello",
                r"^hi",
                r"^helo",
                r"^chào",
                r"^này",
                r"^alo",
                r"^có\s*ai\s*không",
                r"^có\s*ai\s*ở\s*đó\s*không"
            ]
        }
        
        # Danh sách intent theo thứ tự ưu tiên
        self.intent_priority = [
            "greeting",
            "purchase", 
            "inventory", 
            "product", 
            "policy", 
            "store"
        ]
        
        # Intent descriptions for LLM
        self.intent_descriptions = {
            "greeting": "Lời chào hoặc bắt đầu cuộc trò chuyện",
            "purchase": "Thông tin về mua hàng, thanh toán, giá cả, giao hàng",
            "inventory": "Kiểm tra tồn kho, size có sẵn, màu sắc có sẵn, sản phẩm còn hay hết",
            "product": "Thông tin về sản phẩm, đặc điểm, chất liệu, cách sử dụng",
            "policy": "Chính sách đổi trả, bảo hành, hoàn tiền",
            "store": "Thông tin về cửa hàng, địa chỉ, giờ mở cửa",
            "general": "Câu hỏi chung không thuộc các loại trên"
        }
        
    async def detect_intent_llm(self, message: str) -> str:
        """
        Phát hiện intent từ tin nhắn của người dùng sử dụng mô hình LLM.
        
        Args:
            message: Tin nhắn của người dùng
            
        Returns:
            Chuỗi xác định intent: "policy", "product", "inventory", "store", "purchase", "greeting", hoặc "general"
        """
        if not message or not self.llm:
            logger.warning("No message or LLM is not available. Falling back to regex-based intent detection.")
            # Fallback to regex-based intent detection if no message or LLM is not available
            return self.detect_intent(message)
        
        try:
            # Prepare a prompt for the LLM to determine intent
            intent_options = ", ".join(self.intent_priority + ["general"])
            
            prompt = f"""
Hãy xác định ý định (intent) của người dùng trong tin nhắn sau đây. Chỉ trả về MỘT trong các intent sau: {intent_options}.

Mô tả chi tiết của từng intent:
- greeting: {self.intent_descriptions["greeting"]}
- purchase: {self.intent_descriptions["purchase"]}
- inventory: {self.intent_descriptions["inventory"]}
- product: {self.intent_descriptions["product"]}
- policy: {self.intent_descriptions["policy"]}
- store: {self.intent_descriptions["store"]}
- general: {self.intent_descriptions["general"]}

Lưu ý:
1. Nếu tin nhắn hỏi về một sản phẩm có còn hàng/còn size không, đó là intent "inventory" 
2. Nếu tin nhắn chỉ đề cập đến mã sản phẩm và size (ví dụ: "QK008 còn size nào?"), đó là intent "inventory"
3. Nếu tin nhắn hỏi về thông tin chi tiết sản phẩm, đó là intent "product"

Tin nhắn của người dùng: "{message}"

Intent:
"""
            
            # Custom system prompt for intent detection
            system_prompt = "Bạn là hệ thống phân loại ý định cho một chatbot bán hàng thời trang. Nhiệm vụ của bạn là xác định chính xác ý định của người dùng từ tin nhắn của họ."
            
            # Generate response from LLM
            response = await self.llm.generate(prompt, system_prompt)
            
            # Extract intent from response (assuming the LLM will return just the intent name)
            detected_intent = response.strip().lower()
            logger.info(f"LLM detected intent: {detected_intent} for message: {message}")
            
            # Validate the detected intent
            valid_intents = set(self.intent_priority + ["general"])
            if detected_intent not in valid_intents:
                # If LLM returned something unexpected, extract just the intent word
                for intent in valid_intents:
                    if intent in detected_intent:
                        detected_intent = intent
                        break
                else:
                    # If still not found, fallback to regex
                    logger.warning(f"LLM returned invalid intent: {detected_intent}. Falling back to regex.")
                    return self.detect_intent(message)
            
            logger.info(f"LLM detected intent: {detected_intent} for message: {message}")
            return detected_intent
            
        except Exception as e:
            logger.error(f"Error in LLM intent detection: {str(e)}")
            # Fallback to regex-based detection
            return self.detect_intent(message)
    
    def detect_intent(self, message: str) -> str:
        """
        Phát hiện intent từ tin nhắn của người dùng sử dụng regex.
        
        Args:
            message: Tin nhắn của người dùng
            
        Returns:
            Chuỗi xác định intent: "policy", "product", "inventory", "store", "purchase", "greeting", hoặc "general"
        """
        if not message:
            return "general"
        
        # Chuẩn hóa tin nhắn
        normalized_message = self._normalize_text(message)
        
        # Tính điểm cho mỗi intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, normalized_message, re.IGNORECASE)
                score += len(matches)
            intent_scores[intent] = score
        
        logger.debug(f"Intent scores: {intent_scores}")
        
        # Lấy intent có điểm cao nhất
        max_score = 0
        detected_intent = "general"
        
        # Kiểm tra theo thứ tự ưu tiên
        for intent in self.intent_priority:
            if intent_scores[intent] > 0 and intent_scores[intent] >= max_score:
                max_score = intent_scores[intent]
                detected_intent = intent
        
        # Nếu không có intent nào được phát hiện hoặc điểm quá thấp
        if max_score == 0:
            detected_intent = "general"
            
        logger.info(f"Regex detected intent: {detected_intent} for message: {message}")
        return detected_intent
        
    def _normalize_text(self, text: str) -> str:
        """
        Chuẩn hóa văn bản đầu vào để phù hợp với phân tích.
        
        Args:
            text: Văn bản cần chuẩn hóa
            
        Returns:
            Văn bản đã được chuẩn hóa
        """
        # Chuyển thành chữ thường
        text = text.lower()
        
        # Xóa các ký tự đặc biệt
        text = re.sub(r'[^\w\s\dáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
        
        # Xóa khoảng trắng dư thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text 