import re
import logging
from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger("clothing-chatbot")

class IntentDetector:
    """
    Lớp phát hiện ý định (intent) từ câu hỏi của người dùng.
    Sử dụng các mẫu regex và từ khóa để xác định ý định.
    """
    
    def __init__(self):
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
                r"khi\s*nào\s*có\s*hàng"
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
        
    def detect_intent(self, message: str) -> str:
        """
        Phát hiện intent từ tin nhắn của người dùng.
        
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