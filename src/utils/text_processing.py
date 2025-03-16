import re
import unicodedata
import logging
from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger("clothing-chatbot")

# Thử import các thư viện xử lý ngôn ngữ tiếng Việt, bỏ qua nếu không có
try:
    import underthesea
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    logger.warning("underthesea not available. Using basic text processing.")
    UNDERTHESEA_AVAILABLE = False

try:
    import pyvi
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    logger.warning("pyvi not available. Using basic text processing.")
    PYVI_AVAILABLE = False


def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt (loại bỏ dấu câu, khoảng trắng thừa).
    
    Args:
        text: Văn bản đầu vào
        
    Returns:
        Văn bản đã chuẩn hóa
    """
    # Chuyển về chữ thường
    text = text.lower()
    
    # Loại bỏ dấu câu
    text = re.sub(r'[^\w\s\dáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tách từ tiếng Việt.
    
    Args:
        text: Văn bản đầu vào
        
    Returns:
        Danh sách các từ đã tách
    """
    # Thử sử dụng underthesea nếu có
    if UNDERTHESEA_AVAILABLE:
        try:
            return underthesea.word_tokenize(text)
        except Exception as e:
            logger.error(f"Error using underthesea for tokenization: {str(e)}")
    
    # Thử sử dụng pyvi nếu có
    if PYVI_AVAILABLE:
        try:
            tokenized_text = ViTokenizer.tokenize(text)
            return tokenized_text.split()
        except Exception as e:
            logger.error(f"Error using pyvi for tokenization: {str(e)}")
    
    # Nếu không có thư viện chuyên dụng, sử dụng cách tách đơn giản
    return text.split()


def extract_product_info(text: str) -> Dict[str, Any]:
    """
    Trích xuất thông tin sản phẩm từ văn bản.
    
    Args:
        text: Văn bản đầu vào
        
    Returns:
        Thông tin sản phẩm dạng dictionary (product_type, color, size)
    """
    product_info = {
        "product_type": None,
        "color": None,
        "size": None
    }
    
    # Tìm loại sản phẩm
    product_types = {
        "áo thun": ["áo thun", "áo phông", "áo tee", "t-shirt", "tshirt", "tee"],
        "áo sơ mi": ["áo sơ mi", "sơ mi", "shirt", "áo shirt"],
        "quần jeans": ["quần jeans", "quần jean", "quần bò", "jeans", "jean"],
        "quần kaki": ["quần kaki", "quần khaki", "kaki", "khaki"],
        "áo khoác": ["áo khoác", "áo jacket", "jacket", "khoác", "hoodie", "áo hoodie"],
        "váy": ["váy", "đầm", "váy đầm", "dress", "chân váy"]
    }
    
    for product_type, keywords in product_types.items():
        for keyword in keywords:
            if keyword in text.lower():
                product_info["product_type"] = product_type
                break
        if product_info["product_type"]:
            break
    
    # Tìm màu sắc
    colors = {
        "trắng": ["trắng", "white", "màu trắng"],
        "đen": ["đen", "black", "màu đen"],
        "xanh": ["xanh", "blue", "xanh dương", "màu xanh"],
        "đỏ": ["đỏ", "red", "màu đỏ"],
        "vàng": ["vàng", "yellow", "màu vàng"],
        "xám": ["xám", "grey", "gray", "màu xám"],
        "hồng": ["hồng", "pink", "màu hồng"],
        "tím": ["tím", "purple", "màu tím"],
        "xanh lá": ["xanh lá", "green", "màu xanh lá"],
        "cam": ["cam", "orange", "màu cam"],
        "nâu": ["nâu", "brown", "màu nâu"]
    }
    
    for color, keywords in colors.items():
        for keyword in keywords:
            if keyword in text.lower():
                product_info["color"] = color
                break
        if product_info["color"]:
            break
    
    # Tìm kích cỡ
    # Kích cỡ chữ
    letter_sizes = ["s", "m", "l", "xl", "xxl", "xxxl", "size s", "size m", "size l", "size xl", "size xxl"]
    for size in letter_sizes:
        pattern = r'\b' + re.escape(size) + r'\b'
        if re.search(pattern, text.lower()):
            product_info["size"] = size.upper().replace("SIZE ", "")
            break
    
    # Kích cỡ số
    number_sizes = ["28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42"]
    for size in number_sizes:
        pattern = r'\b' + re.escape(size) + r'\b'
        if re.search(pattern, text.lower()):
            product_info["size"] = size
            break
    
    return product_info


def find_quantity_in_text(text: str) -> Optional[int]:
    """
    Tìm số lượng từ văn bản.
    
    Args:
        text: Văn bản đầu vào
        
    Returns:
        Số lượng hoặc None nếu không tìm thấy
    """
    # Tìm mẫu như "2 cái", "mua 3", "lấy 5 áo"
    quantity_patterns = [
        r'(\d+)\s*(cái|chiếc|đôi|món|bộ|sản phẩm|áo|quần|váy)',
        r'mua\s*(\d+)',
        r'lấy\s*(\d+)',
        r'đặt\s*(\d+)',
        r'(\d+)\s*sản phẩm',
        r'số\s*lượng\s*(\d+)',
        r'số\s*lượng[:]?\s*(\d+)',
        r'với\s*số\s*lượng\s*(\d+)',
        r'(\d+)'
    ]
    
    for pattern in quantity_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                quantity = int(match.group(1))
                return quantity
            except ValueError:
                continue
    
    return None 