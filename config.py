import os
from pathlib import Path

# Đường dẫn
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "policies"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "products"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "inventory"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "stores"), exist_ok=True)

# Cấu hình LLM
LLM_MODE = "local"  # Options: "local" or "api"
# LLM_MODEL_NAME = "microsoft/phi-2"
# LLM_MODEL_PATH = os.path.join(MODEL_DIR, "phi-2")
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
LLM_MODEL_PATH = os.path.join(MODEL_DIR, "DeepSeek-R1-Distill-Qwen-14B")
# Cấu hình Vector Database
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")
EMBEDDING_MODEL_NAME = "vinai/phobert-base"

# Cấu hình ứng dụng
APP_NAME = "Clothing Store Chatbot"
APP_PORT = 8000
DEBUG = True

# Cấu hình ý định (Intent)
INTENT_CATEGORIES = [
    "policy",           # Chính sách
    "product",          # Sản phẩm
    "inventory",        # Tồn kho
    "store",            # Cửa hàng
    "purchase",         # Mua hàng
    "greeting",         # Chào hỏi
    "general",          # Câu hỏi chung
]

# Các câu trả lời mẫu
DEFAULT_RESPONSES = {
    "greeting": "Xin chào! Tôi là trợ lý ảo của cửa hàng quần áo. Tôi có thể giúp gì cho bạn?",
    "not_understood": "Tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể diễn đạt lại được không?",
    "fallback": "Xin lỗi, tôi không thể xử lý yêu cầu này lúc này. Bạn có thể thử lại sau.",
}

# Cấu hình Prompt Template
SYSTEM_PROMPT = """Bạn là trợ lý AI cho cửa hàng quần áo ở Việt Nam. 
Nhiệm vụ của bạn là giúp khách hàng với các thông tin về:
- Chính sách đổi trả, bảo hành
- Thông tin và tư vấn về sản phẩm (cách mặc, màu sắc, kích cỡ)
- Kiểm tra tồn kho sản phẩm
- Thông tin về các cửa hàng
- Hỗ trợ mua hàng

Hãy trả lời một cách lịch sự, thân thiện và chính xác bằng tiếng Việt.
Dựa trên thông tin được cung cấp sau đây để trả lời:
{context}

Nếu bạn không biết câu trả lời, hãy thành thật nói rằng bạn không có thông tin về vấn đề đó.
"""

RAG_PROMPT = """
Dưới đây là câu hỏi của khách hàng: {question}

Dựa trên các tài liệu sau đây:
{context}

Hãy trả lời câu hỏi của khách hàng một cách đầy đủ, thân thiện và chính xác bằng tiếng Việt.
"""

# API Configuration (only used if LLM_MODE="api")
API_BASE_URL = "https://public-api.grabgpt.managed.catwalk-k8s.stg-myteksi.com/openai/deployments/gpt-35-turbo"
API_VERSION = "2023-03-15-preview"
API_ENGINE = "gpt-35-turbo"
API_TIMEOUT = 30  # seconds 