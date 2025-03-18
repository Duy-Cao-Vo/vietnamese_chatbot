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
LLM_MODE = "api"  # "local" hoặc "api"
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
    "greeting": "Xin chào! Tôi là trợ lý AI của cửa hàng thời trang. Tôi có thể giúp gì cho bạn hôm nay?",
    "fallback": "Xin lỗi, tôi không hiểu ý của bạn. Bạn có thể diễn đạt lại hoặc hỏi một câu hỏi khác được không?"
}

# Cấu hình Prompt Template
SYSTEM_PROMPT = """
Bạn là trợ lý AI cho một cửa hàng thời trang. Nhiệm vụ của bạn là trả lời các câu hỏi của 
khách hàng về sản phẩm, tồn kho, chính sách của cửa hàng, và các thông tin liên quan.

Hãy trả lời một cách lịch sự, thân thiện và hữu ích. Nếu bạn không biết câu trả lời, 
hãy thành thật và đề nghị khách hàng liên hệ trực tiếp với cửa hàng.

{context}
"""

RAG_PROMPT = """
Người dùng hỏi: {question}

Dựa trên thông tin sau:
{context}

Hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ nhất. 
Chỉ sử dụng thông tin được cung cấp và kiến thức chung về thời trang. 
Đừng bịa ra thông tin không có trong ngữ cảnh.
"""

# API Configuration (only used if LLM_MODE="api")
API_BASE_URL = "https://public-api.grabgpt.managed.catwalk-k8s.stg-myteksi.com/openai/deployments/gpt-35-turbo"
API_VERSION = "2023-03-15-preview"
API_ENGINE = "gpt-35-turbo"
API_TIMEOUT = 30  # seconds 



# Intent recognition
INTENT_MODEL_PATH = os.path.join(BASE_DIR, "models", "intent")

# Inventory service configuration
INVENTORY_MODE = "dummy"  # "db", "api", or "dummy"
INVENTORY_DB_HOST = "localhost"
INVENTORY_DB_PORT = 5432
INVENTORY_DB_USER = "postgres"
INVENTORY_DB_PASSWORD = ""
INVENTORY_DB_NAME = "inventory"
INVENTORY_API_URL = "http://localhost:8000/api"
INVENTORY_API_TIMEOUT = 10  # seconds 