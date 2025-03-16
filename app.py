#!/usr/bin/env python3
"""
Ứng dụng chính cho Clothing Store Chatbot.
"""

import os
import sys
import logging
import asyncio
import uvicorn
import subprocess
import uuid
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.agent.intent_detector import IntentDetector
from src.agent.response_generator import ResponseGenerator
from src.llm.model import LLMModel
from src.database.vector_store import VectorStore
from src.database.inventory_db import InventoryDB

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("clothing-chatbot")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title=config.APP_NAME,
    description="API cho chatbot cửa hàng quần áo",
    version="1.0.0"
)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo các đối tượng cần thiết
intent_detector = IntentDetector()
llm_model = LLMModel()
vector_store = VectorStore()
inventory_db = InventoryDB()
response_generator = ResponseGenerator(llm_model, vector_store)

# Conversations dictionary to store session data
conversations = {}

# Định nghĩa các model dữ liệu
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    session_data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    conversation_id: str
    additional_data: Optional[Dict[str, Any]] = None

class OrderItem(BaseModel):
    product_id: str
    size: str
    color: str
    quantity: int

class Order(BaseModel):
    customer_name: str
    customer_phone: str
    customer_email: Optional[str] = None
    customer_address: str
    items: List[OrderItem]
    notes: Optional[str] = None

class OrderResponse(BaseModel):
    order_id: str
    status: str
    message: str
    order_summary: Dict[str, Any]
    total_amount: float

# API endpoints
@app.get("/")
async def root():
    """Endpoint chính."""
    return {"message": "Chào mừng đến với API của Clothing Store Chatbot", "status": "online"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint xử lý chat.
    """
    try:
        # Lấy hoặc tạo mới conversation_id
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Lấy hoặc tạo mới session data
        if conversation_id not in conversations:
            conversations[conversation_id] = {}
        
        # Cập nhật session data từ request nếu có
        if request.session_data:
            conversations[conversation_id].update(request.session_data)
        
        # Phát hiện intent
        intent = intent_detector.detect_intent(request.message)
        logger.info(f"Detected intent: {intent} for message: {request.message}")
        
        # Tạo câu trả lời
        response = await response_generator.generate_response(
            request.message, intent, conversations[conversation_id]
        )
        
        # Tạo additional_data dựa trên intent
        additional_data = {}
        
        # Nếu là intent tồn kho, thêm thông tin sản phẩm
        if intent == "inventory":
            from src.utils.text_processing import extract_product_info
            product_info = extract_product_info(request.message)
            if product_info["product_type"]:
                # Tìm kiếm sản phẩm
                products = inventory_db.search_products(product_info["product_type"])
                if products:
                    additional_data["products"] = products[:5]  # Giới hạn 5 sản phẩm
        
        return ChatResponse(
            response=response,
            intent=intent,
            conversation_id=conversation_id,
            additional_data=additional_data
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/inventory/{product_id}")
async def get_inventory(product_id: str, size: Optional[str] = None, color: Optional[str] = None):
    """
    Endpoint kiểm tra tồn kho.
    """
    try:
        result = inventory_db.get_product_inventory(product_id, size, color)
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in inventory endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/order", response_model=OrderResponse)
async def create_order(order: Order, background_tasks: BackgroundTasks):
    """
    Endpoint tạo đơn hàng.
    """
    try:
        # Tạo ID đơn hàng
        order_id = f"ORD-{str(uuid.uuid4())[:8].upper()}"
        
        # Tính tổng tiền và tạo order summary
        order_summary = {
            "customer": {
                "name": order.customer_name,
                "phone": order.customer_phone,
                "email": order.customer_email,
                "address": order.customer_address
            },
            "items": []
        }
        
        total_amount = 0
        
        for item in order.items:
            # Lấy thông tin sản phẩm
            product_info = inventory_db.get_product_info(item.product_id)
            if not product_info:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy sản phẩm: {item.product_id}")
            
            # Kiểm tra tồn kho
            inventory = inventory_db.get_product_inventory(item.product_id, item.size, item.color)
            if inventory.get("status") != "in_stock" or inventory.get("quantity", 0) < item.quantity:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Sản phẩm hết hàng hoặc không đủ số lượng: {product_info.get('name')}, size {item.size}, màu {item.color}"
                )
            
            # Tính giá tiền
            price = product_info.get("price", 0)
            item_total = price * item.quantity
            total_amount += item_total
            
            # Thêm vào summary
            order_summary["items"].append({
                "product_id": item.product_id,
                "product_name": product_info.get("name", ""),
                "size": item.size,
                "color": item.color,
                "quantity": item.quantity,
                "price": price,
                "total": item_total
            })
            
            # Cập nhật tồn kho (trừ đi số lượng đã mua)
            background_tasks.add_task(
                inventory_db.update_inventory,
                item.product_id, item.size, item.color, -item.quantity
            )
        
        # Thêm thông tin khác
        order_summary["subtotal"] = total_amount
        order_summary["shipping"] = 30000  # Phí ship mặc định
        order_summary["total"] = total_amount + order_summary["shipping"]
        order_summary["order_id"] = order_id
        order_summary["notes"] = order.notes
        
        # Lưu đơn hàng (giả lập)
        orders_dir = os.path.join(config.DATA_DIR, "orders")
        os.makedirs(orders_dir, exist_ok=True)
        
        with open(os.path.join(orders_dir, f"{order_id}.json"), "w", encoding="utf-8") as f:
            json.dump(order_summary, f, ensure_ascii=False, indent=2)
        
        return OrderResponse(
            order_id=order_id,
            status="success",
            message="Đơn hàng đã được tạo thành công",
            order_summary=order_summary,
            total_amount=order_summary["total"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create order endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products/search")
async def search_products(query: str):
    """
    Endpoint tìm kiếm sản phẩm.
    """
    try:
        results = inventory_db.search_products(query)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error in product search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Endpoint kiểm tra trạng thái.
    """
    return {
        "status": "healthy",
        "components": {
            "llm_model": "loaded" if llm_model.model is not None else "not_loaded",
            "vector_store": "connected",
            "inventory_db": "connected"
        }
    }

def run_streamlit():
    """Khởi chạy ứng dụng Streamlit."""
    streamlit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "streamlit_app.py")
    subprocess.Popen(["streamlit", "run", streamlit_path])

def main():
    # Kiểm tra và tải dữ liệu
    try:
        # Tạo thư mục data nếu chưa có
        os.makedirs(os.path.join(config.DATA_DIR, "orders"), exist_ok=True)
        
        # Kiểm tra nếu vector store chưa có dữ liệu
        if not os.path.exists(config.VECTOR_DB_PATH) or not os.listdir(config.VECTOR_DB_PATH):
            logger.warning("Vector database empty. Please run scripts/ingest_data.py to populate it.")
        
        # Kiểm tra mô hình LLM
        if not os.path.exists(config.LLM_MODEL_PATH):
            logger.warning("LLM model not found. Please run scripts/download_model.py to download it.")
        
        # Try to start Streamlit, but continue if it fails
        try:
            # Khởi chạy Streamlit trong thread riêng
            threading.Thread(target=run_streamlit, daemon=True).start()
            logger.info("Streamlit UI started")
        except Exception as e:
            logger.warning(f"Could not start Streamlit UI: {str(e)}. The API will still be available.")
            logger.warning("Install Streamlit with: pip install streamlit")
        
        # Khởi chạy FastAPI
        logger.info(f"Starting FastAPI server on port {config.APP_PORT}")
        uvicorn.run(
            "app:app",
            host="0.0.0.0", 
            port=config.APP_PORT,
            reload=config.DEBUG
        )
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 