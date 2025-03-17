#!/usr/bin/env python3
"""
Giao diện người dùng Streamlit cho Clothing Store Chatbot.
"""

import os
import sys
import json
import logging
import requests
import streamlit as st
from typing import Dict, Any, List, Optional
import uuid
import time
from PIL import Image
import io
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("streamlit-ui")

# Cấu hình API
API_URL = f"http://localhost:{config.APP_PORT}/api"

# Wait for the API server to start
logger.info("Waiting for API server to start...")
max_retries = 5
for i in range(max_retries):
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            logger.info("API server is ready")
            break
    except Exception:
        logger.info(f"API not ready yet, waiting... ({i+1}/{max_retries})")
        time.sleep(3)  # Wait 3 seconds before trying again

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Các placeholder cho hình ảnh sản phẩm
DEFAULT_PRODUCT_IMAGE = "https://via.placeholder.com/150?text=No+Image"

# Màu sắc và kiểu
PRIMARY_COLOR = "#ff5722"
SECONDARY_COLOR = "#4caf50"
BG_COLOR = "#f5f5f5"
TEXT_COLOR = "#212121"

def init_session_state():
    """Khởi tạo các biến trong session state."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Thêm tin nhắn chào mừng từ bot
        st.session_state.messages.append({
            "role": "assistant", 
            "content": config.DEFAULT_RESPONSES["greeting"]
        })
    
    if "cart" not in st.session_state:
        st.session_state.cart = []
    
    if "checkout" not in st.session_state:
        st.session_state.checkout = False
    
    if "order_result" not in st.session_state:
        st.session_state.order_result = None

def chat_with_bot(message: str) -> Dict[str, Any]:
    """
    Gửi tin nhắn đến API và nhận phản hồi.
    
    Args:
        message: Tin nhắn của người dùng
        
    Returns:
        Phản hồi từ API dạng dictionary
    """
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "conversation_id": st.session_state.conversation_id,
                "session_data": {"cart": st.session_state.cart} if st.session_state.cart else {}
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error from API: {response.status_code} - {response.text}")
            return {
                "response": config.DEFAULT_RESPONSES["fallback"],
                "intent": "error",
                "conversation_id": st.session_state.conversation_id
            }
    except Exception as e:
        logger.error(f"Error communicating with API: {str(e)}")
        return {
            "response": config.DEFAULT_RESPONSES["fallback"],
            "intent": "error",
            "conversation_id": st.session_state.conversation_id
        }

def get_product_inventory(product_id: str, size: Optional[str] = None, color: Optional[str] = None) -> Dict[str, Any]:
    """
    Lấy thông tin tồn kho của sản phẩm.
    
    Args:
        product_id: ID của sản phẩm
        size: Kích cỡ (tùy chọn)
        color: Màu sắc (tùy chọn)
        
    Returns:
        Thông tin tồn kho dạng dictionary
    """
    try:
        url = f"{API_URL}/inventory/{product_id}"
        params = {}
        if size:
            params["size"] = size
        if color:
            params["color"] = color
            
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error getting inventory: {response.status_code} - {response.text}")
            return {"status": "error", "message": "Không thể lấy thông tin tồn kho"}
    except Exception as e:
        logger.error(f"Error getting inventory: {str(e)}")
        return {"status": "error", "message": str(e)}

def search_products(query: str) -> List[Dict[str, Any]]:
    """
    Tìm kiếm sản phẩm.
    
    Args:
        query: Từ khóa tìm kiếm
        
    Returns:
        Danh sách sản phẩm phù hợp
    """
    try:
        response = requests.get(f"{API_URL}/products/search", params={"query": query})
        
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            logger.error(f"Error searching products: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error searching products: {str(e)}")
        return []

def create_order(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tạo đơn hàng.
    
    Args:
        order_data: Thông tin đơn hàng
        
    Returns:
        Kết quả đặt hàng dạng dictionary
    """
    try:
        response = requests.post(f"{API_URL}/order", json=order_data)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error creating order: {response.status_code} - {response.text}")
            error_msg = response.json().get("detail", "Đã xảy ra lỗi khi tạo đơn hàng")
            return {"status": "error", "message": error_msg}
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        return {"status": "error", "message": str(e)}

def add_to_cart(product_id: str, product_name: str, size: str, color: str, quantity: int, price: float):
    """
    Thêm sản phẩm vào giỏ hàng.
    
    Args:
        product_id: ID của sản phẩm
        product_name: Tên sản phẩm
        size: Kích cỡ
        color: Màu sắc
        quantity: Số lượng
        price: Giá tiền
    """
    # Kiểm tra xem sản phẩm đã có trong giỏ hàng chưa
    for item in st.session_state.cart:
        if (item["product_id"] == product_id and 
            item["size"] == size and 
            item["color"] == color):
            item["quantity"] += quantity
            return
    
    # Thêm sản phẩm mới vào giỏ hàng
    st.session_state.cart.append({
        "product_id": product_id,
        "product_name": product_name,
        "size": size,
        "color": color,
        "quantity": quantity,
        "price": price
    })

def display_messages():
    """Hiển thị lịch sử tin nhắn."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Hiển thị thông tin bổ sung nếu có
                if "additional_data" in message and message["additional_data"]:
                    if "products" in message["additional_data"]:
                        st.subheader("Sản phẩm liên quan:")
                        
                        # Hiển thị sản phẩm dạng lưới
                        cols = st.columns(min(3, len(message["additional_data"]["products"])))
                        for i, product in enumerate(message["additional_data"]["products"]):
                            with cols[i % 3]:
                                st.image(DEFAULT_PRODUCT_IMAGE, width=150)
                                st.markdown(f"**{product['product_name']}**")
                                st.write(f"Giá: {product['price']:,} VND")
                                st.write(f"Trạng thái: {'Còn hàng' if product['status'] == 'in_stock' else 'Hết hàng'}")
                                
                                # Tạo các selectbox cho size, màu sắc, và số lượng
                                if product["status"] == "in_stock":
                                    # Lấy thông tin chi tiết về sản phẩm
                                    inventory = get_product_inventory(product["product_id"])
                                    
                                    if inventory["status"] != "error":
                                        sizes = inventory.get("sizes_available", [])
                                        colors = inventory.get("colors_available", [])
                                        
                                        selected_size = st.selectbox(f"Chọn size {i}", sizes, key=f"size_{product['product_id']}")
                                        selected_color = st.selectbox(f"Chọn màu {i}", colors, key=f"color_{product['product_id']}")
                                        selected_quantity = st.number_input(f"Số lượng {i}", min_value=1, max_value=10, value=1, key=f"quantity_{product['product_id']}")
                                        
                                        if st.button(f"Thêm vào giỏ hàng", key=f"add_to_cart_{product['product_id']}"):
                                            add_to_cart(
                                                product["product_id"],
                                                product["product_name"],
                                                selected_size,
                                                selected_color,
                                                selected_quantity,
                                                product["price"]
                                            )
                                            st.success(f"Đã thêm {product['product_name']} vào giỏ hàng!")
                                            st.rerun()

def display_cart():
    """Hiển thị giỏ hàng."""
    if not st.session_state.cart:
        st.info("Giỏ hàng trống")
        return
    
    st.subheader("Giỏ hàng của bạn")
    
    # Tính tổng tiền
    total = sum(item["price"] * item["quantity"] for item in st.session_state.cart)
    
    # Hiển thị từng sản phẩm trong giỏ hàng
    for i, item in enumerate(st.session_state.cart):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{item['product_name']}**")
            st.write(f"Size: {item['size']}, Màu: {item['color']}")
            st.write(f"Giá: {item['price']:,} VND x {item['quantity']} = {item['price'] * item['quantity']:,} VND")
        
        with col2:
            new_quantity = st.number_input(
                f"Số lượng", 
                min_value=1, 
                max_value=10, 
                value=item["quantity"],
                key=f"cart_quantity_{i}"
            )
            if new_quantity != item["quantity"]:
                st.session_state.cart[i]["quantity"] = new_quantity
                st.rerun()
        
        with col3:
            if st.button("Xóa", key=f"remove_{i}"):
                st.session_state.cart.pop(i)
                st.rerun()
        
        st.divider()
    
    # Hiển thị tổng tiền
    st.markdown(f"### Tổng tiền: {total:,} VND")
    
    # Nút thanh toán
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Tiếp tục mua sắm"):
            st.session_state.checkout = False
            st.rerun()
    with col2:
        if st.button("Thanh toán ngay"):
            st.session_state.checkout = True
            st.rerun()

def display_checkout_form():
    """Hiển thị form thanh toán."""
    st.subheader("Thông tin thanh toán")
    
    # Tạo form
    with st.form("checkout_form"):
        st.write("Thông tin khách hàng")
        customer_name = st.text_input("Họ và tên *", help="Vui lòng nhập đầy đủ họ tên")
        col1, col2 = st.columns(2)
        with col1:
            customer_phone = st.text_input("Số điện thoại *", help="Nhập số điện thoại liên hệ")
        with col2:
            customer_email = st.text_input("Email", help="Nhập email để nhận thông tin đơn hàng")
        
        customer_address = st.text_area("Địa chỉ giao hàng *", help="Vui lòng nhập đầy đủ địa chỉ")
        
        notes = st.text_area("Ghi chú", help="Ghi chú thêm về đơn hàng")
        
        st.write("Xem lại đơn hàng")
        for item in st.session_state.cart:
            st.write(f"- {item['product_name']} (Size: {item['size']}, Màu: {item['color']}) x {item['quantity']}: {item['price'] * item['quantity']:,} VND")
        
        # Tính tổng tiền
        subtotal = sum(item["price"] * item["quantity"] for item in st.session_state.cart)
        shipping_fee = 30000  # Phí ship mặc định
        total = subtotal + shipping_fee
        
        st.write(f"Tạm tính: {subtotal:,} VND")
        st.write(f"Phí vận chuyển: {shipping_fee:,} VND")
        st.markdown(f"### Tổng cộng: {total:,} VND")
        
        # Nút đặt hàng
        submit_button = st.form_submit_button("Đặt hàng")
        
        if submit_button:
            # Kiểm tra dữ liệu đầu vào
            if not customer_name or not customer_phone or not customer_address:
                st.error("Vui lòng điền đầy đủ thông tin bắt buộc")
                return
            
            # Tạo đơn hàng
            order_items = []
            for item in st.session_state.cart:
                order_items.append({
                    "product_id": item["product_id"],
                    "size": item["size"],
                    "color": item["color"],
                    "quantity": item["quantity"]
                })
            
            order_data = {
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "customer_email": customer_email,
                "customer_address": customer_address,
                "items": order_items,
                "notes": notes
            }
            
            # Gửi đơn hàng đến API
            result = create_order(order_data)
            
            # Lưu kết quả đơn hàng
            st.session_state.order_result = result
            
            # Xóa giỏ hàng nếu đặt hàng thành công
            if result.get("status") == "success":
                st.session_state.cart = []
                st.session_state.checkout = False
                st.rerun()
            else:
                st.error(result.get("message", "Đã xảy ra lỗi khi đặt hàng"))

def display_order_result():
    """Hiển thị kết quả đặt hàng."""
    result = st.session_state.order_result
    
    if result.get("status") == "success":
        st.success("Đặt hàng thành công!")
        
        st.write("### Thông tin đơn hàng")
        st.write(f"Mã đơn hàng: **{result['order_id']}**")
        
        # Hiển thị thông tin khách hàng
        st.write("### Thông tin khách hàng")
        customer = result["order_summary"]["customer"]
        st.write(f"Họ và tên: {customer['name']}")
        st.write(f"Số điện thoại: {customer['phone']}")
        if customer.get("email"):
            st.write(f"Email: {customer['email']}")
        st.write(f"Địa chỉ: {customer['address']}")
        
        # Hiển thị danh sách sản phẩm
        st.write("### Danh sách sản phẩm")
        for item in result["order_summary"]["items"]:
            st.write(f"- {item['product_name']} (Size: {item['size']}, Màu: {item['color']}) x {item['quantity']}: {item['total']:,} VND")
        
        # Hiển thị tổng tiền
        st.write(f"Tạm tính: {result['order_summary']['subtotal']:,} VND")
        st.write(f"Phí vận chuyển: {result['order_summary']['shipping']:,} VND")
        st.markdown(f"### Tổng cộng: {result['order_summary']['total']:,} VND")
        
        # Nút tiếp tục mua sắm
        if st.button("Tiếp tục mua sắm"):
            st.session_state.order_result = None
            st.rerun()
    else:
        st.error(result.get("message", "Đã xảy ra lỗi khi đặt hàng"))
        
        # Nút thử lại
        if st.button("Thử lại"):
            st.session_state.order_result = None
            st.rerun()

def main():
    # Khởi tạo session state
    init_session_state()
    
    # Tạo UI
    st.title("👕 Clothing Store Chatbot")
    
    # Sidebar
    with st.sidebar:
        st.header("Quản lý giỏ hàng")
        cart_count = len(st.session_state.cart)
        st.write(f"Giỏ hàng ({cart_count} sản phẩm)")
        
        if cart_count > 0:
            if st.button("Xem giỏ hàng"):
                st.session_state.checkout = False
                st.rerun()
        
        st.divider()
        st.write("### Tìm kiếm sản phẩm")
        search_query = st.text_input("Nhập tên sản phẩm")
        if st.button("Tìm kiếm"):
            if search_query:
                user_message = f"Tìm kiếm sản phẩm: {search_query}"
                st.session_state.messages.append({"role": "user", "content": user_message})
                
                # Gửi tin nhắn đến chatbot
                response = chat_with_bot(user_message)
                
                # Lưu phản hồi
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["response"],
                    "additional_data": response.get("additional_data")
                })
                
                st.rerun()
        
        st.divider()
        st.write("### Thông tin liên hệ")
        st.write("Hotline: 1900.1234")
        st.write("Email: support@clothingstore.vn")
    
    # Main content
    if st.session_state.order_result:
        # Hiển thị kết quả đặt hàng
        display_order_result()
    elif st.session_state.checkout:
        # Hiển thị form thanh toán
        display_checkout_form()
    elif cart_count > 0 and len(st.session_state.messages) > 0 and st.session_state.messages[-1].get("role") == "user" and "xem giỏ hàng" in st.session_state.messages[-1].get("content", "").lower():
        # Hiển thị giỏ hàng khi người dùng yêu cầu
        display_cart()
    else:
        # Hiển thị lịch sử tin nhắn
        display_messages()
        
        # Khung chat
        if prompt := st.chat_input("Nhập tin nhắn của bạn..."):
            # Thêm tin nhắn của người dùng
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Hiển thị tin nhắn của người dùng
            st.chat_message("user").write(prompt)
            
            # Xử lý các lệnh đặc biệt
            if prompt.lower() in ["xem giỏ hàng", "giỏ hàng", "cart"]:
                # Hiển thị giỏ hàng
                with st.chat_message("assistant"):
                    if cart_count > 0:
                        st.write("Đây là giỏ hàng của bạn:")
                        display_cart()
                    else:
                        st.write("Giỏ hàng của bạn đang trống.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Đây là giỏ hàng của bạn." if cart_count > 0 else "Giỏ hàng của bạn đang trống."
                })
            else:
                # Gửi tin nhắn đến chatbot
                with st.spinner("Đang xử lý..."):
                    response = chat_with_bot(prompt)
                
                # Hiển thị phản hồi của chatbot
                with st.chat_message("assistant"):
                    st.write(response["response"])
                    
                    # Hiển thị thông tin bổ sung nếu có
                    if "additional_data" in response and response["additional_data"]:
                        if "products" in response["additional_data"]:
                            st.subheader("Sản phẩm liên quan:")
                            
                            # Hiển thị sản phẩm dạng lưới
                            cols = st.columns(min(3, len(response["additional_data"]["products"])))
                            for i, product in enumerate(response["additional_data"]["products"]):
                                with cols[i % 3]:
                                    st.image(DEFAULT_PRODUCT_IMAGE, width=150)
                                    st.markdown(f"**{product['product_name']}**")
                                    st.write(f"Giá: {product['price']:,} VND")
                                    st.write(f"Trạng thái: {'Còn hàng' if product['status'] == 'in_stock' else 'Hết hàng'}")
                                    
                                    # Tạo các selectbox cho size, màu sắc, và số lượng
                                    if product["status"] == "in_stock":
                                        # Lấy thông tin chi tiết về sản phẩm
                                        inventory = get_product_inventory(product["product_id"])
                                        
                                        if inventory["status"] != "error":
                                            sizes = inventory.get("sizes_available", [])
                                            colors = inventory.get("colors_available", [])
                                            
                                            selected_size = st.selectbox(f"Chọn size {i}", sizes, key=f"size_{product['product_id']}")
                                            selected_color = st.selectbox(f"Chọn màu {i}", colors, key=f"color_{product['product_id']}")
                                            selected_quantity = st.number_input(f"Số lượng {i}", min_value=1, max_value=10, value=1, key=f"quantity_{product['product_id']}")
                                            
                                            if st.button(f"Thêm vào giỏ hàng", key=f"add_to_cart_{product['product_id']}"):
                                                add_to_cart(
                                                    product["product_id"],
                                                    product["product_name"],
                                                    selected_size,
                                                    selected_color,
                                                    selected_quantity,
                                                    product["price"]
                                                )
                                                st.success(f"Đã thêm {product['product_name']} vào giỏ hàng!")
                
                # Lưu phản hồi
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["response"],
                    "additional_data": response.get("additional_data")
                })

if __name__ == "__main__":
    main() 