import os
import json
import logging
import sys
from typing import Dict, Any, Optional, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger("clothing-chatbot")

class InventoryDB:
    """
    Quản lý truy vấn tồn kho sản phẩm từ cơ sở dữ liệu JSON.
    """
    
    def __init__(self):
        """Khởi tạo kết nối đến dữ liệu tồn kho."""
        self.inventory_path = os.path.join(config.DATA_DIR, "inventory", "inventory.json")
        self.inventory_data = self._load_inventory()
        
    def _load_inventory(self) -> Dict[str, Any]:
        """
        Tải dữ liệu tồn kho từ tệp JSON.
        
        Returns:
            Dữ liệu tồn kho dạng dictionary
        """
        try:
            if os.path.exists(self.inventory_path):
                with open(self.inventory_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded inventory data from {self.inventory_path}")
                return data
            else:
                logger.warning(f"Inventory file not found at {self.inventory_path}")
                return {"products": []}
        except Exception as e:
            logger.error(f"Error loading inventory data: {str(e)}")
            return {"products": []}
    
    def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin sản phẩm theo ID.
        
        Args:
            product_id: ID của sản phẩm
            
        Returns:
            Thông tin sản phẩm dạng dictionary
        """
        try:
            for product in self.inventory_data.get("products", []):
                if product.get("id") == product_id:
                    return product
            return {}
        except Exception as e:
            logger.error(f"Error getting product info: {str(e)}")
            return {}
    
    def get_product_inventory(self, product_id: str, size: Optional[str] = None, color: Optional[str] = None) -> Dict[str, Any]:
        """
        Kiểm tra tồn kho của sản phẩm.
        
        Args:
            product_id: ID của sản phẩm
            size: Kích cỡ (tùy chọn)
            color: Màu sắc (tùy chọn)
            
        Returns:
            Thông tin tồn kho dạng dictionary
        """
        try:
            product_info = self.get_product_info(product_id)
            if not product_info:
                return {"quantity": 0, "status": "not_found"}
            
            # Nếu không có size hoặc color, trả về thông tin tổng quát
            if not size or not color:
                sizes_available = []
                colors_available = set()
                total_quantity = 0
                
                for size_key, colors in product_info.get("sizes", {}).items():
                    sizes_available.append(size_key)
                    for color_key, quantity in colors.items():
                        colors_available.add(color_key)
                        total_quantity += quantity
                
                return {
                    "product_id": product_id,
                    "product_name": product_info.get("name", ""),
                    "total_quantity": total_quantity,
                    "sizes_available": sizes_available,
                    "colors_available": list(colors_available),
                    "status": "in_stock" if total_quantity > 0 else "out_of_stock"
                }
            
            # Kiểm tra tồn kho cụ thể theo size và color
            if size in product_info.get("sizes", {}):
                if color in product_info["sizes"][size]:
                    quantity = product_info["sizes"][size][color]
                    return {
                        "product_id": product_id,
                        "product_name": product_info.get("name", ""),
                        "size": size,
                        "color": color,
                        "quantity": quantity,
                        "status": "in_stock" if quantity > 0 else "out_of_stock"
                    }
            
            # Nếu không tìm thấy size/color cụ thể
            return {
                "product_id": product_id,
                "product_name": product_info.get("name", ""),
                "quantity": 0,
                "status": "variant_not_found"
            }
            
        except Exception as e:
            logger.error(f"Error getting product inventory: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def search_products(self, query: str) -> List[Dict[str, Any]]:
        """
        Tìm kiếm sản phẩm theo từ khóa.
        
        Args:
            query: Từ khóa tìm kiếm
            
        Returns:
            Danh sách sản phẩm phù hợp
        """
        try:
            query = query.lower()
            results = []
            
            for product in self.inventory_data.get("products", []):
                product_name = product.get("name", "").lower()
                product_category = product.get("category", "").lower()
                
                if query in product_name or query in product_category:
                    # Tính tổng số lượng
                    total_quantity = 0
                    for size_data in product.get("sizes", {}).values():
                        for quantity in size_data.values():
                            total_quantity += quantity
                    
                    results.append({
                        "product_id": product.get("id"),
                        "product_name": product.get("name"),
                        "category": product.get("category"),
                        "price": product.get("price"),
                        "total_quantity": total_quantity,
                        "status": "in_stock" if total_quantity > 0 else "out_of_stock"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching products: {str(e)}")
            return []
    
    def update_inventory(self, product_id: str, size: str, color: str, quantity_change: int) -> bool:
        """
        Cập nhật số lượng tồn kho (thêm hoặc bớt).
        
        Args:
            product_id: ID của sản phẩm
            size: Kích cỡ
            color: Màu sắc
            quantity_change: Số lượng thay đổi (dương là thêm, âm là bớt)
            
        Returns:
            True nếu cập nhật thành công, False nếu thất bại
        """
        try:
            for product in self.inventory_data.get("products", []):
                if product.get("id") == product_id:
                    if size in product.get("sizes", {}) and color in product["sizes"][size]:
                        current_quantity = product["sizes"][size][color]
                        new_quantity = current_quantity + quantity_change
                        
                        # Đảm bảo số lượng không âm
                        if new_quantity < 0:
                            new_quantity = 0
                            
                        product["sizes"][size][color] = new_quantity
                        
                        # Lưu lại dữ liệu
                        with open(self.inventory_path, 'w', encoding='utf-8') as f:
                            json.dump(self.inventory_data, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"Updated inventory for product {product_id} (size: {size}, color: {color}) to {new_quantity}")
                        return True
            
            logger.warning(f"Product variant not found for update: {product_id} (size: {size}, color: {color})")
            return False
            
        except Exception as e:
            logger.error(f"Error updating inventory: {str(e)}")
            return False 