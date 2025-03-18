import os
import sys
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
import aiohttp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger("clothing-chatbot")

class InventoryService:
    """
    Service to interact with inventory database/API directly for real-time data.
    """
    
    def __init__(self):
        """Initialize the inventory service with configuration."""
        # Database connection config
        self.db_config = {
            "host": config.INVENTORY_DB_HOST if hasattr(config, "INVENTORY_DB_HOST") else "localhost",
            "port": config.INVENTORY_DB_PORT if hasattr(config, "INVENTORY_DB_PORT") else 5432,
            "user": config.INVENTORY_DB_USER if hasattr(config, "INVENTORY_DB_USER") else "postgres",
            "password": config.INVENTORY_DB_PASSWORD if hasattr(config, "INVENTORY_DB_PASSWORD") else "",
            "database": config.INVENTORY_DB_NAME if hasattr(config, "INVENTORY_DB_NAME") else "inventory"
        }
        
        # API config
        self.api_config = {
            "base_url": config.INVENTORY_API_URL if hasattr(config, "INVENTORY_API_URL") else "http://localhost:8000/api",
            "timeout": config.INVENTORY_API_TIMEOUT if hasattr(config, "INVENTORY_API_TIMEOUT") else 10,
            "api_key": os.environ.get("INVENTORY_API_KEY", "")
        }
        
        # Set the connection mode: 'db', 'api', or 'dummy'
        self.mode = config.INVENTORY_MODE if hasattr(config, "INVENTORY_MODE") else "dummy"
        
        logger.info(f"Initialized inventory service in {self.mode} mode")
    
    async def check_inventory(self, product_id: Optional[str] = None, 
                             product_name: Optional[str] = None,
                             size: Optional[str] = None, 
                             color: Optional[str] = None) -> Dict[str, Any]:
        """
        Check inventory for a specific product.
        
        Args:
            product_id: ID of the product to check
            product_name: Name of the product to check
            size: Size to filter by
            color: Color to filter by
            
        Returns:
            Inventory data for the requested product
        """
        try:
            if self.mode == "db":
                return await self._check_inventory_db(product_id, product_name, size, color)
            elif self.mode == "api":
                return await self._check_inventory_api(product_id, product_name, size, color)
            else:
                # Dummy mode - use local JSON file
                logger.info("Using dummy mode for inventory check")
                return await self._check_inventory_dummy(product_id, product_name, size, color)
        except Exception as e:
            logger.error(f"Error checking inventory: {str(e)}")
            return {}
    
    async def get_all_inventory(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all inventory items.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of inventory items
        """
        try:
            if self.mode == "db":
                return await self._get_all_inventory_db(limit)
            elif self.mode == "api":
                return await self._get_all_inventory_api(limit)
            else:
                # Dummy mode - use local JSON file
                logger.info("Using dummy mode for inventory check")
                return await self._get_all_inventory_dummy(limit)
        except Exception as e:
            logger.error(f"Error getting all inventory: {str(e)}")
            return []
    
    async def _check_inventory_db(self, product_id, product_name, size, color):
        """Query the database for inventory data."""
        logger.info(f"Querying database for product {product_id or product_name}")
        
        # Here you would implement your database connection code
        # Example using asyncpg:
        """
        import asyncpg
        
        conn = await asyncpg.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            database=self.db_config["database"]
        )
        
        # Build query based on parameters
        query = "SELECT * FROM inventory WHERE "
        params = []
        
        if product_id:
            query += "product_id = $1"
            params.append(product_id)
        elif product_name:
            query += "product_name ILIKE $1"
            params.append(f"%{product_name}%")
        else:
            return {}
        
        result = await conn.fetchrow(query, *params)
        await conn.close()
        
        if result:
            # Transform database record to expected format
            return {
                "id": result["product_id"],
                "name": result["product_name"],
                "category": result["category"],
                "price": result["price"],
                "sizes": json.loads(result["sizes_json"])
            }
        """
        
        # Placeholder - replace with actual DB query
        logger.warning("Database query not implemented, falling back to dummy data")
        return await self._check_inventory_dummy(product_id, product_name, size, color)
    
    async def _check_inventory_api(self, product_id, product_name, size, color):
        """Query the API for inventory data."""
        logger.info(f"Querying API for product {product_id or product_name}")
        
        try:
            params = {}
            endpoint = "inventory"
            
            if product_id:
                endpoint = f"inventory/{product_id}"
            elif product_name:
                params["name"] = product_name
            
            if size:
                params["size"] = size
            if color:
                params["color"] = color
                
            url = f"{self.api_config['base_url']}/{endpoint}"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_config['api_key']}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    params=params,
                    headers=headers,
                    timeout=self.api_config['timeout']
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        logger.error(f"API error: {response.status} - {await response.text()}")
                        
        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"API timeout after {self.api_config['timeout']} seconds")
            
        # Fallback to dummy data on error
        logger.warning("API request failed, falling back to dummy data")
        return await self._check_inventory_dummy(product_id, product_name, size, color)
    
    async def _check_inventory_dummy(self, product_id, product_name, size, color):
        """Use local JSON file for dummy data."""
        try:
            # Path to inventory JSON file
            inventory_path = os.path.join(config.DATA_DIR, "inventory", "inventory.json")
            
            if not os.path.exists(inventory_path):
                logger.error(f"Inventory file not found: {inventory_path}")
                return {}
                
            with open(inventory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find the requested product
            for product in data.get("products", []):
                if product_id and product.get("id") == product_id:
                    return self._filter_product_by_size_color(product, size, color)
                elif product_name and product_name.lower() in product.get("name", "").lower():
                    return self._filter_product_by_size_color(product, size, color)
            
            return {}
                
        except Exception as e:
            logger.error(f"Error reading dummy inventory data: {str(e)}")
            return {}
    
    async def _get_all_inventory_db(self, limit):
        """Query the database for all inventory items."""
        logger.info(f"Querying database for all inventory (limit {limit})")
        
        # Here you would implement your database connection code
        # Placeholder - replace with actual DB query
        logger.warning("Database query not implemented, falling back to dummy data")
        return await self._get_all_inventory_dummy(limit)
    
    async def _get_all_inventory_api(self, limit):
        """Query the API for all inventory items."""
        logger.info(f"Querying API for all inventory (limit {limit})")
        
        try:
            url = f"{self.api_config['base_url']}/inventory"
            params = {"limit": limit}
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_config['api_key']}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    params=params,
                    headers=headers,
                    timeout=self.api_config['timeout']
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        logger.error(f"API error: {response.status} - {await response.text()}")
                        
        except aiohttp.ClientError as e:
            logger.error(f"API connection error: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"API timeout after {self.api_config['timeout']} seconds")
            
        # Fallback to dummy data on error
        logger.warning("API request failed, falling back to dummy data")
        return await self._get_all_inventory_dummy(limit)
    
    async def _get_all_inventory_dummy(self, limit):
        """Use local JSON file for dummy data."""
        try:
            # Path to inventory JSON file
            inventory_path = os.path.join(config.DATA_DIR, "inventory", "inventory.json")
            
            if not os.path.exists(inventory_path):
                logger.error(f"Inventory file not found: {inventory_path}")
                return []
                
            with open(inventory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Return all products up to the limit
            return data.get("products", [])[:limit]
                
        except Exception as e:
            logger.error(f"Error reading dummy inventory data: {str(e)}")
            return []
    
    def _filter_product_by_size_color(self, product, size=None, color=None):
        """Filter product inventory by size and color."""
        # If no filters, return the entire product
        if not size and not color:
            return product
            
        # Make a copy to avoid modifying the original
        filtered_product = product.copy()
        
        # If size filter is applied
        if size and "sizes" in filtered_product:
            # Keep only the requested size
            if size in filtered_product["sizes"]:
                filtered_product["sizes"] = {size: filtered_product["sizes"][size]}
            else:
                filtered_product["sizes"] = {}
        
        # If color filter is applied
        if color and "sizes" in filtered_product:
            # For each size, keep only the requested color
            for s in list(filtered_product["sizes"].keys()):
                if color in filtered_product["sizes"][s]:
                    filtered_product["sizes"][s] = {color: filtered_product["sizes"][s][color]}
                else:
                    filtered_product["sizes"][s] = {}
        
        return filtered_product 