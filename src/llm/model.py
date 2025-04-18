# src/llm/model.py
import os
import sys
import logging
import asyncio
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import aiohttp
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger("clothing-chatbot")

class LLMModel:
    """
    Tích hợp với mô hình ngôn ngữ (local hoặc API) để tạo ra các câu trả lời.
    """
    
    def __init__(self):
        """Khởi tạo mô hình LLM dựa trên cấu hình."""
        # Common configuration
        self.mode = config.LLM_MODE  # "local", "api", or "dummy"
        self.max_new_tokens = 1024
        self.system_prompt = config.SYSTEM_PROMPT
        self.rag_prompt = config.RAG_PROMPT
        
        logger.info(f"Initializing LLM in {self.mode} mode")
        
        if self.mode == "local":
            # Local model configuration
            self.model_path = config.LLM_MODEL_PATH
            self.model_name = config.LLM_MODEL_NAME
            self.max_context_length = 8192
            
            # Check for Apple Silicon GPU
            if torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            elif torch.cuda.is_available():
                self.device = "cuda"  # NVIDIA GPU
            else:
                self.device = "cpu"  # Fallback to CPU
            
            # Initialize the local model
            self._init_local_model()
            
        elif self.mode == "api":
            # API configuration
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.api_base = config.API_BASE_URL
            self.api_version = config.API_VERSION
            self.api_engine = config.API_ENGINE
            self.api_timeout = config.API_TIMEOUT
            self.max_context_length = 16000  # Default for GPT-3.5
            
            # Validate API configuration
            if not self.api_key:
                logger.error("No OpenAI API key found in environment variables")
                self.api_available = False
                logger.warning("API mode selected but no API key available")
            else:
                self.api_available = True
                logger.info(f"API configured with engine: {self.api_engine}")
                
            # Set model and tokenizer to None for API mode
            self.model = None
            self.tokenizer = None
        
        elif self.mode == "dummy":
            # Dummy mode for simple rule-based responses
            logger.info("Using dummy mode for LLM responses")
            self.model = None
            self.tokenizer = None
            
        else:
            logger.error(f"Invalid LLM_MODE in config: {self.mode}")
            self.model = None
            self.tokenizer = None
    
    def _init_local_model(self):
        """Initialize the local model."""
        try:
            # Khởi tạo tokenizer và model
            logger.info(f"Initializing local LLM model: {self.model_name} on {self.device}")
            
            if os.path.exists(self.model_path):
                # Sử dụng mô hình local
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                # Move model to the appropriate device after loading
                if self.device != "cpu":
                    self.model = self.model.to(self.device)
                logger.info(f"Loaded local model from {self.model_path}")
            else:
                # Tải mô hình từ Hugging Face Hub
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                # Move model to the appropriate device after loading
                if self.device != "cpu":
                    self.model = self.model.to(self.device)
                logger.info(f"Loaded model from Hugging Face Hub: {self.model_name}")
                
                # Lưu mô hình
                os.makedirs(self.model_path, exist_ok=True)
                self.tokenizer.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path)
                logger.info(f"Saved model to {self.model_path}")
        
        except Exception as e:
            logger.error(f"Error loading local LLM model: {str(e)}")
            self.tokenizer = None
            self.model = None
            logger.warning("Using simulated LLM responses due to model loading failure")
    
    async def _generate_with_model(self, inputs):
        """Generate with local model."""
        with torch.no_grad():
            return self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
    
    async def _generate_with_api(self, prompt: str, system_prompt: str) -> str:
        """Generate with API."""
        try:
            # Tạo messages array cho API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Use aiohttp for async API call
            async with aiohttp.ClientSession() as session:
                try:
                    # API request payload
                    payload = {
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": self.max_new_tokens,
                        "top_p": 0.9,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    }
                    
                    headers = {
                        "api-key": self.api_key,
                        "Content-Type": "application/json"
                    }
                    
                    # Format URL based on API version and endpoint
                    url = f"{self.api_base}/chat/completions?api-version={self.api_version}"
                    
                    # Make API request with timeout
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.api_timeout
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                        else:
                            error_text = await response.text()
                            logger.error(f"API error: {response.status} - {error_text}")
                            return f"Xin lỗi, có lỗi khi gọi API: {response.status}"
                
                except asyncio.TimeoutError:
                    logger.warning(f"API request timed out after {self.api_timeout} seconds")
                    return "Xin lỗi, API không phản hồi trong thời gian cho phép. Vui lòng thử lại sau."
        
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            return f"Xin lỗi, không thể kết nối với API. Lỗi: {str(e)}"
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Tạo câu trả lời từ mô hình LLM (local hoặc API).
        
        Args:
            prompt: Câu hỏi hoặc prompt đầu vào
            system_prompt: System prompt (tùy chọn)
            
        Returns:
            Câu trả lời từ mô hình
        """
        # Sử dụng system prompt mặc định nếu không được cung cấp
        if not system_prompt:
            system_prompt = "Bạn là trợ lý AI hữu ích, thân thiện và trung thực."
        
        # === API MODE ===
        if self.mode == "api":
            if not self.api_available:
                logger.warning("Using simulated response because API key was not provided")
                return f"Tôi đã nhận được yêu cầu của bạn: {prompt}. Tuy nhiên, OpenAI API hiện không khả dụng."
            logger.info(f"Generating response with API: {prompt}")
            return await self._generate_with_api(prompt, system_prompt)
            
        # === DUMMY MODE ===
        elif self.mode == "dummy":
            return await self._generate_dummy_response(prompt, system_prompt)
            
        # === LOCAL MODE ===
        else:
            if not self.model or not self.tokenizer:
                # Trả về câu trả lời mặc định nếu không có mô hình
                logger.warning("Using simulated response because model was not loaded")
                return await self._generate_dummy_response(prompt, system_prompt)
            
            try:
                # Tạo prompt đầy đủ
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                # Đặt input_ids
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
                
                # Set a timeout for generation
                timeout_seconds = 60  # 1 minute timeout
                
                # Create an asyncio task
                generation_task = asyncio.create_task(self._generate_with_model(inputs))
                
                # Wait for the task with timeout
                try:
                    outputs = await asyncio.wait_for(generation_task, timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    logger.warning(f"Generation timed out after {timeout_seconds} seconds")
                    return "Xin lỗi, việc xử lý câu trả lời mất quá nhiều thời gian. Vui lòng thử câu hỏi ngắn hơn hoặc đơn giản hơn."
                
                # Giải mã kết quả
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Trích xuất phần trả lời
                assistant_text = generated_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
                
                return assistant_text
            
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return f"Xin lỗi, tôi không thể xử lý yêu cầu của bạn lúc này. Lỗi: {str(e)}"
                
    async def _generate_dummy_response(self, prompt: str, system_prompt: str) -> str:
        """
        Generate a rule-based dummy response for simple intent detection and other basic tasks.
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt
            
        Returns:
            A simple rule-based response
        """
        # For intent detection queries
        if "xác định ý định" in prompt.lower() or "intent" in prompt.lower():
            query_text = ""
            
            # Extract the user query from the prompt
            if "tin nhắn của người dùng:" in prompt.lower():
                query_text = prompt.lower().split("tin nhắn của người dùng:")[-1].strip().strip('"')
            
            # Product ID pattern (detect QK008 type codes)
            product_id_pattern = r'[a-zA-Z]{2,3}\d{3,4}'
            has_product_id = re.search(product_id_pattern, query_text)
            
            # Simple rule-based intent detection
            if has_product_id and any(word in query_text for word in ["còn", "size", "có", "hàng"]):
                # Product ID + availability question = inventory
                return "inventory"
            elif any(word in query_text for word in ["xin chào", "chào", "hello", "hi"]):
                return "greeting"
            elif any(word in query_text for word in ["còn hàng", "còn size", "size nào", "tồn kho"]):
                return "inventory"
            elif any(word in query_text for word in ["giá", "mua", "đặt hàng", "thanh toán"]):
                return "purchase"
            elif any(word in query_text for word in ["chính sách", "đổi trả", "bảo hành"]):
                return "policy"
            elif any(word in query_text for word in ["cửa hàng", "chi nhánh", "địa chỉ"]):
                return "store"
            elif any(word in query_text for word in ["áo", "quần", "váy", "giày", "size"]):
                # Check if this is an inventory query
                if any(word in query_text for word in ["còn", "hết", "có sẵn"]):
                    return "inventory"
                return "product"
            else:
                return "general"
                
        # For other queries, return a simple acknowledgment
        return "Đã nhận được yêu cầu của bạn. Đây là chế độ dummy, không sử dụng mô hình LLM thật."
    
    async def generate_rag_response(self, question: str, context: List[str]) -> str:
        """
        Tạo câu trả lời dựa trên RAG (Retrieval-Augmented Generation).
        """
        try:
            # Get token counter based on mode
            if self.mode == "api":
                try:
                    import tiktoken
                    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    def count_tokens(text):
                        return len(enc.encode(text))
                except ImportError:
                    logger.warning("tiktoken not available, using rough estimation")
                    def count_tokens(text):
                        return len(text.split()) * 1.3
            else:
                # Use tokenizer if available, otherwise estimate
                if self.tokenizer:
                    def count_tokens(text):
                        return len(self.tokenizer.encode(text))
                else:
                    def count_tokens(text):
                        return len(text.split()) * 1.3
            
            # Calculate token usage
            question_tokens = count_tokens(question)
            system_tokens = count_tokens(self.system_prompt)
            rag_template_tokens = count_tokens(self.rag_prompt.format(question="", context=""))
            
            # Calculate max tokens for context
            reserve_tokens = 800  # Reserve for completion and safety margin
            max_context_tokens = self.max_context_length - question_tokens - system_tokens - rag_template_tokens - reserve_tokens
            
            # Early check for impossible situation
            if max_context_tokens <= 0:
                logger.warning(f"Question too long, leaving no room for context: {question_tokens} tokens")
                return "Xin lỗi, câu hỏi của bạn quá dài. Vui lòng rút ngắn câu hỏi để nhận được trả lời tốt hơn."
            
            # Truncate context carefully
            truncated_context = []
            current_tokens = 0
            
            for ctx in context:
                ctx_tokens = count_tokens(ctx)
                
                # If this single context is already too big, need to truncate it
                if ctx_tokens > max_context_tokens and not truncated_context:
                    # Special case: need to truncate first chunk
                    if self.mode == "api":
                        # For API, we can encode/decode to truncate by exact tokens
                        try:
                            tokens = enc.encode(ctx)[:max_context_tokens]
                            truncated_text = enc.decode(tokens)
                        except:
                            # Fallback to rough word-based truncation
                            words = ctx.split()
                            safe_word_count = int(max_context_tokens / 1.3)
                            truncated_text = " ".join(words[:safe_word_count])
                    else:
                        # For local model use tokenizer if available
                        if self.tokenizer:
                            tokens = self.tokenizer.encode(ctx)[:max_context_tokens]
                            truncated_text = self.tokenizer.decode(tokens)
                        else:
                            words = ctx.split()
                            safe_word_count = int(max_context_tokens / 1.3)
                            truncated_text = " ".join(words[:safe_word_count])
                    
                    truncated_context.append(truncated_text)
                    current_tokens = count_tokens(truncated_text)
                    continue
                
                # For normal sized contexts, add if there's room
                if current_tokens + ctx_tokens <= max_context_tokens:
                    truncated_context.append(ctx)
                    current_tokens += ctx_tokens
                else:
                    # If we have enough context already, stop
                    if current_tokens > 0:
                        break
            
            # Combine truncated context
            combined_context = "\n\n".join(truncated_context)
            final_context_tokens = count_tokens(combined_context)
            
            logger.info(f"Using {final_context_tokens} tokens for context, max allowed: {max_context_tokens}")
            
            # Create RAG prompt with all included context
            rag_prompt = self.rag_prompt.format(
                question=question,
                context=combined_context
            )
            
            # Generate response
            response = await self.generate(rag_prompt, self.system_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return f"Xin lỗi, tôi không thể trả lời câu hỏi của bạn lúc này. Lỗi: {str(e)}"