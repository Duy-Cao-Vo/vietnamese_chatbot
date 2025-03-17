# src/llm/model.py
import os
import sys
import logging
import asyncio
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import aiohttp

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
        self.mode = config.LLM_MODE  # "local" or "api"
        self.max_new_tokens = 512
        self.system_prompt = config.SYSTEM_PROMPT
        self.rag_prompt = config.RAG_PROMPT
        
        logger.info(f"Initializing LLM in {self.mode} mode")
        
        if self.mode == "local":
            # Local model configuration
            self.model_path = config.LLM_MODEL_PATH
            self.model_name = config.LLM_MODEL_NAME
            self.max_context_length = 4096  # Default for most smaller models
            
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
            self.max_context_length = 4000  # Default for GPT-3.5
            
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
            
            return await self._generate_with_api(prompt, system_prompt)
            
        # === LOCAL MODE ===
        else:
            if not self.model or not self.tokenizer:
                # Trả về câu trả lời mặc định nếu không có mô hình
                logger.warning("Using simulated response because model was not loaded")
                return f"Tôi đã nhận được yêu cầu của bạn: {prompt}. Tuy nhiên, mô hình LLM hiện không khả dụng."
            
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
    
    async def generate_rag_response(self, question: str, context: List[str]) -> str:
        """
        Tạo câu trả lời dựa trên RAG (Retrieval-Augmented Generation).
        
        Args:
            question: Câu hỏi của người dùng
            context: Danh sách các đoạn văn bản liên quan
            
        Returns:
            Câu trả lời từ mô hình
        """
        try:
            # In API mode, we prioritize accuracy by using as much context as possible
            if self.mode == "api":
                # API mode has larger context window, we can use more context data
                max_context_tokens = self.max_context_length - 800  # Reserve tokens for the question and completion
                
                # For API mode, we still need some basic truncation for extremely large contexts
                truncated_context = []
                current_tokens = 0
                
                for ctx in context:
                    # Simple token estimation for API mode
                    ctx_tokens = len(ctx.split()) * 1.5  # Approximate tokens
                    if current_tokens + ctx_tokens <= max_context_tokens:
                        truncated_context.append(ctx)
                        current_tokens += ctx_tokens
                    else:
                        # If we already have significant context, just stop adding more
                        if current_tokens > max_context_tokens * 0.5:
                            break
                            
                        # If this is the first context and it's too big, try to include part of it
                        if not truncated_context:
                            # Simple word-based truncation for API
                            words = ctx.split()
                            remaining_tokens = max_context_tokens - current_tokens
                            max_words = int(remaining_tokens / 1.5)
                            truncated_text = " ".join(words[:max_words])
                            truncated_context.append(truncated_text)
                        break
                
                # Combine context and prioritize accuracy
                combined_context = "\n\n".join(truncated_context)
                
            # For local mode, keep original careful truncation
            else:
                if not self.tokenizer:
                    logger.warning("No tokenizer available for accurate token counting")
                    # Fallback to word-based estimation
                    question_tokens = len(question.split()) * 1.5
                else:
                    question_tokens = len(self.tokenizer.encode(question))
                
                prompt_template_tokens = 200  # Rough estimate for your templates
                
                # Calculate max tokens for context (with safety margin)
                max_context_tokens = self.max_context_length - int(question_tokens) - prompt_template_tokens - 50
                
                # Truncate context carefully for local models
                truncated_context = []
                current_tokens = 0
                
                for ctx in context:
                    if self.tokenizer:
                        ctx_tokens = len(self.tokenizer.encode(ctx))
                    else:
                        ctx_tokens = len(ctx.split()) * 1.5
                    
                    if current_tokens + ctx_tokens <= max_context_tokens:
                        truncated_context.append(ctx)
                        current_tokens += ctx_tokens
                    else:
                        if not truncated_context:
                            if self.tokenizer:
                                truncated_text = self.tokenizer.decode(
                                    self.tokenizer.encode(ctx)[:max_context_tokens]
                                )
                            else:
                                words = ctx.split()
                                max_words = int(max_context_tokens / 1.5)
                                truncated_text = " ".join(words[:max_words])
                            
                            truncated_context.append(truncated_text)
                        break
                
                combined_context = "\n\n".join(truncated_context)
            
            # Create RAG prompt with all included context
            rag_prompt = self.rag_prompt.format(
                question=question,
                context=combined_context
            )
            
            # Generate comprehensive, accurate response
            response = await self.generate(rag_prompt, self.system_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return f"Xin lỗi, tôi không thể trả lời câu hỏi của bạn lúc này. Lỗi: {str(e)}"