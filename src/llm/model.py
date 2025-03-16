import os
import sys
import logging
import asyncio
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger("clothing-chatbot")

class LLMModel:
    """
    Tích hợp với mô hình ngôn ngữ DeepSeek R1 để tạo ra các câu trả lời.
    """
    
    def __init__(self):
        """Khởi tạo và tải mô hình LLM."""
        self.model_path = config.LLM_MODEL_PATH
        self.model_name = config.LLM_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_context_length = 2048
        self.max_new_tokens = 512
        
        # Các prompt mẫu
        self.system_prompt = config.SYSTEM_PROMPT
        self.rag_prompt = config.RAG_PROMPT
        
        # Khởi tạo tokenizer và model
        logger.info(f"Initializing LLM model: {self.model_name} on {self.device}")
        
        try:
            # Kiểm tra xem mô hình đã được tải chưa
            if os.path.exists(self.model_path):
                # Sử dụng mô hình local
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    load_in_4bit=True,
                    trust_remote_code=True
                )
                logger.info(f"Loaded local model from {self.model_path}")
            else:
                # Tải mô hình từ Hugging Face Hub
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                logger.info(f"Loaded model from Hugging Face Hub: {self.model_name}")
                
                # Lưu mô hình
                os.makedirs(self.model_path, exist_ok=True)
                self.tokenizer.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path)
                logger.info(f"Saved model to {self.model_path}")
        
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            # Giả lập một LLM model đơn giản nếu không tải được mô hình
            self.tokenizer = None
            self.model = None
            logger.warning("Using simulated LLM responses due to model loading failure")
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Tạo câu trả lời từ mô hình LLM.
        
        Args:
            prompt: Câu hỏi hoặc prompt đầu vào
            system_prompt: System prompt (tùy chọn)
            
        Returns:
            Câu trả lời từ mô hình
        """
        if not self.model or not self.tokenizer:
            # Trả về câu trả lời mặc định nếu không có mô hình
            logger.warning("Using simulated response because model was not loaded")
            return f"Tôi đã nhận được yêu cầu của bạn: {prompt}. Tuy nhiên, mô hình LLM hiện không khả dụng."
        
        try:
            # Sử dụng system prompt mặc định nếu không được cung cấp
            if not system_prompt:
                system_prompt = "Bạn là trợ lý AI hữu ích, thân thiện và trung thực."
            
            # Tạo prompt đầy đủ
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Đặt input_ids
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            # Tạo ra token
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
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
            # Kết hợp ngữ cảnh
            combined_context = "\n\n".join(context)
            
            # Tạo prompt RAG
            rag_prompt = self.rag_prompt.format(
                question=question,
                context=combined_context
            )
            
            # Tạo câu trả lời
            response = await self.generate(rag_prompt, self.system_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return f"Xin lỗi, tôi không thể trả lời câu hỏi của bạn lúc này. Lỗi: {str(e)}" 