#!/usr/bin/env python3
"""
Script để kiểm thử chatbot.
"""

import os
import sys
import logging
import asyncio
import argparse
import json
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.agent.intent_detector import IntentDetector
from src.llm.model import LLMModel
from src.database.vector_store import VectorStore
from src.agent.response_generator import ResponseGenerator

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test-chatbot")

class ChatbotTester:
    """
    Lớp kiểm thử chatbot.
    """
    
    def __init__(self):
        """Khởi tạo các thành phần của chatbot."""
        self.llm_model = LLMModel()
        # Initialize intent detector with the LLM model
        self.intent_detector = IntentDetector(llm_model=self.llm_model)
        self.vector_store = VectorStore()
        self.response_generator = ResponseGenerator(self.llm_model, self.vector_store)
        self.conversation_id = f"test_{int(asyncio.get_event_loop().time())}"
        self.session_data = {}
        # Flag to control whether to use LLM-based intent detection
        self.use_llm_intent = True
        
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Xử lý tin nhắn và trả về kết quả.
        
        Args:
            message: Tin nhắn của người dùng
            
        Returns:
            Kết quả xử lý dạng dictionary
        """
        try:
            # Phát hiện intent - use LLM-based or regex-based detection based on flag
            if self.use_llm_intent:
                intent = await self.intent_detector.detect_intent_llm(message)
                logger.info(f"LLM detected intent: {intent} for message: {message}")
            else:
                intent = self.intent_detector.detect_intent(message)
                logger.info(f"Regex detected intent: {intent} for message: {message}")
            
            # Tạo câu trả lời
            response = await self.response_generator.generate_response(
                message, intent, self.session_data
            )
            
            return {
                "message": message,
                "intent": intent,
                "response": response,
                "conversation_id": self.conversation_id
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "message": message,
                "intent": "error",
                "response": config.DEFAULT_RESPONSES["fallback"],
                "conversation_id": self.conversation_id,
                "error": str(e)
            }
    
    async def interactive_mode(self):
        """Chạy chatbot ở chế độ tương tác."""
        print("\n=== Chatbot Test Mode ===")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'debug on' to enable debug mode.")
        print("Type 'debug off' to disable debug mode.")
        print("Type 'llm intent on' to use LLM-based intent detection.")
        print("Type 'llm intent off' to use regex-based intent detection.\n")
        
        debug_mode = False
        
        while True:
            try:
                # Nhận tin nhắn từ người dùng
                user_input = input("\nYou: ")
                
                # Kiểm tra lệnh thoát
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                
                # Kiểm tra lệnh debug
                if user_input.lower() == "debug on":
                    debug_mode = True
                    print("Debug mode enabled.")
                    continue
                
                if user_input.lower() == "debug off":
                    debug_mode = False
                    print("Debug mode disabled.")
                    continue
                
                # Toggle LLM intent detection
                if user_input.lower() == "llm intent on":
                    self.use_llm_intent = True
                    print("LLM-based intent detection enabled.")
                    continue
                
                if user_input.lower() == "llm intent off":
                    self.use_llm_intent = False
                    print("Regex-based intent detection enabled.")
                    continue
                
                # Xử lý tin nhắn
                result = await self.process_message(user_input)
                
                # Hiển thị kết quả
                print(f"\nBot: {result['response']}")
                
                # Hiển thị thông tin debug nếu được bật
                if debug_mode:
                    print("\n--- Debug Info ---")
                    print(f"Intent: {result['intent']}")
                    print(f"Intent Detection Method: {'LLM' if self.use_llm_intent else 'Regex'}")
                    print(f"Conversation ID: {result['conversation_id']}")
                    if "error" in result:
                        print(f"Error: {result['error']}")
                    print("------------------")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print(f"\nAn error occurred: {str(e)}")
    
    async def test_predefined_questions(self, questions: List[str]):
        """
        Kiểm thử chatbot với danh sách câu hỏi định sẵn.
        
        Args:
            questions: Danh sách câu hỏi
        """
        results = []
        
        for i, question in enumerate(questions):
            print(f"\nTesting question {i+1}/{len(questions)}: {question}")
            
            # Xử lý câu hỏi
            result = await self.process_message(question)
            
            # Hiển thị kết quả
            print(f"Intent: {result['intent']}")
            print(f"Response: {result['response']}")
            
            results.append(result)
        
        return results

async def main():
    parser = argparse.ArgumentParser(description="Test the chatbot functionality")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--questions", "-q", type=str, help="Path to JSON file with test questions")
    parser.add_argument("--output", "-o", type=str, help="Path to save test results")
    
    args = parser.parse_args()
    
    # Khởi tạo tester
    tester = ChatbotTester()
    
    if args.interactive:
        # Chế độ tương tác
        await tester.interactive_mode()
    elif args.questions:
        # Chế độ kiểm thử với câu hỏi định sẵn
        try:
            with open(args.questions, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            if not isinstance(questions, list):
                logger.error("Questions file must contain a JSON array of strings")
                return
            
            results = await tester.test_predefined_questions(questions)
            
            # Lưu kết quả nếu được yêu cầu
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Test results saved to {args.output}")
                
        except Exception as e:
            logger.error(f"Error testing with predefined questions: {str(e)}")
    else:
        # Mặc định chạy chế độ tương tác
        await tester.interactive_mode()

if __name__ == "__main__":
    asyncio.run(main()) 