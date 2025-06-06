from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def load_config():
    """Load environment variables"""
    load_dotenv()
    if not os.getenv("key"):
        raise ValueError("GOOGLE_API_KEY not found in .env file")

def setup_gemini_llm():
    """Configure and return Gemini LLM instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )