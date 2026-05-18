from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

base_model = ChatGroq(
       model="openai/gpt-oss-120b",
       api_key=os.getenv("GROQ_API_KEY_OPENAI"),
       temperature=0.5,
       timeout=30,
)

advanced_model= ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY_QWEN"),
    temperature=0.7
)
