import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic.types import SecretStr

load_dotenv()
raw_api_groq = os.getenv("GROQ_API_KEY_OPENAI")
raw_api_qwen = os.getenv("GROQ_API_KEY_QWEN")
base_model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=SecretStr(raw_api_groq) if raw_api_groq is not None else None,
    temperature=0.5,
    timeout=30,
)

advanced_model = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=SecretStr(raw_api_qwen) if raw_api_qwen is not None else None,
    temperature=0.7,
)
