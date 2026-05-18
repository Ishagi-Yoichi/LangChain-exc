from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

load_dotenv()

@tool
def get_weather(city: str)-> str:
    """ Get weather for a given city"""
    return f"It's sunny in {city}"

model = ChatGroq(
       model="openai/gpt-oss-120b",

       temperature=0.5,
       timeout=30,
)

agent = create_agent(
    model = model,
    tools = [get_weather],
    system_prompt = "You are a helpful assistant",

)
print("Sending request to OpenRouter...")

try:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in san francisco?"}]}
    )
    print("\n--- Final Result ---")
    print(result["messages"][-1].content_blocks)
except Exception as e:
    print(f"\nAn error occurred: {e}")
