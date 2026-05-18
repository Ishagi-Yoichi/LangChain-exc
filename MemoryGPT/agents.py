from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from models import base_model, advanced_model



@tool
def get_weather(city: str)-> str:
    """ Get weather for a given city"""
    return f"It's sunny in {city}"



@wrap_model_call
def dynamic_model_selection(request:ModelRequest, handler) -> ModelResponse:
    """"Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 2:
        model = advanced_model
    else:
        model = base_model
    return handler(request.override(model=model))

agent = create_agent(
    model = base_model,
    tools = [get_weather],
    middleware=[dynamic_model_selection],
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
