from typing import Any, Callable, List, TypedDict, Union

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage

from models import advanced_model, base_model


class CustomAgentState(TypedDict):
    messages: List[Union[AnyMessage, dict]]
    authenticated: bool


@tool
def public_get_weather(city: str) -> str:
    """Get basic weather information for a public city."""
    return f"It's sunny in {city}"


@tool
def authenticated_get_user_profile(user_id: str) -> str:
    """Retrieve confidential customer account profile metrics."""
    return f"Profile metrics for {user_id}: Active subscriber."


@tool
def advanced_search(complex_query: str) -> str:
    """Execute deep vector indexing across historical analytics archives."""
    return f"Deep search results for: '{complex_query}'"


MASTER_TOOLS = [public_get_weather, authenticated_get_user_profile, advanced_search]


def _get_tool_name(t: Any) -> str:
    if hasattr(t, "name"):
        return str(t.name)
    if isinstance(t, dict):
        if "name" in t:
            return str(t["name"])
        if "function" in t and isinstance(t["function"], dict):
            return str(t["function"].get("name", ""))
    return ""


@wrap_model_call
def state_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:

    state = request.state if request.state is not None else {}
    messages = state.get("messages", [])
    is_authenticated = state.get("authenticated", False)
    msg_count = len(messages)

    filtered_tools: List[Union[Any, dict[str, Any]]] = list(MASTER_TOOLS)

    if not is_authenticated:
        filtered_tools = [
            t for t in filtered_tools if _get_tool_name(t).startswith("public_")
        ]

    if msg_count < 5:
        filtered_tools = [
            t for t in filtered_tools if _get_tool_name(t) != "advanced_search"
        ]

    visible_names = [_get_tool_name(t) for t in filtered_tools]
    print(f"   [Middleware Check] Exposed Tools: {visible_names}")

    request = request.override(tools=filtered_tools)
    return handler(request)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    state = request.state if request.state is not None else {}
    messages = state.get("messages", [])
    message_count = len(messages)

    if message_count > 2:
        model = advanced_model
    else:
        model = base_model
    return handler(request.override(model=model))


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )


agent = create_agent(
    model=base_model,
    tools=MASTER_TOOLS,
    state_schema=CustomAgentState,
    middleware=[state_based_tools, dynamic_model_selection, handle_tool_errors],
    system_prompt="You are a helpful assistant",
)

if __name__ == "__main__":
    print("=== STARTING DYNAMIC TOOL SELECTION VERIFICATION TESTS ===\n")

    print("--- TEST 1: Unauthenticated User Request ---")
    input_unauth: CustomAgentState = {
        "authenticated": False,
        "messages": [HumanMessage(content="What's the weather in San Francisco?")],
    }
    try:
        agent.invoke(input_unauth)
    except Exception:
        pass

    print("\n--- TEST 2: Authenticated User (Short Conversation < 5 messages) ---")
    input_auth_short: CustomAgentState = {
        "authenticated": True,
        "messages": [
            HumanMessage(content="Check internal profile metrics for user_123")
        ],
    }
    try:
        agent.invoke(input_auth_short)
    except Exception:
        pass

    print("\n--- TEST 3: Authenticated User (Long Conversation >= 5 messages) ---")
    input_auth_long: CustomAgentState = {
        "authenticated": True,
        "messages": [
            HumanMessage(content="Hi"),
            HumanMessage(content="How are you"),
            HumanMessage(content="Need some technical assistance"),
            HumanMessage(content="Let's look into our deep storage analytics indices"),
            HumanMessage(content="Run an advanced search on system logs"),
        ],
    }
    try:
        agent.invoke(input_auth_long)
    except Exception:
        pass
