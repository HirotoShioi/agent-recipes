from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.tools import tool
from langsmith import traceable

llm = ChatOpenAI(model="gpt-4o-mini")


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

llm_with_tools = llm.bind_tools(tools)


@traceable(name="tool_calling")
def main():
    query = "What is 3 * 12? Also, what is 11 + 49?"

    messages: list[BaseMessage] = [HumanMessage(query)]

    ai_msg = llm_with_tools.invoke(messages)

    assert isinstance(ai_msg, AIMessage), "Expected AIMessage but got different type"
    messages.append(ai_msg)
    tools_by_name = {t.name: t for t in tools}
    for tool_call in ai_msg.tool_calls:
        selected_tool = tools_by_name[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

    result = llm_with_tools.invoke(messages)

    print(result.content)


if __name__ == "__main__":
    main()
