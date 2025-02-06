import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


class State(TypedDict):
    prompts: list[str]
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int


async def call_llm(state: State):
    print(f"Step {state['iteration_count'] + 1}")
    prompt = state["prompts"][state["iteration_count"]]
    chat_messages = state["messages"].copy() + [HumanMessage(content=prompt)]
    chain = model | parser
    response = await chain.ainvoke(chat_messages)
    print(f"Response: {response}")
    return {
        "messages": [HumanMessage(content=prompt), AIMessage(content=response)],
        "iteration_count": state["iteration_count"] + 1,
    }


def should_continue(state: State) -> Literal["call_llm", END]:  # type: ignore
    if state["iteration_count"] < len(state["prompts"]):
        return "call_llm"
    else:
        return END


graph = StateGraph(State)
graph.add_node("call_llm", call_llm)
graph.set_entry_point("call_llm")
graph.add_conditional_edges("call_llm", should_continue)
graph.add_edge("call_llm", END)
graph.compile()
agent = graph.compile()


@traceable(name="prompt_chaining")
async def prompt_chaining(task: str, prompts: list[str]) -> str:
    response = await agent.ainvoke(
        {
            "prompts": prompts,
            "messages": [HumanMessage(content=task)],
            "iteration_count": 0,
        },
        debug=True,
    )
    return response["messages"][-1].content


async def main():
    task = "Sally earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
    prompts = [
        "Given the math problem, ONLY extract any relevant numerical information and how it can be used.",
        "Given the numberical information extracted, ONLY express the steps you would take to solve the problem.",
        "Given the steps, express the final answer to the problem.",
    ]
    response = await prompt_chaining(task, prompts)
    print(response)


asyncio.run(main())
