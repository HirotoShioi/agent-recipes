# https://www.agentrecipes.com/routing
# Conditional Router Agent Workflow
from typing import Annotated, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define node descriptions for routing
AVAILABLE_NODES = [
    {"id": "code_generation", "description": "Suited for code generation"},
    {"id": "trip_planner", "description": "Suited for trip planning"},
    {"id": "story_teller", "description": "Suited for story telling"},
]


class RouterState(TypedDict):
    """State for the router workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    input_query: str
    route: str
    response: str


class RouterSchema(TypedDict):
    """A schema for the router workflow."""

    reason: Annotated[str, ..., "The reason for selecting the route"]
    route: Annotated[
        str, ..., "The route selected. Must be one of the ids in the list of routes."
    ]


async def route_task(state: RouterState):
    """Route the task to the appropriate node."""
    ROUTER_PROMPT = """Given a user prompt/query: {user_query}, select the best option out of the following routes:
    {routes}. Answer only in JSON format."""
    prompt = PromptTemplate.from_template(ROUTER_PROMPT)
    model_routes_str = "\n".join(
        [f"id: {v['id']}, description: {v['description']}" for v in AVAILABLE_NODES]
    )
    chain = prompt | model.with_structured_output(
        RouterSchema, method="json_schema", strict=True
    )
    response = await chain.ainvoke(
        {"user_query": state["input_query"], "routes": model_routes_str}
    )

    return {
        "messages": [
            AIMessage(
                content=f"Selected route: {response['route']} because {response['reason']}"
            )
        ],
        "route": response["route"],
        "response": "",
    }


async def execute_node(state: RouterState, system_prompt: str) -> RouterState:
    """Common implementation for executing a node with a specific system prompt."""
    messages = [("system", system_prompt), ("user", state["input_query"])]
    parser = StrOutputParser()
    chain = model | parser
    response = await chain.ainvoke(messages)

    return {
        "messages": [AIMessage(content=response)],
        "response": response,
        "input_query": state["input_query"],
        "route": state["route"],
    }


async def code_generation(state: RouterState):
    """Generate code based on the input query."""
    return await execute_node(
        state,
        "You are a helpful assistant that generates code. Always provide clear, well-documented code with explanations.",
    )


async def trip_planner(state: RouterState):
    """Plan trips based on the input query."""
    return await execute_node(
        state,
        "You are a helpful travel planner. Provide detailed itineraries with practical recommendations.",
    )


async def story_teller(state: RouterState):
    """Tell stories based on the input query."""
    return await execute_node(
        state,
        "You are a creative storyteller. Create engaging and imaginative stories.",
    )


workflow = StateGraph(RouterState)

# Add nodes
workflow.add_node("router", route_task)
workflow.add_node("code_generation", code_generation)
workflow.add_node("trip_planner", trip_planner)
workflow.add_node("story_teller", story_teller)

# Add edges
workflow.set_entry_point("router")


def choose_node(state: RouterState):
    return state["route"]


workflow.add_conditional_edges(
    "router", choose_node, {node["id"]: node["id"] for node in AVAILABLE_NODES}
)

for node in AVAILABLE_NODES:
    workflow.add_edge(node["id"], END)

agent = workflow.compile(debug=True)


async def execute_task(task: str):
    return await agent.ainvoke(
        {
            "input_query": task,
        }
    )


async def main():
    tasks = [
        "Write a Python function to check if a number is prime.",
        "Plan a 2-week trip to Europe.",
        "Write a story about a brave knight and a dragon.",
    ]

    for task in tasks:
        response = await execute_task(task)
        print(response["response"])


if __name__ == "__main__":
    asyncio.run(main())
