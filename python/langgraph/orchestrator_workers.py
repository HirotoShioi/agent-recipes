# https://www.agentrecipes.com/orchestrator-workers
# This workflow begins with an LLM breaking down the task into subtasks
# that are dynamically determined based on the input. These subtasks are
# then processed in parallel by multiple worker LLMs. Finally, the orchestrator
# LLM synthesizes the workers' outputs into the final result.
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing import Annotated, List, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
import asyncio
import operator
from langsmith import traceable

load_dotenv()


class Task(TypedDict):
    reasoning: Annotated[str, ..., "The reason for the task"]
    type: Annotated[
        Literal["Formal", "Conversational", "Hybrid"], ..., "The type of the task"
    ]
    description: Annotated[str, ..., "The description of the task"]


class TaskList(TypedDict):
    analysis: Annotated[str, ..., "The analysis of the task"]
    tasks: Annotated[List[Task], ..., "The approaches to tackle the task"]


class WorkflowState(TypedDict):
    """State for the orchestrator workflow."""

    input: str
    tasks: List[Task]
    analysis: str
    responses: Annotated[List[str], operator.add]


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


ORCHESTRATOR_PROMPT = """
Analyze this task and break it down into 2-3 distinct approaches:

Task: {task}

Provide an Analysis:

Explain your understanding of the task and which variations would be valuable.
Focus on how each approach serves different aspects of the task.

Along with the analysis, provide 2-3 approaches to tackle the task, each with a brief description:

Formal style: Write technically and precisely, focusing on detailed specifications
Conversational style: Write in a friendly and engaging way that connects with the reader
Hybrid style: Tell a story that includes technical details, combining emotional elements with specifications

Return only JSON output.
"""


@traceable(name="analyze_task")
async def analyze_task(state: WorkflowState):
    """Analyze the task and break it down into subtasks."""
    prompt = PromptTemplate.from_template(ORCHESTRATOR_PROMPT)
    chain = prompt | model.with_structured_output(
        TaskList, method="json_schema", strict=True
    )
    response = await chain.ainvoke({"task": state["input"]})
    return {
        "tasks": response["tasks"],
        "analysis": response["analysis"],
        "responses": [],
    }


class ProcessTaskState(TypedDict):
    task: Task
    original_task: str


WORKER_PROMPT = """
Generate content based on:
Task: {original_task}
Style: {task_type}
Guidelines: {task_description}

Return only your response:
[Your content here, maintaining the specified style and fully addressing requirements.]
"""


@traceable(name="process_task")
async def process_task(state: ProcessTaskState):
    """Process all tasks in parallel."""
    prompt = PromptTemplate.from_template(WORKER_PROMPT)
    chain = prompt | model | parser
    task: Task = state["task"]
    response = await chain.ainvoke(
        {
            "original_task": state["original_task"],
            "task_type": task["type"],
            "task_description": task["description"],
        }
    )

    print("\n=== WORKER OUTPUT ===")
    print(f"\nTask: {response}")
    return {"responses": [response]}


def map_tasks(state: WorkflowState):
    return [
        Send("process", {"task": task, "original_task": state["input"]})
        for task in state["tasks"]
    ]


# Create the workflow
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("analyze", analyze_task)
workflow.add_node("process", process_task)
# Add edges
workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", map_tasks, ["process"])
workflow.add_edge("process", END)

# Compile the workflow
agent = workflow.compile()


async def orchestrator_workers(task: str) -> list[str]:
    """Run the orchestrator workflow."""
    response = await agent.ainvoke(
        {
            "task": task,
        },
        debug=True,
    )
    return response["responses"]


async def main():
    task = """Write a product description for a new eco-friendly water bottle. 
    The target_audience is environmentally conscious millennials and key product
    features are: plastic-free, insulated, lifetime warranty
    """
    responses = await orchestrator_workers(task)
    print("\n=== FINAL OUTPUT ===")
    for response in responses:
        print(f"\nResponse: {response}")


if __name__ == "__main__":
    asyncio.run(main())
