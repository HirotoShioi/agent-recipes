# https://www.agentrecipes.com/orchestrator-workers
# This workflow begins with an LLM breaking down the task into subtasks
# that are dynamically determined based on the input. These subtasks are
# then processed in parallel by multiple worker LLMs. Finally, the orchestrator
# LLM synthesizes the workers' outputs into the final result.
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing import Annotated, List, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import asyncio
import json
from langsmith import traceable

load_dotenv()

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

WORKER_PROMPT = """
Generate content based on:
Task: {original_task}
Style: {task_type}
Guidelines: {task_description}

Return only your response:
[Your content here, maintaining the specified style and fully addressing requirements.]
"""

task = """Write a product description for a new eco-friendly water bottle.
The target_audience is environmentally conscious millennials and key product features are: plastic-free, insulated, lifetime warranty
"""


class Task(TypedDict):
    reasoning: Annotated[str, ..., "The reason for the task"]
    type: Annotated[
        Literal["Formal", "Conversational", "Hybrid"], ..., "The type of the task"
    ]
    description: Annotated[str, ..., "The description of the task"]


class TaskList(TypedDict):
    analysis: Annotated[str, ..., "The analysis of the task"]
    tasks: Annotated[List[Task], ..., "The approaches to tackle the task"]


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


@traceable(name="run_task")
async def run_task(task: Task) -> str:
    prompt = PromptTemplate.from_template(WORKER_PROMPT)
    chain = prompt | model | parser
    response = await chain.ainvoke(
        {
            "original_task": task["description"],
            "task_type": task["type"],
            "task_description": task["description"],
        }
    )
    return response


@traceable(name="orchestrator_workers")
async def orchestrator_workers(task: str) -> list[str]:
    prompt = PromptTemplate.from_template(ORCHESTRATOR_PROMPT)
    chain = prompt | model.with_structured_output(
        TaskList, method="json_schema", strict=True
    )
    response = await chain.ainvoke({"task": task})
    analysis = response["analysis"]
    tasks = response["tasks"]
    print("\n=== ORCHESTRATOR OUTPUT ===")
    print(f"\nAnalysis: {analysis}")
    print(f"\nTasks: {json.dumps(tasks, indent=2)}")
    return await asyncio.gather(*[run_task(task) for task in tasks])


async def main():
    task = """Write a product description for a new eco-friendly water bottle. 
    The target_audience is environmentally conscious millennials and key product
    features are: plastic-free, insulated, lifetime warranty
    """
    tasks = await orchestrator_workers(task)
    print("\n=== WORKER OUTPUT ===")
    for task in tasks:
        print(f"\nTask: {task}")


if __name__ == "__main__":
    asyncio.run(main())
