from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph.message import add_messages
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Define the state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    feedbacks: Annotated[list[str], add_messages]
    code: str
    iteration_count: int


# Prompt templates
GENERATOR_PROMPT = """You are an expert software engineer tasked with generating high-quality code solutions.

TASK DESCRIPTION:
{task}

REQUIREMENTS:
1. Write clean, efficient, and well-documented code
2. Include appropriate error handling
3. Follow language-specific best practices and conventions
4. Consider edge cases in your implementation
5. Optimize for both readability and performance

PREVIOUS FEEDBACK:
If there is feedback from previous iterations, carefully consider and address each point in your next solution.

YOUR RESPONSE SHOULD:
1. Explain your approach and any important design decisions
2. Highlight any assumptions you're making
3. Provide the implementation with clear comments
4. Note any potential improvements or alternative approaches

Remember to maintain a balance between code quality, performance, and maintainability.
"""


class GenerateCodeOutput(TypedDict):
    thoughts: Annotated[
        str,
        ...,
        "Your understanding of the task and feedback and how you plan to improve",
    ]
    code: Annotated[str, ..., "The code you generated"]


# Node functions
async def generate_code(state: AgentState):
    message = ChatPromptTemplate.from_messages(
        [("system", GENERATOR_PROMPT), ("human", "Feedbacks: {feedbacks}")]
    )
    # Prepare the generation prompt
    generation_chain = message | model.with_structured_output(
        GenerateCodeOutput, method="json_schema", strict=True
    )

    # Generate code
    response = await generation_chain.ainvoke(
        {
            "task": state["task"],
            "feedbacks": [f"\nFeedback: {feedback}" for feedback in state["feedbacks"]],
            "code": state["code"],
        }
    )

    # Prepare the new message with thoughts and code
    content = f"""
    Thoughts: {response['thoughts']}
    Code: {response['code']}
    """

    return {
        "messages": [AIMessage(content=content)],
        "code": response["code"],
        "iteration_count": state["iteration_count"] + 1,
    }


class EvaluateCodeOutput(TypedDict):
    feedback: Annotated[str, ..., "Detailed feedback on the code"]
    score: Annotated[
        int,
        ...,
        "Score between 0 and 100 where 100 is the highest score, and 0 is the lowest score",
    ]


EVALUATOR_PROMPT = """
Evaluate this following code implementation for:
1. code correctness
2. time complexity
3. style and best practices

You should be evaluating only and not attempting to solve the task.

Score should be between 0 and 100 where 100 is the highest score, and 0 is the lowest score.

Provide detailed feedback if there are areas that need improvement. You should specify what needs improvement and why.

Only output JSON.
{code}
"""


async def evaluate_code(state: AgentState):

    message = PromptTemplate.from_template(EVALUATOR_PROMPT)
    # Prepare the evaluation chain
    evaluation_chain = message | model.with_structured_output(
        EvaluateCodeOutput, method="json_schema", strict=True
    )

    # Evaluate the code
    response = await evaluation_chain.ainvoke({"code": state["code"]})

    return {
        "messages": [HumanMessage(content=response["feedback"])],
        "score": response["score"],
        "feedbacks": [response["feedback"]],
    }


# Routing function
def should_continue(state: AgentState):
    # Check iteration count
    if state["iteration_count"] >= 2:
        return END
    # Check evaluation result
    score = state.get("score", 0)  # Get evaluation score with default 0
    if isinstance(score, (int, float)) and score >= 80:
        return END

    return "generate"


# Workflow setup
# Initialize the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generate", generate_code)
workflow.add_node("evaluate", evaluate_code)

# Add edges
workflow.set_entry_point("generate")
workflow.add_edge("generate", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    should_continue,
)
agent = workflow.compile()


async def create_evaluator_optimizer_workflow(task: str):
    response = await agent.ainvoke(
        {"code": "", "iteration_count": 0, "task": task, "feedbacks": []}
    )
    return response["code"]


async def main():
    task = """
    Implement a Stack with:
    1. push(x)
    2. pop()
    3. getMin()
    All operations should be O(1).
    """
    response = await create_evaluator_optimizer_workflow(task)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
