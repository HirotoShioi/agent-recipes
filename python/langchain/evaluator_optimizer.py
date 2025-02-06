# https://www.agentrecipes.com/evaluator-optimizer
# The evaluator-optimizer workflow ensures task requirements
# are fully met through iterative refinement. An LLM performs
# a task, followed by a second LLM evaluating whether the result
# satisfies all specified criteria. If not, the process repeats
# with adjustments, continuing until the evaluator confirms all requirements are met.

import asyncio
from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)


# Define response types for better type safety
class GeneratorResponse(TypedDict):
    thoughts: Annotated[
        str,
        ...,
        "Your understanding of the task and feedback and how you plan to improve",
    ]
    code: Annotated[str, ..., "Your code implementation here"]


class EvaluatorResponse(TypedDict):
    feedback: Annotated[str, ..., "Detailed feedback on the code implementation"]
    evaluation: Annotated[
        Literal["PASS", "NEEDS_IMPROVEMENT", "FAIL"],
        ...,
        "Whether the code implementation passes all criteria",
    ]


# Generator function to create code based on task and feedback
async def generate_code(
    task: str, feedback: Optional[str] = None, code: Optional[str] = None
) -> GeneratorResponse:
    GENERATOR_PROMPT = """
Your objective is to execute the task delineated by <user input>. Should there be any critiques from your prior iterations,
Generate high-quality code that includes the following components:
- Proper error handling: Ensure the code effectively catches and handles potential errors and exceptions.
- Comprehensive documentation: Include detailed comments explaining the purpose and functionality of the code.
- Adherence to best practices: Follow industry standards for code quality and maintainability.

# Steps
- Identify potential points of failure in the code and implement error handling.
- Write clear and informative comments that explain both what each function does and why certain coding decisions were made.
- Follow best practices such as using meaningful variable names, writing modular code, and ensuring code readability.

# Output Format
Produce the code with comments embedded, error handling code blocks, and indications of adherence to best practices.

Task:
{task}
    """
    messages = [
        ("system", GENERATOR_PROMPT.format(task=task)),
    ]

    if code:
        messages.append(("user", f"Code: {code}\n\nFeedback: {feedback}"))

    chain = ChatPromptTemplate.from_messages(messages) | model.with_structured_output(
        GeneratorResponse, method="json_schema", strict=True
    )
    response = await chain.ainvoke({})
    return GeneratorResponse(thoughts=response["thoughts"], code=response["code"])


# Evaluator function to assess code quality
@traceable(name="evaluate_task")
async def evaluate_code(task: str, code: str) -> EvaluatorResponse:
    EVALUATOR_PROMPT = """
    Evaluate this following code implementation for:
    1. code correctness
    2. time complexity
    3. style and best practices

    You should be evaluating only and not attempting to solve the task.

    Only output "PASS" if all criteria are met and you have no further suggestions for improvements.

    Provide detailed feedback if there are areas that need improvement. You should specify what needs improvement and why.

    Only output JSON.
    """

    messages = [
        ("system", EVALUATOR_PROMPT),
        ("user", f"Task: {task}\n\nCode: {code}"),
    ]

    chain = model.with_structured_output(
        EvaluatorResponse, method="json_schema", strict=True
    )
    response = await chain.ainvoke(messages)
    return EvaluatorResponse(
        feedback=response["feedback"], evaluation=response["evaluation"]
    )


# Main workflow function using a while loop instead of recursion
@traceable(name="optimize_code")
async def optimize_code(task: str) -> str:
    MAX_ITERATIONS = 3
    current_iteration = 0
    current_code = ""
    current_feedback = None

    while current_iteration < MAX_ITERATIONS:
        # Generate code based on current feedback
        generated_result = await generate_code(task, current_feedback, current_code)
        current_code = generated_result["code"]

        # Evaluate the generated code
        evaluation = await evaluate_code(task, current_code)

        # Check if code passes all criteria
        if evaluation["evaluation"] == "PASS":
            return current_code

        # Update feedback and continue loop
        current_feedback = evaluation["feedback"]
        current_iteration += 1

        # Log progress
        print(
            f"Iteration {current_iteration}/{MAX_ITERATIONS}: {evaluation['evaluation']}"
        )
        print("Feedback:", current_feedback)

    print("Reached maximum iterations. Returning last generated code.")
    return current_code


async def main():
    task = """
    Implement a Stack with:
    1. push(x)
    2. pop()
    3. getMin()
    All operations should be O(1).
    """
    result = await optimize_code(task)
    print("Final result:", result)


if __name__ == "__main__":
    asyncio.run(main())
