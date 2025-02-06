# https://www.agentrecipes.com/prompt-chaining
# A workflow where the output of one LLM call becomes the input for the next.
# This sequential design allows for structured reasoning and step-by-step task completion.

import asyncio
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


@traceable(name="prompt_chaining")
async def prompt_chaining(input_query: str, prompts: List[str]) -> str:
    response_chain = []
    messages: List[BaseMessage] = [
        SystemMessage(
            content="You are a helpful assistant that can solve math problems."
        ),
        HumanMessage(content=input_query),
    ]
    response = input_query
    for i, prompt in enumerate(prompts):
        print(f"Step {i+1}")
        messages.append(HumanMessage(content=prompt))
        chain = model | parser
        response = await chain.ainvoke(messages)
        messages.append(AIMessage(content=response))
        response_chain.append(response)
        print(f"Response: {response}")
    return response_chain[-1]


if __name__ == "__main__":

    question = "Sally earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

    prompt_chain = [
        """Given the math problem, ONLY extract any relevant numerical information and how it can be used.""",
        """Given the numberical information extracted, ONLY express the steps you would take to solve the problem.""",
        """Given the steps, express the final answer to the problem.""",
    ]
    response = asyncio.run(prompt_chaining(question, prompt_chain))
    print(response)
