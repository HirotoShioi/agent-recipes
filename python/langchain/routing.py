# https://www.agentrecipes.com/routing
# Conditional Router Agent Workflow
from typing import Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
import asyncio
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class Assistant:
    """An assistant is a model that can be used to solve a task."""

    id: str
    description: str
    system_prompt: str

    def __init__(self, id: str, description: str, system_prompt: str):
        self.id = id
        self.description = description
        self.system_prompt = system_prompt

    async def run(self, input_query: str) -> str:
        messages = [("system", self.system_prompt), ("user", input_query)]
        parser = StrOutputParser()
        chain = model | parser
        response = await chain.ainvoke(messages)
        return response


class RouterSchema(TypedDict):
    """A schema for the router workflow."""

    reason: Annotated[str, ..., "The reason for selecting the route"]
    route: Annotated[
        str, ..., "The route selected. Must be one of the ids in the list of routes."
    ]


class RouterWorkflow:
    """A workflow that routes a task to the assistant best suited for the task."""

    def __init__(self, assistants: List[Assistant]):
        self.assistants = assistants

    @traceable(name="routing")
    async def run(self, input_query: str) -> str:
        """Given a `input_query` and a dictionary of `routes` containing options and details for each.
        Selects the best route for the task and return the response from the model.
        """
        ROUTER_PROMPT = """Given a user prompt/query: {user_query}, select the best option out of the following routes:
        {routes}. Answer only in JSON format."""
        prompt = PromptTemplate.from_template(ROUTER_PROMPT)
        model_routes_str = "\n".join(
            [f"id: {v.id}, description: {v.description}" for v in self.assistants]
        )
        chain = prompt | model.with_structured_output(
            RouterSchema, method="json_schema", strict=True
        )
        response = await chain.ainvoke(
            {"user_query": input_query, "routes": model_routes_str}
        )
        profiles_dict = {profile.id: profile for profile in self.assistants}
        assistant = profiles_dict[response["route"]]
        if not assistant:
            raise ValueError(f"Assistant with id {response['route']} not found")
        return await assistant.run(input_query)


async def main():
    assistants = [
        Assistant(
            id="code_generation",
            description="Suited for code generation",
            system_prompt="You are a helpful assistant that generates code for a website",
        ),
        Assistant(
            id="trip_planner",
            description="Suited for trip planning",
            system_prompt="You are a helpful assistant that plans trips",
        ),
        Assistant(
            id="story_teller",
            description="Suited for story telling",
            system_prompt="You are a helpful assistant that tells stories",
        ),
    ]
    tasks = [
        "Write a Python function to check if a number is prime.",
        "Plan a 2-week trip to Europe.",
        "Write a story about a brave knight and a dragon.",
    ]
    router = RouterWorkflow(assistants)
    responses = await asyncio.gather(*[router.run(task) for task in tasks])
    return responses


if __name__ == "__main__":
    response = asyncio.run(main())
    print(response)
