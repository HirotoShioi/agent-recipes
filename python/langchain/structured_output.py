from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated
import json
from langsmith import traceable


class Ingredient(TypedDict):
    name: Annotated[str, ..., "材料の名前"]
    amount: Annotated[str, ..., "材料の量"]


class Recipe(TypedDict):
    reasoning: Annotated[str, ..., "どのような料理を作るかを考える"]
    name: Annotated[str, ..., "料理の名前"]
    ingredients: Annotated[list[Ingredient], ..., "材料"]
    instructions: Annotated[list[str], ..., "作り方"]


@traceable(name="recipe_search")
def main(query: str):
    model = ChatOpenAI(model="gpt-4o-mini")
    # model = ChatOllama(model="qwen2.5-coder:1.5b")
    messages = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたはプロ料理研究家です。"),
            ("user", "{query}"),
        ]
    )
    model_with_structured_output = model.with_structured_output(Recipe)

    parser = StrOutputParser()
    # prompt = messages.invoke({"topic": "AI"})
    # result = model.invoke(prompt)
    # result = parser.invoke(result)
    chain = messages | model_with_structured_output
    result = chain.invoke({"query": query})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main("子どもが好きな料理のレシピを教えてください。")
