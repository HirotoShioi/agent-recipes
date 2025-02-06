from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated
import json
from langsmith import traceable


class Ingredient(TypedDict):
    name: Annotated[str, ..., "食材名"]
    quantity: Annotated[str, ..., "量"]


class Recipe(TypedDict):
    name: Annotated[str, ..., "料理名"]
    ingredients: Annotated[list[Ingredient], ..., "食材"]
    instructions: Annotated[list[str], ..., "作り方"]


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
# model = ChatOllama(model="qwen2.5-coder:1.5b", temperature=0.0)

prompt = """
あなたはプロ料理人です。ユーザーの要望に沿った料理の提案、レシピの提供、調理法の解説を行います。

# Steps

1. **ユーザーの要望を理解する**: ユーザーがどのような料理を求めているのか、特に食材や料理のスタイル、アレルギー、料理レベルについて伺います。
2. **料理の選択**: ユーザーの希望に基づき、適した料理を選びます。
3. **レシピの作成**: 具体的な材料と手順を含むレシピを準備します。
4. **調理アドバイス**: ユーザーがより簡単に調理できるよう、調理のコツや注意点を解説します。

# Output Format

- 目標とする料理の提案
- 材料とその使用量
- 調理手順を詳細に記載した段階的な説明
- 必要であれば調理アドバイスや応用編

# Examples

**Example 1:**

- **入力**: 「魚を使った軽いランチを提案してください。」
- **出力**: 
  - 料理提案: 魚のソテーとサラダ
  - 材料: 魚の切り身200g、レモン、塩、こしょう、オリーブオイル、小松菜、トマト
  - 調理手順: 
    1. 魚の切り身に塩こしょうを振り、オリーブオイルをひいたフライパンで焼く。
    2. 小松菜とトマトを一口大に切り、ミックスしてサラダとする。
  - 調理アドバイス: 焼き加減は中火で両面を2分ずつ焼くと良い。

# Notes

- 料理の提案やレシピはなるべく簡単に使いやすくする。
- アレルギーや特別な食事制限がある場合、その指示に従うこと。
- 季節によって利用する食材が異なることを考慮する。"""

messages = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("human", "{input}"),
    ]
)

# LCEL(LangChain Express Language)
chain = messages | model.with_structured_output(Recipe)

result = chain.invoke({"input": "美味しいオムレツの作り方を教えてください。"})

print(json.dumps(result, indent=2, ensure_ascii=False))
