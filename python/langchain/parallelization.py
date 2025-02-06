# https://www.agentrecipes.com/parallelization
# Parallelization takes advantage of tasks that can broken up into discrete independent parts.
# The user's prompt is passed to multiple LLMs simultaneously. Once all the LLMs respond,
# their answers are all sent to a final LLM call to be aggregated for the final answer.

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv

load_dotenv()

# In this example, we are analyzing a lengthy document by dividing
# it into sections and assigning each section to a separate LLM for summarization,
# then combining the summaries into a comprehensive overview.

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


@traceable(name="summarize_chunk")
async def summarize_chunk(document: Document) -> str:
    prompt = PromptTemplate.from_template(
        "Write a concise summary of the following: {chunk}"
    )
    chain = prompt | model | parser
    return await chain.ainvoke({"chunk": document.page_content})


@traceable(name="aggregate_summaries")
async def aggregate_summaries(summaries: List[str]) -> str:
    prompt = PromptTemplate.from_template(
        """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""
    )
    chain = prompt | model | parser
    return await chain.ainvoke({"docs": "\n\n".join(summaries)})


@traceable(name="parallelization")
async def parallelization(url: str) -> str:
    """Create a workflow that uses multiple LLMs to analyze a URL and return a summary of the content."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    # Gather and await all summary tasks
    summaries = await asyncio.gather(*[summarize_chunk(text) for text in texts])
    return await aggregate_summaries(summaries)


async def main():
    url = "https://www.anthropic.com/research/building-effective-agents"
    summary = await parallelization(url)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
