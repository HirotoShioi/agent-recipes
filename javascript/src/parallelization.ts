import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import dotenv from "dotenv";
import path from "path";
import { traceable } from "langsmith/traceable";

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
const parser = new StringOutputParser();

async function summarize_chunk(document: Document) {
  const prompt = PromptTemplate.fromTemplate(
    "Write a concise summary of the following: {chunk}"
  );
  const chain = prompt.pipe(model).pipe(parser);
  return await chain.invoke({ chunk: document.pageContent });
}

async function aggregate_summaries(summaries: string[]) {
  const prompt = PromptTemplate.fromTemplate(
    `The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes.`
  );
  const chain = prompt.pipe(model).pipe(parser);
  return await chain.invoke({ docs: summaries.join("\n\n") });
}

async function parallelization(url: string) {
  async function run(input: { url: string }) {
    const loader = new CheerioWebBaseLoader(input.url);
    const docs = await loader.load();
    const text_splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 5000,
      chunkOverlap: 200,
    });
    const texts = await text_splitter.splitDocuments(docs);
    const summaries = await Promise.all(texts.map(summarize_chunk));
    return await aggregate_summaries(summaries);
  }
  return traceable(run, { name: "parallelization" })({ url });
}

async function main() {
  const url = "https://lilianweng.github.io/posts/2023-06-23-agent/";
  const summary = await parallelization(url);
  console.log(summary);
}

main();
