import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import dotenv from "dotenv";
import { Document } from "langchain/document";
dotenv.config();

async function ingestDocuments() {
  // 3. Initialize embeddings model
  const embeddings = new OpenAIEmbeddings();

  // 1. Load documents
  const urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  ];
  const docSplits: Document<Record<string, any>>[] = [];
  for (const url of urls) {
    const loader = new CheerioWebBaseLoader(url);
    const docs = await loader.load();

    // 2. Split text into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 100,
    });
    const ds = await textSplitter.splitDocuments(docs);
    docSplits.push(...ds);
  }
  // 4. Create and persist Chroma vector store
  await Chroma.fromDocuments(docSplits, embeddings, {
    collectionName: "ai_blog_posts",
    url: "http://localhost:8000", // Chroma server URL
  });
}

async function main() {
  await ingestDocuments();
}

main();
