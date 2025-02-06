import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import dotenv from "dotenv";

dotenv.config();

// 埋め込みモデルの初期化
const embeddings = new OpenAIEmbeddings();

// Chromaベクターストアの作成
const vectorstore = new Chroma(embeddings, {
  collectionName: "ai_blog_posts",
  url: "http://localhost:8000",
});

// レトリーバーの作成
const retriever = vectorstore.asRetriever({
  searchKwargs: { fetchK: 5 }, // 5つの関連文書を取得
});

// LLMの初期化
const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.7 });

// プロンプトテンプレート
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "以下のコンテキストを使用して質問に答えてください。関連する情報のみを使用し、わからない場合は正直に「わかりません」と答えてください。",
  ],
  ["human", "質問: {question}\n\nコンテキスト: {context}"],
]);

// RAGチェーンの構築
const ragChain = prompt.pipe(llm).pipe(new StringOutputParser());

// 質問への回答
async function askQuestion(question: string) {
  const documents = await retriever.invoke(question);
  const response = await ragChain.invoke({
    question,
    context: documents.map((doc) => doc.pageContent).join("\n"),
  });
  return response;
};

async function main() {
  // 使用例
  console.log(await askQuestion("AIエージェントとは何ですか？"));
  console.log(await askQuestion("プロンプトエンジニアリングの基本的な戦略は？"));
}

main();
