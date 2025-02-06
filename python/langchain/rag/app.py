from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# 埋め込みモデルの初期化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chromaベクターストアの作成
vectorstore = Chroma(
    collection_name="ai_blog_posts",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

# レトリーバーの作成
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 最大限の多様性を持つ文書を検索
    search_kwargs={"k": 5},  # 5つの関連文書を取得
)

# result = retriever.invoke("AIエージェントとは何ですか？")
# print(result)
# LLMの初期化
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# プロンプトテンプレート
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "以下のコンテキストを使用して質問に答えてください。関連する情報のみを使用し、わからない場合は正直に「わかりません」と答えてください。",
        ),
        ("human", "質問: {question}\n\nコンテキスト: {context}"),
    ]
)

# RAGチェーンの構築
rag_chain = prompt | llm | StrOutputParser()


# 質問への回答
@traceable(name="ask_question")
def ask_question(question):
    documents = retriever.invoke(question)
    response = rag_chain.invoke(
        {
            "question": question,
            "context": [doc.page_content for doc in documents],
        }
    )
    return response


# 使用例
print(ask_question("AIエージェントとは何ですか？"))
print(ask_question("プロンプトエンジニアリングの基本的な戦略は？"))
