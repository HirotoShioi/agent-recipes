from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader

# 1. ドキュメントの読み込み
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://www.anthropic.com/research/building-effective-agents",
]
loader = WebBaseLoader(urls)
docs = loader.load()

# テキストの分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
doc_splits = text_splitter.split_documents(docs)

# 埋め込みモデルの初期化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chromaベクターストアの作成
# FAISS
# PINECONE
vectorstore = Chroma(
    collection_name="ai_blog_posts",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

# ドキュメントの保存
vectorstore.add_documents(doc_splits)
