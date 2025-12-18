# langchain-google-rag-hello-world

import os
import time
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

print("--- LangChain RAG with HuggingFace Embeddings & Gemini Chat ---")

# --- 0. Environment ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# if not GOOGLE_API_KEY:
#     raise ValueError("GEMINI_API_KEY not set")
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Pinecone environment variables not set")

# IMPORTANT: New index name because embeddings changed
INDEX_NAME = "langchain-hf-rag"
LLM_MODEL_NAME = "gemini-2.5-flash"

# --- 1. Load Documents ---
print("\n1. Loading documents...")
loader = TextLoader("data.txt", encoding="utf-8")
docs = loader.load()
print(f"Loaded {len(docs)} document(s)")

# --- 2. Split Documents ---
print("2. Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunks")

# --- 3. Initialize HuggingFace Embeddings (LOCAL) ---
print("3. Initializing HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 4. Pinecone Setup ---
print(f"4. Initializing Pinecone index '{INDEX_NAME}'...")
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = pinecone_client.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    print("Creating new Pinecone index...")
    vector_dimension = len(embeddings.embed_query("test"))
    print(f"Embedding dimension: {vector_dimension}")

    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=vector_dimension,
        metric="cosine",
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
    )

    while not pinecone_client.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    print("Index ready. Uploading embeddings...")
    vectorstore = PineconeVectorStore.from_documents(
        split_docs,
        embeddings,
        index_name=INDEX_NAME
    )
else:
    print("Connecting to existing index...")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

# --- 5. Gemini Chat Model ---
print("\n5. Initializing Gemini chat model...")
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0.2
)

# --- 6. Build RAG Chain ---
print("6. Building RAG chain...")

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. "
     "Answer only from the given context. "
     "If the answer is not in the context, say you do not know."),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 7. Query ---
print("\n7. Querying RAG system...")
query = "Who is Muhammad Younus?"
response = retrieval_chain.invoke({"input": query})

print("\n--- Answer ---")
print(response["answer"])

print("\n--- Sources ---")
for doc in response["context"]:
    print(doc.page_content[:120], "...\n")

print("\n--- RAG Complete ---")
