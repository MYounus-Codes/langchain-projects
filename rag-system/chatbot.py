import os
import time
from pinecone import Pinecone, PodSpec

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

print("--- LangChain RAG with Google Embeddings & Pinecone ---")

# --- 0. Configuration and Environment Setup ---
# Ensure your environment variables are set before running this script
# export GEMINI_API_KEY='your_GEMINI_API_KEY'
# export PINECONE_API_KEY='your_pinecone_api_key'
# export PINECONE_ENVIRONMENT='your_pinecone_environment' # e.g., gcp-starter, us-east-1

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT environment variables not set.")

# Pinecone index details
INDEX_NAME = "langchain-google-rag-hello-world"
EMBEDDING_MODEL_NAME = "models/embedding-001" # Google's text embedding model
LLM_MODEL_NAME = "gemini-2.5-flash" # Google's generative model

# --- 1. Load Documents ---
print("\n1. Loading documents...")
loader = TextLoader("data.txt")
docs = loader.load()
print(f"Loaded {len(docs)} document(s).")

# --- 2. Split Documents ---
print("2. Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunks.")

# --- 3. Initialize Google Embeddings ---
print(f"3. Initializing Google Embeddings with model: {EMBEDDING_MODEL_NAME}...")
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# --- 4. Initialize Pinecone and Create/Connect to Index ---
print(f"4. Initializing Pinecone client and checking for index '{INDEX_NAME}'...")
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Get current indexes
existing_indexes = pinecone_client.list_indexes().names()
print(f"Existing Pinecone indexes: {existing_indexes}")

if INDEX_NAME not in existing_indexes:
    print(f"Creating new Pinecone index: '{INDEX_NAME}'...")
    # Determine the dimension from an example embedding.
    # It's good practice to get this dynamically if possible,
    # or know the dimension of your chosen embedding model.
    # Google's `embedding-001` has a dimension of 768.
    example_embedding = embeddings.embed_query("test query")
    vector_dimension = len(example_embedding)
    print(f"Detected embedding dimension: {vector_dimension}")

    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=vector_dimension, # Dimension of Google's embedding-001 model
        metric="cosine", # Cosine similarity is common for text embeddings
        spec=PodSpec(environment=PINECONE_ENVIRONMENT) # Specify environment for starter plan
    )
    print(f"Index '{INDEX_NAME}' created. Waiting for it to be ready...")
    # Wait for the index to be ready
    while not pinecone_client.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
    print(f"Index '{INDEX_NAME}' is ready.")
    
    # --- 5. Store Embeddings in Pinecone (Populate the Vector Store) ---
    print(f"5. Populating Pinecone index '{INDEX_NAME}' with document embeddings...")
    vectorstore = PineconeVectorStore.from_documents(
        split_docs,
        embeddings,
        index_name=INDEX_NAME
    )
    print("Documents successfully embedded and uploaded to Pinecone.")
else:
    # --- 5. Connect to Existing Pinecone Index ---
    print(f"5. Connecting to existing Pinecone index '{INDEX_NAME}'...")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    print("Connected to existing Pinecone index.")

# --- 6. Initialize Google LLM ---
print(f"\n6. Initializing Google LLM with model: {LLM_MODEL_NAME}...")
llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2) # temperature=0.2 for slightly creative but factual responses

# --- 7. Create the RAG Chain with LCEL ---
print("7. Building the RAG chain...")

# Define the prompt template for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant for answering questions about AI and Machine Learning. "
               "Use the following retrieved context to answer the question concisely and accurately. "
               "If you don't know the answer based on the provided context, just say that you don't know."),
    ("human", "Context: {context}\nQuestion: {input}") # 'input' is the user's question, 'context' is from the retriever
])

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# This chain takes the documents retrieved and stuffs them into the prompt
document_chain = create_stuff_documents_chain(llm, prompt)

# This is the main retrieval chain which first retrieves documents, then passes them to the document_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("RAG chain created.")

# --- 8. Query the RAG System ---
print("\n8. Querying the RAG system...")
query = "What is the difference between machine learning and deep learning?"
print(f"User Query: '{query}'")

response = retrieval_chain.invoke({"input": query})

print("\n--- RAG Response ---")
# The response object contains:
# - 'input': The original user query
# - 'context': The list of Document objects retrieved by the retriever
# - 'answer': The LLM's generated answer
print("Answer:", response["answer"])
print("\n--- Retrieved Context (Sources) ---")
for doc in response["context"]:
    print(f"- Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
    print(f"  Snippet: {doc.page_content[:150]}...\n")


# --- 9. Optional: Clean up Pinecone Index ---
# Uncomment the following lines if you want to delete the index after running the script.
# This is useful for development to avoid accumulating indexes.
# Be careful with this in production environments!
# print(f"\n9. Deleting Pinecone index '{INDEX_NAME}'...")
# try:
#     pinecone_client.delete_index(INDEX_NAME)
#     print(f"Index '{INDEX_NAME}' deleted successfully.")
# except Exception as e:
#     print(f"Error deleting index: {e}")

print("\n--- RAG Hello World Complete ---")