import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma

load_dotenv()

PDF_PATH = "data/sample.pdf"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("Pages loaded:", len(documents))

# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# Store embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

print("\nPDF Chatbot Ready!")

while True:

    question = input("\nAsk a question: ")

    if question.lower() in ["exit", "quit"]:
        break

    # Retrieve relevant documents
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )

    prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using only
the information provided in the context.

If the answer is not present in the context,
say that the information is not available
in the document.

Context:
