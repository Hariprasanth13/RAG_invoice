import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    convert_system_message_to_human=True
)

# Load and split the PDF document
pdf_loader = PyPDFLoader(r"C:\Projects\Langchain\Rag_doc\Applied Natural Language Processing.pdf")
pages = pdf_loader.load_and_split()

# Set text splitting parameters
chunk_size = 500
chunk_overlap = 100  # About 20% of chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Prepare the context from the loaded pages
context = "\n\n".join(str(p.page_content) for p in pages)
context = context.replace("\n", "")
texts = text_splitter.split_text(context)

# Generate embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
text_embeddings = embeddings.embed_documents(texts)

# Create a FAISS HNSW index
dimension = len(text_embeddings[0])  # Dimensionality of the embeddings
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors in HNSW

# Convert embeddings to NumPy array and add to the index
embedding_matrix = np.array(text_embeddings).astype('float32')
index.add(embedding_matrix)

# Create index_to_docstore_id mapping and an InMemoryDocstore
index_to_docstore_id = {i: str(i) for i in range(len(texts))}  # Map FAISS indices to document IDs
documents = [Document(page_content=text) for text in texts]  # Create a list of Document objects
docstore = InMemoryDocstore(dict(zip(index_to_docstore_id.values(), documents)))  # Create a SimpleDocstore

# Create the FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings.embed_query,
    index=index,
    index_to_docstore_id=index_to_docstore_id,
    docstore=docstore
)

# Create retriever with search kwargs (k nearest neighbors = 3)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Set up your QA chain
template = """You are an expert in document extraction. Answer the questions with relevant information by referring to the context provided
{context}
Question: {question}"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Example question
question = "What is neural networks?"
result = qa_chain({"query": question})

# Output the result
print(result["result"])
