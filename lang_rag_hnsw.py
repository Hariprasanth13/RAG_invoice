from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

from langchain import hub
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import numpy as np
#from langchain.docstore import InMemoryDocstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document

def read_file():
    with open (r"C:\Projects\Langchain\Rag_doc\data.txt",'r') as f:
        content = f.read()

    return content

# split - chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
splits = text_splitter.split_text(read_file())

# Embed
model_name = "BAAI/bge-small-en"
model_kwargs = {"device":"cpu"}
encode_kwargs = {"normalize_embeddings" : True}
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,model_kwargs = model_kwargs,encode_kwargs =encode_kwargs
)

embeddings = hf_embeddings.embed_documents(splits)

#convert embedding in to numpy array
embeddings_np = np.array(embeddings).astype('float32')

#create HNSW index
nlist = 100 #No of clusters
m = 16 # Number of bi-directional links created for the graph
index = faiss.IndexHNSWFlat(embeddings_np.shape[1],m)

# Add embedding to the index
index.add(embeddings_np)

#Create docstore to hold the documents
documents = [Document(page_content = split) for split in splits]
docstore = InMemoryDocstore({i:doc for i,doc in enumerate(documents)})

#create mapping between FAISS indices and docstore IDs
index_to_docstore_id ={i:i for i in range(len(documents))}

# create the FAISS vector store
faiss_vectorstore = FAISS(
    index = index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=hf_embeddings.embed_query
    )



# Create a retriever from the FAISS vector store
retriever = faiss_vectorstore.as_retriever()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="You are an expert in extracting entities from invoices. Try extracting entities from the input text with the help of sample annotation given Input invoice text = Invoice Text: Invoice No. 30001 issued by Harish Ltd. on 2024-10-07. Amount Due: $25536. Part: LED Display, Received: 2024-10-03. Buyer: ID 103, Prasanth. Use the annotations only as samples to help in the process of extracting entities on given invoice. Sample annotations: {context}, answer the question: {question}"
)

llm = ChatGroq(model = "llama3-8b-8192",temperature =0) # 8B parameters and 8192 input tokens
#chain - Method 1
rag_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

query ="What is the name of the buyer in the given invoice?"
result = rag_chain.invoke(query)
print(result)
# print(len(result["source_documents"]))
# for i,doc in enumerate(result["source_documents"]):
#     print(f"Text {i}: {doc.page_content}")+2