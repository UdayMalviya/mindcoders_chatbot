
import os
import getpass
import nest_asyncio
from typing import List
import faiss
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set Google API Key
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass('Enter Your api key')

# Define loader
loader = WebBaseLoader([
    'https://learning.mindcoders.in/',
    'https://learning.mindcoders.in/about',
    'https://learning.mindcoders.in/course',
    'https://learning.mindcoders.in/course/front_end',
    'https://learning.mindcoders.in/course/mern',
    'https://learning.mindcoders.in/course/digital_marketing',
    'https://learning.mindcoders.in/course/data_analitics',
    'https://learning.mindcoders.in/course/back_end',
    'https://learning.mindcoders.in/course/react'
])

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type='QUESTION_ANSWERING')

# Async loader
def load_data(loader: WebBaseLoader):
    nest_asyncio.apply()
    loader.requests_per_second = 1
    docs = loader.aload()
    return docs

def embad_and_store(docs: List):
    sample_vec = embeddings.embed_query("Hello world")
    embedding_dim = len(sample_vec)

    index = faiss.IndexFlatL2(embedding_dim)  # Inner product
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(docs)
    
    vector_store.save_local("faiss_store")
    return vector_store

# Run everything
doc = (load_data(loader))
v = embad_and_store(doc)


