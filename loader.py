from langchain_community.document_loaders import WebBaseLoader
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

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
        ]) # List of urls 
nest_asyncio.apply()
loader.requests_per_second = 1
docs =  loader.aload() # REading web pages
# print(docs)

# print(f"Loaded {len(docs)} documents")
# for i, doc in enumerate(docs[:3]):
    # print(f"--- Document {i} ---")
    # print(doc.page_content[:300])  # sample

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",task_type='QUESTION_ANSWERING') # Embadding model 


embedding_dim = len(embeddings.embed_query("hello world")) # manually creating embedding dimensions
# test = (embeddings.embed_query("hello world"))
# print(f"Embedding dimension: {len(test)}")
# print(f"Vector norm: {sum(x**2 for x in test) ** 0.5:.4f}")
index = faiss.IndexFlatIP(embedding_dim) # Creating  FAISS vector DB index

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
) 

vector_store.add_documents(docs) # Adding docs in vector store 
vector_store.save_local("faiss_store") # Saving faiss index and file locally.


