from langchain_community.vectorstores import FAISS
from langchain_google_genai import  ChatGoogleGenerativeAI

from chat import embeddings


# Load FAISS vector store
vector_store = FAISS.load_local(
    "faiss_store",
    embeddings,
    allow_dangerous_deserialization=True
)

#  Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.2

)

print(" MindCoders Chatbot — type 'exit' to quit")

# Chat loop
while True:
    query = input("\nYou: ") # User query
    if query.lower() in ["exit", "quit"]: # to exit the loop
        print(" Goodbye!")
        break

    # Retrieve top documents
    docs = vector_store.similarity_search(query, k=3) # change K if you want to increase or decrease the similarity search instance
    context = "\n\n".join([doc.page_content for doc in docs]) # final data going to pass on to generator (i.e.) model

    # System prompt can be modified according to need.
    prompt = f"""
    You are a helpful assistant for MindCoders training center.
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query} 
    """

    # Get answer from Gemini
    response = llm.invoke(prompt) 
    print("\n:", response.content)