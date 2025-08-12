from langchain_community.vectorstores import FAISS
from langchain_google_genai import  ChatGoogleGenerativeAI

from chat import embeddings


# 2. Load FAISS vector store
vector_store = FAISS.load_local(
    "faiss_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.2

)

print(" MindCoders Chatbot — type 'exit' to quit")

# 4. Chat loop
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        print(" Goodbye!")
        break

    # Retrieve top documents
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build prompt
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