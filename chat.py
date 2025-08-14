from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
from markdown import markdown
from loader import embeddings



#  Load FAISS vector store
def load_vector_store(store_path="faiss_store"):
    return FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


#  Initialize Gemini chat model
def init_llm(model_name="gemini-2.5-pro", temperature=0.2):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature
    )

# convert markdown format to text
def markdown_to_text(text):
    html = markdown(text)
    # Parse and extract text
    soup = BeautifulSoup(html, 'html.parser')
    return soup.text

# Generate the answer for chat 
def generate_answer(query, vector_store, llm, k=3):
    docs = vector_store.similarity_search(query, k=k)
    print(f"[DEBUG] Retrieved {len(docs)} docs")
    for i, doc in enumerate(docs):
        print(f"[DEBUG] Doc {i} >> : {doc.page_content[:100]}")

    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are a helpful assistant for MindCoders training center.
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}
    """

    try:
        response = llm.invoke(prompt)
        return getattr(response, "content", "").strip() or "[No response from model]"
    except Exception as e:
        return f"[ERROR] {e}"


# 4. CLI loop (optional for testing before FastAPI)
# def chat_loop():
#     vector_store = load_vector_store()
#     llm = init_llm()

#     print("MindCoders Chatbot — type 'exit' to quit")

#     while True:
#         query = input("\nYou: ")
#         if query.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break

#         answer = generate_answer(query, vector_store, llm)
#         print("\n:", markdown_to_text(answer))

# if __name__ == "__main__":
    # chat_loop()