from fastapi import FastAPI
from pydantic import BaseModel
from chat import init_llm, generate_answer, load_vector_store
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = ['*'] # list of allowed origins
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_methods = ['*'],
                   allow_headers= ['*'],) # adding middleware

# Load once at startup
vector_store = load_vector_store()
llm = init_llm()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = generate_answer(request.question, vector_store, llm)
    return {"question": request.question,"answer": answer}

@app.get("/ask")
def ask_question_get(question: str):
    answer = generate_answer(question, vector_store, llm)
    return {"answer": answer}
