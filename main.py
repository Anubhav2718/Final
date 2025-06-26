from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Validate API keys
groq_api_key = os.getenv("GROQ_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
if not groq_api_key or not cohere_api_key:
    raise ValueError("Missing GROQ_API_KEY or COHERE_API_KEY in environment.")

# Initialize embeddings
embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=cohere_api_key,
    user_agent="krcl-rulebot"
)

# Load FAISS vectorstores
db1 = FAISS.load_local("faiss_krcl_cohere", embeddings, allow_dangerous_deserialization=True)
db2 = FAISS.load_local("faiss_manual_cohere", embeddings, allow_dangerous_deserialization=True)

# Merge both FAISS indices
db1.merge_from(db2)

# Create retriever
retriever = db1.as_retriever(search_kwargs={"k": 4})

# Define prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for Indian Railways rule-based queries.
Answer the following question using the provided documents.
If the answer is not found in the documents, reply "I don’t know".

Question: {input}
Documents:
{context}
""")

# Load LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Chain setup
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# FastAPI App
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request model
class Query(BaseModel):
    input: str

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "KRCL RuleBot backend is live!"}

# Ask endpoint
@app.post("/ask")
async def ask(query: Query):
    try:
        result = retrieval_chain.invoke({"input": query.input})
        return {"answer": result.get("answer", "No answer generated.")}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
