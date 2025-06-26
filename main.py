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

# Load environment variables
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
db1.merge_from(db2)

# Create retriever
retriever = db1.as_retriever(search_kwargs={"k": 6})  # Increased k for better context

# Define improved prompt
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a domain expert assistant specializing in the Konkan Railway's General & Subsidiary Rules (G&SR) and Accident Manual.

Your goal is to provide **structured, and accurate** answers based solely on the retrieved documents. **Do not fabricate or assume rules** under any circumstance.

### Guidelines:
- Always reference specific rules or sections (e.g., **S.R.2.4.1**, **G.R.1.2**, **2.31**).
- Structure your response using **clear headings, bullet points, or numbered steps**.
- Clarify any complex railway terms or procedures where necessary.
- If the answer cannot be found in the provided documents, clearly state:  
  **"I don’t know based on the available documents."**

---

### Question:
{input}

---

### Retrieved Documents:
{context}

---

### Your Answer:
""")


# LLM setup
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Create the retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input model
class Query(BaseModel):
    input: str

@app.get("/")
def root():
    return {"message": "KRCL RuleBot backend is live!"}

@app.post("/ask")
async def ask(query: Query):
    try:
        result = retrieval_chain.invoke({"input": query.input})
        return {
            "answer": result.get("answer", "I don’t know based on the available documents.")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}