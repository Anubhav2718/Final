# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from typing import List

# Load environment variables
load_dotenv()

# Validate API keys
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

# Define embedding class
class HFHubEmbeddings:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hf_token: str = None,
    ):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {hf_token}"
        }

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self.api_url, headers=self.headers, json={"inputs": text}
        )
        if response.status_code != 200:
            raise Exception(
                f"HuggingFace API Error: {response.status_code} - {response.text}"
            )

        embedding = response.json()

        # Mean pool if shape is [1 x seq_len x hidden]
        if isinstance(embedding[0][0], list):
            pooled = [sum(x) / len(x) for x in zip(*embedding[0])]
            return pooled
        else:
            return embedding[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

# ✅ CORRECT: Pass token here
embeddings = HFHubEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    hf_token=hf_token
)
# Allow dangerous deserialization only if you trust the source!
db1 = FAISS.load_local("faiss_index_krcl1", embeddings, allow_dangerous_deserialization=True)
db2 = FAISS.load_local("faiss_manual_krcl1", embeddings, allow_dangerous_deserialization=True)

retriever1 = db1.as_retriever(search_kwargs={"k": 4})
retriever2 = db2.as_retriever(search_kwargs={"k": 4})

# Define Retriever Tools
def gsr_retriever_tool(query: str) -> str:
    results = retriever1.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in results]) if results else "No relevant rules found in G&SR."

def manual_retriever_tool(query: str) -> str:
    results = retriever2.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in results]) if results else "No relevant content found."

def combined_retriever_tool(query: str) -> str:
    results1 = retriever1.get_relevant_documents(query)
    results2 = retriever2.get_relevant_documents(query)
    combined = results1[:4] + results2[:4]
    return "\n\n".join([doc.page_content for doc in combined]) if combined else "No relevant content found."

# LLM (Groq's LLaMA3)
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-8b-8192"
)

# Agent Setup
tools = [
    Tool(
        name="General and Subsidiary Rules",
        func=gsr_retriever_tool,
        description="Use for queries related to General and Subsidiary Rules (G&SR)"
    ),
    Tool(
        name="Accidental Manual",
        func=manual_retriever_tool,
        description="Use for queries related to the Accidental Manual"
    ),
    Tool(
        name="Combined",
        func=combined_retriever_tool,
        description="Use when query may be relevant to both G&SR and Manual"
    ),
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# FastAPI App Setup
app = FastAPI()

# CORS (Allow everything for dev — restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Schema
class Query(BaseModel):
    input: str

# Health Check Endpoint
@app.get("/")
def read_root():
    return {"message": "KRCL RuleBot backend is live!"}

# Main Ask Endpoint
@app.post("/ask")
async def ask_query(query: Query):
    try:
        result = agent_executor.invoke({"input": query.input, "chat_history": []})
        return {
            "final_answer": result.get("output", ""),
            "action": result["intermediate_steps"][0][0].tool if result.get("intermediate_steps") else "",
            "observation": result["intermediate_steps"][0][1] if result.get("intermediate_steps") else ""
        }
    except Exception as e:
        return {"error": str(e)}
