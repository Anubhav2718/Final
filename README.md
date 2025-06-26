# KRCL RuleBot — Railway Rules Q&A API

A FastAPI-based backend that uses Retrieval-Augmented Generation (RAG) to answer queries from Indian Railways rulebooks like KRCL G&SR 2020 and other manuals. Built using LangChain, FAISS, Cohere embeddings, and Groq's LLaMA3 LLMs.

---

##  Features

-  FastAPI-powered REST backend
-  FAISS vector store for semantic document retrieval
-  Cohere embeddings via `embed-english-light-v3.0`
-  LLaMA3-8B via Groq for ultra-fast, smart responses
-  `.env`-based secure API key management
-  CORS enabled for easy frontend integration (Streamlit, FlutterFlow, etc.)

---

##  Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/Anubhav2718/Final.git
cd Final
