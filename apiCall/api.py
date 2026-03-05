import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Ensure the root project directory is in the path so we can find model.model and model.modelLG
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import RAGSystem         # Your original Linear RAG file
from model.modelLG import ArchitectRAG    # Your new LangGraph Agent file

app = FastAPI(title="The Architect RAG API")

# Initialize both systems
# Note: Ensure these classes handle their own LLM/Embedding initialization
standard_rag = RAGSystem()
architect_rag = ArchitectRAG()

# --- Data Models ---
class QuestionRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str
    sources: str
    mode: str
    logs: List[str]

# --- Endpoints ---

@app.get("/")
async def home():
    return {
        "status": "online",
        "message": "RAG System API is running. Access the UI via Streamlit or /docs for API testing."
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Saves the PDF and indexes it into both the Standard and Architect vector databases.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process for both systems so they share the same knowledge base
        # Ensure your process_pdf methods return the number of chunks created
        chunks = standard_rag.process_pdf(temp_path)
        architect_rag.process_pdf(temp_path) 
        
        return {
            "status": "success", 
            "filename": file.filename, 
            "chunks_created": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path) # Clean up the physical file after indexing

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QuestionRequest, mode: str = Query("standard", enum=["standard", "architect"])):
    """
    Toggle between 'standard' (Linear RAG) and 'architect' (Agentic LangGraph with Web Search).
    """
    try:
        if mode == "architect":
            # 1. Build the graph
            agent_app = architect_rag.build_graph()
            
            # 2. Invoke the agentic workflow
            # Ensure architect_rag.vector_db is not None (user must upload PDF first)
            if architect_rag.vector_db is None:
                return {
                    "answer": "Please upload and index a PDF before using the Architect mode.",
                    "sources": "None",
                    "mode": "Architect Agent",
                    "logs": ["Error: No vector database found."]
                }

            result = agent_app.invoke({
                "question": request.prompt, 
                "documents": [], 
                "logs": []
            })
            
            # 3. Determine if Web Search was actually utilized for the source label
            used_web = any("tavily_search" in str(doc.metadata.get("source", "")) for doc in result.get("documents", []))
            
            return {
                "answer": result.get("generation", "No answer generated."),
                "sources": "PDF + Web Search" if used_web else "PDF Only",
                "mode": "Architect Agent",
                "logs": result.get("logs", [])
            }
        
        # --- Default to Standard Logic ---
        # Ensure standard_rag is ready
        answer = standard_rag.ask(request.prompt)
        return {
            "answer": answer, 
            "sources": "PDF Only", 
            "mode": "Standard RAG",
            "logs": ["Executed standard linear retrieval-generation."]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# --- Execution ---
if __name__ == "__main__":
    import uvicorn
    # 127.0.0.1 is safer for local browser access than 0.0.0.0
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)