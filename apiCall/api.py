from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
import shutil
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import RAGSystem         # Your original file
from model.modelLG import ArchitectRAG    # Your new LangGraph file

app = FastAPI()

# Initialize both systems
standard_rag = RAGSystem()
architect_rag = ArchitectRAG()

class QuestionRequest(BaseModel):
    prompt: str

@app.get("/")
async def home():
    return {"message": "RAG System API is running. Visit /docs for documentation."}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process for both systems so they share the same knowledge
    chunks = standard_rag.process_pdf(temp_path)
    architect_rag.process_pdf(temp_path) 
    
    os.remove(temp_path) # Clean up
    return {"status": "success", "chunks_created": chunks}

@app.post("/chat")
async def chat(request: QuestionRequest, mode: str = Query("standard", enum=["standard", "architect"])):
    """
    Toggle between 'standard' (Linear RAG) and 'architect' (Agentic LangGraph).
    """
    if mode == "architect":
        # Build and invoke the graph
        agent_app = architect_rag.build_graph()
        result = agent_app.invoke({"question": request.prompt})
        return {
            "answer": result["generation"],
            "sources": "PDF + Web Search" if result.get("web_search") == "Yes" else "PDF Only",
            "mode": "Architect Agent"
        }
    
    # Default to your original logic
    answer = standard_rag.ask(request.prompt)
    return {"answer": answer, "sources": "PDF Only", "mode": "Standard RAG"}

if __name__ == "__main__":
    import uvicorn
    # Set your Tavily API Key in environment if not already done
    # os.environ["TAVILY_API_KEY"] = "your_key_here"
    uvicorn.run(app, host="0.0.0.0", port=8000)