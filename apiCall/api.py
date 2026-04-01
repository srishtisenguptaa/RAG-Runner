import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import RAGSystem
from model.modelLG import ArchitectRAG

app = FastAPI(title="The Architect RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

standard_rag = RAGSystem()
architect_rag = ArchitectRAG()

# ── Data Models ────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    prompt: str
    session_id: str = "default"   # ← frontend sends a per-tab UUID

class SourceDetail(BaseModel):
    page: Any
    snippet: str
    score: float

class HistoryMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: str
    sources_detail: List[SourceDetail]
    followups: List[str]
    mode: str
    logs: List[str]

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def home():
    return {"status": "online", "message": "RAG API is running."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = standard_rag.process_pdf(temp_path)
        architect_rag.process_pdf(temp_path)

        return {"status": "success", "filename": file.filename, "chunks_created": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: QuestionRequest,
    mode: str = Query("standard", enum=["standard", "architect"])
):
    session_id = request.session_id   # unique per browser tab / page load

    try:
        # ── Architect Mode ─────────────────────────────────────────────────────
        if mode == "architect":
            agent_app = architect_rag.build_graph()

            # Pass full chat history into the graph state
            chat_history = architect_rag.get_history_messages(session_id)

            result = agent_app.invoke({
                "question": request.prompt,
                "documents": [],
                "logs": [],
                "web_search": "No",
                "generation": "",
                "chat_history": chat_history,   # ← injected here
            })

            answer = result.get("generation", "No answer generated.")

            # Persist the new turn
            architect_rag.save_turn(session_id, request.prompt, answer)

            used_web = any(
                "tavily_search" in str(doc.metadata.get("source", ""))
                for doc in result.get("documents", [])
            )
            has_pdf = architect_rag.vector_db is not None
            if used_web and has_pdf:
                sources_label = "PDF + Web Search"
            elif used_web:
                sources_label = "Web Search Only"
            else:
                sources_label = "PDF Only"

            followups = standard_rag.generate_followups(request.prompt, answer)

            return ChatResponse(
                answer=answer,
                sources=sources_label,
                sources_detail=[],
                followups=followups,
                mode="Architect Agent",
                logs=result.get("logs", [])
            )

        # ── Standard Mode ──────────────────────────────────────────────────────
        if standard_rag.vector_db is None:
            return ChatResponse(
                answer="Please upload a PDF before using Standard mode.",
                sources="None",
                sources_detail=[],
                followups=[],
                mode="Standard RAG",
                logs=["Error: No PDF uploaded. Standard mode requires a document."]
            )

        result = standard_rag.ask_with_sources(request.prompt, session_id=session_id)
        answer = result["answer"]
        raw_sources = result["sources"]

        followups = standard_rag.generate_followups(request.prompt, answer)

        return ChatResponse(
            answer=answer,
            sources="PDF Only",
            sources_detail=[SourceDetail(**s) for s in raw_sources],
            followups=followups,
            mode="Standard RAG",
            logs=["Executed standard retrieval with source transparency."]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/history/{session_id}", response_model=List[HistoryMessage])
async def get_history(session_id: str):
    """Returns the chat history for a session (useful for page reload recovery)."""
    return standard_rag.get_history_as_dicts(session_id)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """
    Clears chat memory for a session.
    Called automatically by the frontend on page load (simulates refresh = new chat).
    """
    standard_rag.clear_history(session_id)
    architect_rag.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
