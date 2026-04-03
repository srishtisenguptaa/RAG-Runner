import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import RAGSystem
from model.modelLG import ArchitectRAG

app = FastAPI(title="The Architect RAG API — Multi-File (PDF + Excel)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

standard_rag  = RAGSystem()
architect_rag = ArchitectRAG()

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".xlsm", ".csv"}

# ── Models ─────────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    prompt: str
    session_id: str = "default"

class SourceDetail(BaseModel):
    page: Any
    snippet: str
    score: float
    source_file: str = "unknown"
    file_type: str   = "pdf"
    sheets: str      = ""

class HistoryMessage(BaseModel):
    role: str
    content: str

class IndexedFile(BaseModel):
    filename: str
    type: str       # "pdf" or "excel"
    chunks: int
    sheets: List[str] = []

class ChatResponse(BaseModel):
    answer: str
    sources: str
    sources_detail: List[SourceDetail]
    followups: List[str]
    mode: str
    logs: List[str]

class UploadResponse(BaseModel):
    status: str
    filename: str
    file_type: str
    chunks_created: int
    indexed_files: List[IndexedFile]

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def home():
    return {"status": "online", "message": "Multi-file RAG API (PDF + Excel) is running."}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    original_name = file.filename
    ext           = os.path.splitext(original_name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: PDF, XLSX, XLS, XLSM, CSV"
        )

    temp_path = f"temp_{original_name}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = standard_rag.process_file(temp_path, original_name)
        architect_rag.process_file(temp_path, original_name)

        file_type = "excel" if ext != ".pdf" else "pdf"
        return UploadResponse(
            status="success",
            filename=original_name,
            file_type=file_type,
            chunks_created=chunks,
            indexed_files=[IndexedFile(**f) for f in standard_rag.get_indexed_files()]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/files/{filename}")
async def remove_file(filename: str):
    standard_rag.remove_file(filename)
    architect_rag.remove_file(filename)
    return {"status": "removed", "filename": filename,
            "indexed_files": standard_rag.get_indexed_files()}


@app.get("/files", response_model=List[IndexedFile])
async def list_files():
    return [IndexedFile(**f) for f in standard_rag.get_indexed_files()]


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: QuestionRequest,
    mode: str = Query("standard", enum=["standard", "architect"])
):
    session_id = request.session_id
    try:
        # ── Architect Mode ─────────────────────────────────────────────────────
        if mode == "architect":
            agent_app    = architect_rag.build_graph()
            chat_history = architect_rag.get_history_messages(session_id)
            result = agent_app.invoke({
                "question": request.prompt,
                "documents": [], "logs": [],
                "web_search": "No", "generation": "",
                "chat_history": chat_history,
            })
            answer = result.get("generation", "No answer generated.")
            architect_rag.save_turn(session_id, request.prompt, answer)

            used_web  = any("tavily_search" in str(d.metadata.get("source", ""))
                            for d in result.get("documents", []))
            has_local = architect_rag.vector_db is not None
            sources_label = ("Files + Web Search" if used_web and has_local
                             else "Web Search Only" if used_web else "Indexed Files")

            return ChatResponse(
                answer=answer, sources=sources_label, sources_detail=[],
                followups=standard_rag.generate_followups(request.prompt, answer),
                mode="Architect Agent", logs=result.get("logs", [])
            )

        # ── Standard Mode ──────────────────────────────────────────────────────
        if standard_rag.vector_db is None:
            return ChatResponse(
                answer="Please upload at least one PDF or Excel file before using Standard mode.",
                sources="None", sources_detail=[], followups=[],
                mode="Standard RAG", logs=["Error: No files indexed."]
            )

        result  = standard_rag.ask_with_sources(request.prompt, session_id=session_id)
        answer  = result["answer"]
        followups = standard_rag.generate_followups(request.prompt, answer)

        return ChatResponse(
            answer=answer, sources="Indexed Files",
            sources_detail=[SourceDetail(**s) for s in result["sources"]],
            followups=followups, mode="Standard RAG",
            logs=["Executed standard retrieval with source transparency."]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/history/{session_id}", response_model=List[HistoryMessage])
async def get_history(session_id: str):
    return standard_rag.get_history_as_dicts(session_id)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    standard_rag.clear_history(session_id)
    architect_rag.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
