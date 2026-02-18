from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import RAGSystem

app = FastAPI()
rag = RAGSystem()

class QuestionRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"message": "Basic API is working", "status": "online"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    num_chunks = rag.process_pdf(temp_path)
    return {"status": "success", "chunks_created": num_chunks}

@app.post("/chat")
async def chat(request: QuestionRequest):
    answer = rag.ask(request.prompt)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

