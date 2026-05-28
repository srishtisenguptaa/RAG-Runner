import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# ── Excel Extraction Helper ───────────────────────────────────────────────────

def extract_excel_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    sections = []
    try:
        sheets = {"Sheet1": pd.read_csv(file_path)} if ext == ".csv" else pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in sheets.items():
            df = df.dropna(how="all").reset_index(drop=True)
            lines = []
            lines.append(f"=== Sheet: {sheet_name} ({df.shape[0]} rows x {df.shape[1]} columns) ===")
            lines.append(f"Columns: {', '.join(str(c) for c in df.columns)}")
            lines.append("")
            lines.append("--- Data ---")
            for _, row in df.head(2000).iterrows():
                row_str = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                lines.append(row_str)
            num_df = df.select_dtypes(include="number")
            if not num_df.empty:
                lines.append("")
                lines.append("--- Numeric Summary ---")
                lines.append(num_df.describe().round(4).to_string())
            sections.append("\n".join(lines))
    except Exception as e:
        sections.append(f"[Excel extraction failed: {e}]")
    return "\n\n".join(sections) if sections else "[No data extracted]"


# ── RAGSystem ──────────────────────────────────────────────────────────────────

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.vector_db = None
        self.rag_chain = None
        self.indexed_files: dict[str, dict] = {}
        self._history_store: dict[str, ChatMessageHistory] = {}

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._history_store:
            self._history_store[session_id] = ChatMessageHistory()
        return self._history_store[session_id]

    def clear_history(self, session_id: str):
        self._history_store.pop(session_id, None)

    def _rebuild_chain(self):
        system_prompt = (
            "You are a helpful research assistant. "
            "Use the provided context from indexed documents to answer the user's question. "
            "Documents may include PDFs and/or Excel spreadsheets (tabular data). "
            "When answering about spreadsheet data, reference column names, row values, "
            "sheet names, and numeric summaries where relevant. "
            "If the context doesn't contain the answer, say so honestly. "
            "You have access to conversation history to understand follow-up questions.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain   = create_stuff_documents_chain(self.llm, prompt)
        base_chain = create_retrieval_chain(self.vector_db.as_retriever(), qa_chain)
        self.rag_chain = RunnableWithMessageHistory(
            base_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def process_pdf(self, file_path: str, filename: str = None) -> int:
        fname  = filename or os.path.basename(file_path)
        loader = PyPDFLoader(file_path)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = fname
            doc.metadata["file_type"]   = "pdf"
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks   = splitter.split_documents(docs)
        self._upsert_chunks(chunks)
        self.indexed_files[fname] = {"type": "pdf", "chunks": len(chunks), "sheets": []}
        self._rebuild_chain()
        return len(chunks)

    def process_excel(self, file_path: str, filename: str = None) -> int:
        fname = filename or os.path.basename(file_path)
        ext   = os.path.splitext(file_path)[1].lower()
        try:
            sheet_names = ["Sheet1"] if ext == ".csv" else pd.ExcelFile(file_path).sheet_names
        except Exception:
            sheet_names = []
        raw_text = extract_excel_text(file_path)
        doc = Document(
            page_content=raw_text,
            metadata={
                "source": fname, "source_file": fname,
                "file_type": "excel", "sheets": ", ".join(sheet_names),
            }
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks   = splitter.split_documents([doc])
        self._upsert_chunks(chunks)
        self.indexed_files[fname] = {"type": "excel", "chunks": len(chunks), "sheets": sheet_names}
        self._rebuild_chain()
        return len(chunks)

    def process_file(self, file_path: str, filename: str = None) -> int:
        fname = (filename or os.path.basename(file_path)).lower()
        if fname.endswith(".pdf"):
            return self.process_pdf(file_path, filename or os.path.basename(file_path))
        elif fname.endswith((".xlsx", ".xls", ".xlsm", ".csv")):
            return self.process_excel(file_path, filename or os.path.basename(file_path))
        else:
            raise ValueError(f"Unsupported file type: {os.path.splitext(fname)[1]}")

    def _upsert_chunks(self, chunks):
        if self.vector_db is None:
            self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_db.add_documents(chunks)

    def remove_file(self, filename: str):
        if filename not in self.indexed_files:
            return
        del self.indexed_files[filename]
        if not self.indexed_files:
            self.vector_db = None; self.rag_chain = None; return
        remaining_docs = [
            doc for _, doc in self.vector_db.docstore._dict.items()
            if doc.metadata.get("source_file") != filename
        ]
        if not remaining_docs:
            self.vector_db = None; self.rag_chain = None; return
        self.vector_db = FAISS.from_documents(remaining_docs, self.embeddings)
        self._rebuild_chain()

    def ask_with_sources(self, question: str, session_id: str = "default") -> dict:
        if not self.rag_chain or not self.vector_db:
            return {"answer": "Please upload at least one file first.", "sources": []}
        result = self.rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        answer = result["answer"]
        docs_with_scores = self.vector_db.similarity_search_with_score(question, k=3)
        sources = []
        for doc, score in docs_with_scores:
            raw_page    = doc.metadata.get("page", None)
            page_num    = (int(raw_page) + 1) if raw_page is not None else "N/A"
            source_file = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            sources.append({
                "page":        page_num,
                "snippet":     doc.page_content[:220].strip(),
                "score":       round(float(score), 3),
                "source_file": source_file,
                "file_type":   doc.metadata.get("file_type", "pdf"),
                "sheets":      doc.metadata.get("sheets", ""),
            })
        return {"answer": answer, "sources": sources}

    def generate_followups(self, question: str, answer: str) -> list[str]:
        prompt = (
            f"Given this Q&A from indexed PDFs and/or Excel sheets, suggest exactly 3 short "
            f"follow-up questions. Return ONLY the questions, one per line, no numbering.\n\n"
            f"Question: {question}\nAnswer: {answer}"
        )
        try:
            raw = self.llm.invoke(prompt).content
            return [q.strip() for q in raw.strip().split("\n") if q.strip()][:3]
        except Exception:
            return []

    def get_history_as_dicts(self, session_id: str) -> list[dict]:
        history = self._get_session_history(session_id)
        result  = []
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result

    def get_indexed_files(self) -> list[dict]:
        return [{"filename": f, **m} for f, m in self.indexed_files.items()]
