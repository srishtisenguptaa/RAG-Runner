import os
import pandas as pd
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# ── Excel Extraction Helper ───────────────────────────────────────────────────

def extract_excel_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    sections = []
    try:
        sheets = {"Sheet1": pd.read_csv(file_path)} if ext == ".csv" else pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in sheets.items():
            df = df.dropna(how="all").reset_index(drop=True)
            lines = [
                f"=== Sheet: {sheet_name} ({df.shape[0]} rows x {df.shape[1]} columns) ===",
                f"Columns: {', '.join(str(c) for c in df.columns)}", "",
                "--- Data ---",
            ]
            for _, row in df.head(2000).iterrows():
                lines.append(" | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val)))
            num_df = df.select_dtypes(include="number")
            if not num_df.empty:
                lines += ["", "--- Numeric Summary ---", num_df.describe().round(4).to_string()]
            sections.append("\n".join(lines))
    except Exception as e:
        sections.append(f"[Excel extraction failed: {e}]")
    return "\n\n".join(sections) if sections else "[No data extracted]"


# ── Graph State ────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    logs: List[str]
    chat_history: List[BaseMessage]


class ArchitectRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.web_search_tool = TavilySearch(max_results=3)
        self.vector_db = None
        self.indexed_files: dict[str, dict] = {}
        self._history_store: dict[str, ChatMessageHistory] = {}

    # ── History helpers ────────────────────────────────────────────────────────

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._history_store:
            self._history_store[session_id] = ChatMessageHistory()
        return self._history_store[session_id]

    def clear_history(self, session_id: str):
        self._history_store.pop(session_id, None)

    def get_history_messages(self, session_id: str) -> List[BaseMessage]:
        return self._get_session_history(session_id).messages

    def save_turn(self, session_id: str, question: str, answer: str):
        hist = self._get_session_history(session_id)
        hist.add_user_message(question)
        hist.add_ai_message(answer)

    # ── File Processing ────────────────────────────────────────────────────────

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
        return len(chunks)

    def process_file(self, file_path: str, filename: str = None) -> int:
        fname = (filename or os.path.basename(file_path)).lower()
        if fname.endswith(".pdf"):
            return self.process_pdf(file_path, filename or os.path.basename(file_path))
        elif fname.endswith((".xlsx", ".xls", ".xlsm", ".csv")):
            return self.process_excel(file_path, filename or os.path.basename(file_path))
        else:
            raise ValueError(f"Unsupported file type: {os.path.splitext(fname)[1]}")

    def _upsert_chunks(self, chunks: List[Document]):
        if self.vector_db is None:
            self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_db.add_documents(chunks)

    def remove_file(self, filename: str):
        if filename not in self.indexed_files:
            return
        del self.indexed_files[filename]
        if not self.indexed_files:
            self.vector_db = None; return
        remaining_docs = [
            doc for _, doc in self.vector_db.docstore._dict.items()
            if doc.metadata.get("source_file") != filename
        ]
        self.vector_db = FAISS.from_documents(remaining_docs, self.embeddings) if remaining_docs else None

    def get_indexed_files(self) -> list[dict]:
        return [{"filename": f, **m} for f, m in self.indexed_files.items()]

    # ── Graph Nodes ────────────────────────────────────────────────────────────

    def retrieve(self, state: GraphState):
        question = state["question"]
        if self.vector_db is None:
            return {
                "documents": [], "question": question,
                "logs": ["[Retrieve Agent] No files indexed — will fall back to web search."]
            }
        documents = self.vector_db.similarity_search(question)
        return {
            "documents": documents, "question": question,
            "logs": [f"[Retrieve Agent] Retrieved {len(documents)} chunk(s) from indexed files."]
        }

    def grade_documents(self, state: GraphState):
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search    = "No"

        grader_prompt = ChatPromptTemplate.from_template(
            "Respond ONLY with 'yes' or 'no'. Is this document relevant to '{question}'?\nDoc: {doc}"
        )
        grader_chain = grader_prompt | self.llm

        for d in documents:
            score = grader_chain.invoke({"question": question, "doc": d.page_content})
            if "yes" in score.content.lower():
                filtered_docs.append(d)

        if not filtered_docs:
            web_search = "Yes"

        return {
            "documents": filtered_docs, "web_search": web_search,
            "logs": state.get("logs", []) + [
                f"[Grade Agent] {len(filtered_docs)} relevant chunk(s) kept. Web search: {web_search}"
            ]
        }

    def web_search(self, state: GraphState):
        question  = state["question"]
        documents = state.get("documents", [])
        results   = self.web_search_tool.invoke({"query": question})

        if isinstance(results, str):
            web_content = results
        elif isinstance(results, list):
            parts = []
            for r in results:
                parts.append(r if isinstance(r, str) else r.get("content", "") or r.get("text", ""))
            web_content = "\n".join(parts)
        else:
            web_content = str(results)

        documents.append(Document(page_content=web_content, metadata={"source": "tavily_search"}))
        return {
            "documents": documents, "question": question,
            "logs": state.get("logs", []) + ["[Web Search Agent] Fetched real-time data from Tavily."]
        }

    def generate(self, state: GraphState):
        context     = "\n\n".join([d.page_content for d in state["documents"]])
        sources     = {d.metadata.get("source_file", d.metadata.get("source", "unknown")) for d in state["documents"]}
        chat_history = state.get("chat_history", [])

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful research assistant. "
             "Use the provided context and conversation history to answer accurately. "
             "Context may include PDFs and/or Excel spreadsheet data.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        response = (prompt | self.llm).invoke({
            "context": context,
            "question": state["question"],
            "chat_history": chat_history,
        })

        final_answer = f"{response.content}\n\n**Sources:** {', '.join(sources)}"
        return {
            "generation": final_answer,
            "logs": state.get("logs", []) + ["[Generate Agent] Final answer composed."]
        }

    def decide_to_generate(self, state: GraphState):
        return "search_web" if state["web_search"] == "Yes" else "generate"

    def build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve",        self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate",        self.generate)
        workflow.add_node("web_search",      self.web_search)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents", self.decide_to_generate,
            {"search_web": "web_search", "generate": "generate"}
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()
