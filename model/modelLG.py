import os
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

# ── Graph State ────────────────────────────────────────────────────────────────
# Added `chat_history` so every node can see prior conversation turns.
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    logs: List[str]
    chat_history: List[BaseMessage]   # ← NEW: carries full session history


class ArchitectRAG:
    """
    LangGraph agent with 4 specialised sub-agent nodes:

    Node / Sub-Agent       Role
    ─────────────────────  ──────────────────────────────────────────────────
    retrieve               Similarity-search the FAISS vector store
    grade_documents        Relevance grader — filters noise, decides web fallback
    web_search             Tavily real-time search (triggered when grader says Yes)
    generate               Final answer composer (history-aware)

    Routing agent: decide_to_generate — conditional edge that reads `web_search`
                   flag and routes to either `web_search` node or `generate` node.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.web_search_tool = TavilySearch(max_results=3)
        self.vector_db = None

        # ── Session history store (same pattern as RAGSystem) ──
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
        """Persist a completed Q-A pair into the session store."""
        hist = self._get_session_history(session_id)
        hist.add_user_message(question)
        hist.add_ai_message(answer)

    # ── PDF Indexing ───────────────────────────────────────────────────────────

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    # ── Sub-Agent Node 1: Retrieve ─────────────────────────────────────────────

    def retrieve(self, state: GraphState):
        """Retrieves top-k similar chunks from the FAISS vector store."""
        question = state["question"]
        if self.vector_db is None:
            return {
                "documents": [],
                "question": question,
                "logs": ["[Retrieve Agent] No PDF indexed — will fall back to web search."]
            }
        documents = self.vector_db.similarity_search(question)
        return {
            "documents": documents,
            "question": question,
            "logs": ["[Retrieve Agent] Retrieved PDF context chunks."]
        }

    # ── Sub-Agent Node 2: Grade Documents ─────────────────────────────────────

    def grade_documents(self, state: GraphState):
        """
        Relevance Grader — scores each retrieved chunk against the question.
        Sets web_search='Yes' if no relevant chunks survive filtering.
        """
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"

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
            "documents": filtered_docs,
            "web_search": web_search,
            "logs": state.get("logs", []) + [
                f"[Grade Agent] {len(filtered_docs)} relevant chunk(s) kept. Web search needed: {web_search}"
            ]
        }

    # ── Sub-Agent Node 3: Web Search ───────────────────────────────────────────

    def web_search(self, state: GraphState):
        """Tavily real-time search fallback when PDF context is insufficient."""
        question = state["question"]
        documents = state.get("documents", [])

        search_results = self.web_search_tool.invoke({"query": question})

        if isinstance(search_results, str):
            web_content = search_results
        elif isinstance(search_results, list):
            parts = []
            for res in search_results:
                if isinstance(res, str):
                    parts.append(res)
                elif isinstance(res, dict):
                    parts.append(res.get("content", "") or res.get("text", ""))
            web_content = "\n".join(parts)
        else:
            web_content = str(search_results)

        web_doc = Document(page_content=web_content, metadata={"source": "tavily_search"})
        documents.append(web_doc)

        return {
            "documents": documents,
            "question": question,
            "logs": state.get("logs", []) + ["[Web Search Agent] Fetched real-time data from Tavily."]
        }

    # ── Sub-Agent Node 4: Generate ─────────────────────────────────────────────

    def generate(self, state: GraphState):
        """
        Final answer generator.
        History-aware: prepends prior conversation turns so the LLM can
        resolve follow-up questions (e.g. 'what did I just ask?', 'my name is X').
        """
        context = "\n\n".join([d.page_content for d in state["documents"]])
        sources = {d.metadata.get("source", "PDF") for d in state["documents"]}
        chat_history = state.get("chat_history", [])

        # History-aware generation prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful research assistant. "
             "Use the provided context and conversation history to answer accurately.\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),   # ← prior turns injected here
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

    # ── Routing Agent ──────────────────────────────────────────────────────────

    def decide_to_generate(self, state: GraphState):
        """Conditional edge: routes to web_search or generate based on grader output."""
        return "search_web" if state["web_search"] == "Yes" else "generate"

    # ── Graph Builder ──────────────────────────────────────────────────────────

    def build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("web_search", self.web_search)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {"search_web": "web_search", "generate": "generate"}
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()
