import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

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

        # ── Chat History Store (session_id → ChatMessageHistory) ──
        # Lives in memory; cleared on server restart (page refresh triggers new session)
        self._history_store: dict[str, ChatMessageHistory] = {}

    # ── Internal: get or create history for a session ──
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._history_store:
            self._history_store[session_id] = ChatMessageHistory()
        return self._history_store[session_id]

    def clear_history(self, session_id: str):
        """Wipe history for a session (called on frontend refresh)."""
        self._history_store.pop(session_id, None)

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)

        # ── Chain 1: Stuff Documents Chain ──
        # Combines retrieved docs into context, aware of chat history
        system_prompt = (
            "You are a helpful research assistant. "
            "Use the provided context from the document to answer the user's question. "
            "If the context doesn't contain the answer, say so honestly. "
            "You also have access to the conversation history to understand follow-up questions.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),   # ← injects prior turns
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(self.llm, prompt)

        # ── Chain 2: Retrieval Chain ──
        # Wraps retriever + qa_chain into a full RAG pipeline
        base_chain = create_retrieval_chain(
            self.vector_db.as_retriever(), qa_chain
        )

        # ── Chain 3: History-Aware Wrapper ──
        # Automatically injects + saves chat_history per session_id
        self.rag_chain = RunnableWithMessageHistory(
            base_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return len(chunks)

    def ask(self, question: str, session_id: str = "default"):
        if not self.rag_chain:
            return "Please upload a PDF first."
        result = self.rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        return result["answer"]

    def ask_with_sources(self, question: str, session_id: str = "default"):
        """Returns answer + top-3 source chunks, honouring chat history."""
        if not self.rag_chain or not self.vector_db:
            return {"answer": "Please upload a PDF first.", "sources": []}

        result = self.rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        answer = result["answer"]

        # Retrieve top-3 chunks with similarity scores
        docs_with_scores = self.vector_db.similarity_search_with_score(question, k=3)
        sources = []
        for doc, score in docs_with_scores:
            raw_page = doc.metadata.get("page", None)
            page_num = (int(raw_page) + 1) if raw_page is not None else "?"
            sources.append({
                "page": page_num,
                "snippet": doc.page_content[:220].strip(),
                "score": round(float(score), 3)
            })

        return {"answer": answer, "sources": sources}

    def generate_followups(self, question: str, answer: str):
        """Ask the LLM to suggest 3 relevant follow-up questions."""
        prompt = (
            f"Given this question and answer from a document, suggest exactly 3 short, "
            f"insightful follow-up questions a reader might ask next. "
            f"Return ONLY the questions, one per line, no numbering, no bullets.\n\n"
            f"Question: {question}\nAnswer: {answer}"
        )
        try:
            raw = self.llm.invoke(prompt).content
            followups = [q.strip() for q in raw.strip().split("\n") if q.strip()][:3]
            return followups
        except Exception:
            return []

    def get_history_as_dicts(self, session_id: str) -> list[dict]:
        """Returns chat history as a list of {role, content} dicts for the frontend."""
        history = self._get_session_history(session_id)
        result = []
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result
