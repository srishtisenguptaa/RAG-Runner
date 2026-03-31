import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        self.vector_db = FAISS.from_documents(chunks, self.embeddings)

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based ONLY on context:\n{context}\nQuestion: {input}"
        )
        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.vector_db.as_retriever(), qa_chain)
        return len(chunks)

    def ask(self, question):
        if not self.rag_chain:
            return "Please upload a book first."
        return self.rag_chain.invoke({"input": question})["answer"]

    def ask_with_sources(self, question):
        """Returns answer + top-3 source chunks with page number and snippet."""
        if not self.rag_chain or not self.vector_db:
            return {"answer": "Please upload a PDF first.", "sources": []}

        # Get the answer via chain
        result = self.rag_chain.invoke({"input": question})
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

    def generate_followups(self, question, answer):
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