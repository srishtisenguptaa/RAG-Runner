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
        # Runs on CPU instantly
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
        if not self.rag_chain: return "Please upload a book first."
        return self.rag_chain.invoke({"input": question})["answer"]