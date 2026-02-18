import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGSystem:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        self.vector_db = None
        self.rag_chain = None

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Use semantic splitting to group text by meaning
        text_splitter = SemanticChunker(self.embeddings)
        chunks = text_splitter.split_documents(docs)
        
        # Create the Vector Database
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        
        # Setup the chain
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based ONLY on the provided context:\n\n{context}\n\nQuestion: {input}"
        )
        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.vector_db.as_retriever(), qa_chain)
        return len(chunks)

    def ask(self, question):
        if not self.rag_chain:
            return "Please upload a book first."
        response = self.rag_chain.invoke({"input": question})
        return response["answer"]