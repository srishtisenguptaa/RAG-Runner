import os
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings # Swapped for speed/deployment
from langchain_groq import ChatGroq                # Swapped from Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # Faster indexing

load_dotenv()

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str 
    documents: List[Document]
    logs: List[str]

class ArchitectRAG:
    def __init__(self):
        # 1. Faster, CPU-friendly embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Lightning-fast Groq LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.web_search_tool = TavilySearch(max_results=3) 
        self.vector_db = None

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Switched to Recursive for speed; Semantic is 10x slower on indexing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    # --- Nodes (Logic remains identical, just uses Groq) ---
    def retrieve(self, state: GraphState):
        question = state["question"]
        documents = self.vector_db.similarity_search(question)
        return {"documents": documents, "question": question, "logs": ["Retrieved PDF context."]}

    def grade_documents(self, state: GraphState):
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"
        
        grader_prompt = ChatPromptTemplate.from_template(
            "Respond ONLY with 'yes' or 'no'. Is this doc relevant to '{question}'?\nDoc: {doc}"
        )
        grader_chain = grader_prompt | self.llm
        
        for d in documents:
            score = grader_chain.invoke({"question": question, "doc": d.page_content})
            if "yes" in score.content.lower():
                filtered_docs.append(d)
        
        if not filtered_docs: web_search = "Yes"
        
        return {
            "documents": filtered_docs, 
            "web_search": web_search, 
            "logs": state.get("logs", []) + [f"Grader: {len(filtered_docs)} relevant docs found."]
        }

    def web_search(self, state: GraphState):
        question = state["question"]
        search_results = self.web_search_tool.invoke({"query": question})
        web_doc = Document(page_content=str(search_results), metadata={"source": "tavily_search"})
        docs = state.get("documents", [])
        docs.append(web_doc)
        return {"documents": docs, "logs": state.get("logs", []) + ["Added Web Search data."]}

    def generate(self, state: GraphState):
        context = "\n\n".join([d.page_content for d in state["documents"]])
        sources = {d.metadata.get("source", "PDF") for d in state["documents"]}
        
        prompt = ChatPromptTemplate.from_template(
            "Use context to answer: {question}\nContext: {context}"
        )
        response = (prompt | self.llm).invoke({"context": context, "question": state["question"]})
        
        final_answer = f"{response.content}\n\n**Sources:** {', '.join(sources)}"
        return {"generation": final_answer, "logs": state.get("logs", []) + ["Final answer generated."]}

    def decide_to_generate(self, state: GraphState):
        return "search_web" if state["web_search"] == "Yes" else "generate"

    def build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("web_search", self.web_search)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", self.decide_to_generate, 
                                       {"search_web": "web_search", "generate": "generate"})
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()