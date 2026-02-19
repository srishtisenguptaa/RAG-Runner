import os
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

# Use the modern Tavily import to avoid deprecation warnings
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document

# Load .env at the very top
load_dotenv()

# --- Define the State ---
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str 
    documents: List[str]
    # Logs will store our "Thought Trace" for Streamlit
    logs: List[str]

class ArchitectRAG:
    def __init__(self):
        if not os.getenv("TAVILY_API_KEY"):
            print("⚠️ WARNING: TAVILY_API_KEY not found in environment.")
            
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        
        # FIX: Use the class name and 'max_results' parameter
        self.web_search_tool = TavilySearch(max_results=3) 
        self.vector_db = None

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = SemanticChunker(self.embeddings)
        chunks = text_splitter.split_documents(docs)
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    # --- Node 1: Retrieve ---
    def retrieve(self, state: GraphState):
        print("---RETRIEVING FROM PDF---")
        question = state["question"]
        documents = self.vector_db.similarity_search(question)
        
        return {
            "documents": documents, 
            "question": question, 
            "logs": ["Successfully extracted relevant passages from the PDF."]
        }

    # --- Node 2: Grade Documents ---
    def grade_documents(self, state: GraphState):
        print("---CHECKING RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        web_search = "No"
        
        # Optimized prompt for Llama 3.2
        grading_prompt = ChatPromptTemplate.from_template(
            """Determine if the following document is highly relevant to the question: '{question}'
            Document: {doc}
            Respond only with 'yes' or 'no'."""
        )
        grader_chain = grading_prompt | self.llm
        
        for d in documents:
            score = grader_chain.invoke({"question": question, "doc": d.page_content})
            if "yes" in score.content.lower():
                filtered_docs.append(d)
            else:
                # If even one document is irrelevant, we flag for web search to be safe
                web_search = "Yes" 
        
        log_msg = f"Relevance Check: {len(filtered_docs)}/{len(documents)} docs passed. Web search: {web_search}"
        
        return {
            "documents": filtered_docs, 
            "question": question, 
            "web_search": web_search, 
            "logs": state.get("logs", []) + [log_msg]
        }

    # --- Node 3: Web Search Fallback ---
    def web_search(self, state: GraphState):
        print("---SEARCHING THE WEB---")
        question = state["question"]
        documents = state.get("documents", [])
        
        # 1. Get the search results
        search_results = self.web_search_tool.invoke({"query": question})
        
        # 2. Safely parse the results
        # If search_results is already a string, use it directly
        if isinstance(search_results, str):
            web_content = search_results
        else:
            # If it's a list of dicts, join the content
            web_content = "\n".join([res.get("content", "") for res in search_results])
        
        # 3. Create a Document object (to keep state consistent)
        web_doc = Document(page_content=web_content, metadata={"source": "tavily_search"})
        documents.append(web_doc)
        
        return {
            "documents": documents, 
            "question": question, 
            "logs": state.get("logs", []) + ["Internal PDF data insufficient. Fetched real-time data from Tavily Search."]
    }

    # --- Node 4: Generate Answer ---
  # --- Node 4: Generate Answer ---
    def generate(self, state: GraphState):
        print("---GENERATING FINAL ANSWER---")
        question = state["question"]
        documents = state["documents"]
        
        # ADD THIS HERE: 
        # This converts the list of Document objects into one big string
        context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents])
        
        prompt = ChatPromptTemplate.from_template(
            "Answer the question using ONLY the following context:\n{context}\n\nQuestion: {question}"
        )
        
        chain = prompt | self.llm
        
        # Pass 'context' (the string) instead of 'documents' (the list)
        response = chain.invoke({"context": context, "question": question})
        
        return {"generation": response.content}

    # --- Logic: Routing ---
    def decide_to_generate(self, state: GraphState):
        if state["web_search"] == "Yes":
            return "search_web"
        return "generate"

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
            {
                "search_web": "web_search",
                "generate": "generate",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()