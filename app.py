from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# 1. LOAD
print("🔄 Loading PDF...")
loader = PyPDFLoader("pride-and-prejudice-jane-austen.pdf")
data = loader.load()

# 2. SEMANTIC SPLIT
# We use Ollama's nomic-embed-text to "understand" the sentences
print("🧠 Analyzing text meaning for semantic splits...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# The 'percentile' threshold means: "Split if the difference in meaning 
# between sentences is in the top 10% of differences found in the doc."
text_splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90 
)
chunks = text_splitter.split_documents(data)
print(f"✅ Created {len(chunks)} semantically coherent chunks.")

# 3. VECTOR STORE
vector_db = FAISS.from_documents(chunks, embeddings)

# 4. RAG CHAIN
llm = ChatOllama(model="llama3.2", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

# 5. TERMINAL UI
while True:
    query = input("\n[Semantic AI] Ask anything (or 'exit'): ")
    if query in ['exit', 'quit']: break
    
    response = qa_chain.invoke(query)
    print(f"\nAnswer: {response['result']}")