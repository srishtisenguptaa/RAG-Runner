import streamlit as st
import requests

st.set_page_config(page_title="The Architect RAG", layout="wide")
st.title("📚 Intelligent Research Assistant")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Toggle for Model Selection
    # This prevents the backend from running both; it only calls the one selected.
    rag_mode = st.radio(
        "Select RAG Engine:",
        options=["Standard", "Architect"],
        help="Standard: Fast, PDF-only retrieval. \nArchitect: Advanced, self-correcting agent with Web Search fallback."
    )
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing & Chunking..."):
            # Sending the file to the shared upload endpoint
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post("http://localhost:8000/upload", files=files)
            if response.status_code == 200:
                st.success(f"Indexed into {response.json().get('chunks_created')} semantic chunks!")

# Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Running {rag_mode} logic..."):
            try:
                # We pass the 'mode' as a query parameter to match our FastAPI update
                mode_param = rag_mode.lower()
                res = requests.post(
                    f"http://localhost:8000/chat?mode={mode_param}", 
                    json={"prompt": prompt}
                )
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer")
                    source_info = data.get("sources", "N/A")
                    
                    # Displaying the answer with a small sub-header for transparency
                    st.markdown(answer)
                    st.caption(f"🔍 Source: {source_info} | 🧠 Engine: {data.get('mode')}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Backend error. Make sure the PDF is uploaded and FastAPI is running.")
            
            except Exception as e:
                st.error(f"Connection failed: {e}")