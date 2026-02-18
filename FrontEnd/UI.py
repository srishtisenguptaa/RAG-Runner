import streamlit as st
import requests

st.title("📖 Semantic Book Reader")

# Sidebar for Uploads
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload a Book (PDF)", type="pdf")
    if uploaded_file and st.button("Process Book"):
        with st.spinner("Processing..."):
            files = {"file": uploaded_file.getvalue()}
            # Note: We send the filename as a string
            response = requests.post("http://localhost:8000/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue())})
            if response.status_code == 200:
                st.success("Book Indexed Successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the book..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        res = requests.post("http://localhost:8000/chat", json={"prompt": prompt})
        answer = res.json().get("answer", "Error connecting to API.")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})