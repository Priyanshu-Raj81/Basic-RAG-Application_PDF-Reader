import streamlit as st
import os
from rag_utils import build_rag_chain

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("RAG Application – Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")

    if "rag_chain" not in st.session_state:
        with st.spinner("Building knowledge base..."):
            st.session_state.rag_chain = build_rag_chain(pdf_path)

    question = st.chat_input("Ask a question from the PDF")

    if question:
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(question)

        st.chat_message("user").write(question)
        st.chat_message("assistant").write(response.content)
