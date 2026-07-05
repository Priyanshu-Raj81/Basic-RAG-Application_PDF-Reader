import os
import hashlib

import streamlit as st
from rag_utils import build_rag_chain, RAGError

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("RAG Application – Chat with your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "sources": [...]}
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getbuffer()
    file_id = hashlib.md5(file_bytes).hexdigest()

    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)

    # Only (re)build the chain when the uploaded file actually changes.
    # This fixes the original bug where uploading a second PDF kept
    # answering questions using the first PDF's index.
    if file_id != st.session_state.current_file_id:
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        with st.spinner("Building knowledge base..."):
            try:
                st.session_state.rag_chain = build_rag_chain(pdf_path)
                st.session_state.current_file_id = file_id
                st.session_state.messages = []  # fresh chat for a new document
                st.success(f"'{uploaded_file.name}' is ready. Ask a question below.")
            except RAGError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error while processing the PDF: {e}")
                st.stop()

    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**Page {src['page']}:** {src['excerpt']}...")

    question = st.chat_input("Ask a question from the PDF")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain.invoke(question)
                except RAGError as e:
                    result = {"answer": f"Something went wrong: {e}", "sources": []}
                except Exception as e:
                    result = {"answer": f"Unexpected error: {e}", "sources": []}

            st.write(result["answer"])
            if result.get("sources"):
                with st.expander("Sources"):
                    for src in result["sources"]:
                        st.markdown(f"**Page {src['page']}:** {src['excerpt']}...")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            }
        )
else:
    st.info("Upload a PDF to get started.")