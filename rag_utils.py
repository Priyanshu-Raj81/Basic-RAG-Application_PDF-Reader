from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def build_rag_chain(pdf_path):
    # 1️. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2️. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # 3️. Create embeddings & FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    # 4️. Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    # 5️. LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # 6️. RAG Chain (LCEL / Runnable)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain
