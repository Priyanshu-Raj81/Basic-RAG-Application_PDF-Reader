import os
import re
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = "faiss_indexes"

# Simple, fast heuristic for greetings / small talk so we never send
# these to the retriever+LLM with a "context-only" prompt. This is what
# was causing "Hyy" -> "I don't know".
GREETING_PATTERN = re.compile(
    r"^\s*(hi+|he+llo+|hy+|hey+|yo|sup|good\s?(morning|afternoon|evening)|"
    r"thanks?|thank\s?you|bye|goodbye)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

# Questions ABOUT the assistant itself ("what can you do", "how can you help
# me", "who are you") are not document questions either — they were falling
# through to the retriever and coming back as "I couldn't find that in the
# document", which is confusing since the user isn't asking about the PDF.
META_PATTERN = re.compile(
    r"^\s*(what\s+can\s+you\s+do|how\s+can\s+you\s+help(\s+me)?|"
    r"what\s+do\s+you\s+do|who\s+are\s+you|what\s+is\s+this(\s+app)?|"
    r"how\s+(does\s+this|do\s+you)\s+work|how\s+are\s+you)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


class RAGError(Exception):
    """Base exception for problems building or running the RAG chain."""


class PDFLoadError(RAGError):
    """Raised when the PDF can't be loaded or has no extractable text."""


class MissingAPIKeyError(RAGError):
    """Raised when OPENAI_API_KEY is not set."""


def is_small_talk(text: str) -> bool:
    """Detect greetings/small talk that shouldn't be treated as document questions."""
    return bool(GREETING_PATTERN.match(text or ""))


def is_meta_question(text: str) -> bool:
    """Detect questions about the assistant itself, e.g. 'how can you help me'."""
    return bool(META_PATTERN.match(text or ""))


def _check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project root:\n"
            "OPENAI_API_KEY=sk-..."
        )


def _index_path_for(pdf_path: str) -> str:
    """Derive a stable, per-file index directory so we can cache embeddings."""
    file_hash = hashlib.md5(os.path.abspath(pdf_path).encode()).hexdigest()[:12]
    return os.path.join(INDEX_DIR, file_hash)


def build_rag_chain(pdf_path: str):
    """
    Build (or load a cached) retrieval chain for the given PDF.

    Returns a Runnable that takes a plain question string and returns:
        {"answer": str, "sources": list[dict], "small_talk": bool}
    """
    _check_api_key()

    if not os.path.exists(pdf_path):
        raise PDFLoadError(f"File not found: {pdf_path}")

    embeddings = OpenAIEmbeddings()
    index_path = _index_path_for(pdf_path)

    if os.path.exists(index_path):
        # Reuse cached embeddings instead of re-embedding on every run.
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            raise PDFLoadError(f"Could not read PDF '{pdf_path}': {e}") from e

        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise PDFLoadError(
                "No extractable text found in this PDF. It may be a scanned/"
                "image-only document that needs OCR."
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            raise RAGError(f"Failed to build embeddings/vector store: {e}") from e

        os.makedirs(INDEX_DIR, exist_ok=True)
        vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that answers questions about a specific PDF document.

        Guidelines:
        - Base your answer on the context below, but you are not limited to
          copying it verbatim. You may summarize, compare, explain, or make
          suggestions/recommendations (e.g. project ideas, examples, use cases)
          as long as they are reasonably grounded in the topics and ideas
          present in the context.
        - Do not invent specific facts, numbers, or claims that aren't
          supported by the context.
        - Only say "I couldn't find that in the document." if the topic the
          user is asking about is genuinely not covered anywhere in the
          context — not just because the exact wording isn't there.

        Context:
        {context}

        Question:
        {question}
        """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _format_sources(docs):
        seen = set()
        sources = []
        for d in docs:
            page = d.metadata.get("page", "?")
            if page in seen:
                continue
            seen.add(page)
            sources.append({"page": page, "excerpt": d.page_content[:200]})
        return sources

    # Retrieve once, reuse the docs both for the prompt context and for
    # returning source citations to the UI.
    retrieval_step = RunnableParallel(
        question=RunnableLambda(lambda q: q),
        docs=retriever,
    )

    def _answer(inputs):
        question = inputs["question"]
        docs = inputs["docs"]

        context = "\n\n".join(d.page_content for d in docs)
        messages = prompt.format_messages(context=context, question=question)
        try:
            response = llm.invoke(messages)
        except Exception as e:
            raise RAGError(f"LLM call failed: {e}") from e

        return {
            "answer": response.content,
            "sources": _format_sources(docs),
            "small_talk": False,
        }

    def _small_talk_response(question):
        return {
            "answer": "Hey! I'm ready to answer questions about the PDF you uploaded — what would you like to know?",
            "sources": [],
            "small_talk": True,
        }

    def _meta_response(question):
        return {
            "answer": (
                "I've read the PDF you uploaded and can answer questions about it, "
                "summarize sections, explain concepts it covers, or make suggestions "
                "(like project ideas) based on its content. Just ask away!"
            ),
            "sources": [],
            "small_talk": True,
        }

    full_rag_step = retrieval_step | RunnableLambda(_answer)

    rag_chain = RunnableBranch(
        (lambda q: is_small_talk(q), RunnableLambda(_small_talk_response)),
        (lambda q: is_meta_question(q), RunnableLambda(_meta_response)),
        full_rag_step,
    )
    return rag_chain