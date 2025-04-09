from langchain import hub

from rag_ta.document_loaders import (
    load_ipynb_documents,
    split_documents,
    load_all_transcripts,
)
from rag_ta.vector_store import initialize_vector_store

from langchain.chat_models import init_chat_model


def init_vector_store(root_path="/Users/sglyon/Downloads/course_24", populate=True):
    vs = initialize_vector_store()

    if populate:
        # Identify knowledge base
        ipynbs = load_ipynb_documents(root_path)
        transcripts = load_all_transcripts(root_path)
        docs = ipynbs + transcripts

        # split into chunks
        chunks = split_documents(docs)

        # embed in vector store with metadata
        vs.add_documents(documents=chunks)

    return vs


class RAG:
    def __init__(self, populate=True):
        self.vs = init_vector_store(populate=populate)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    def run(self, question: str):
        # User enters question
        # embed question to get related cos
        retrieved_docs = self.vs.similarity_search(question)

        # get context
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # generate response
        messages = self.prompt.invoke({"question": question, "context": docs_content})
        response = self.llm.invoke(messages)

        return {"response": response, "docs": retrieved_docs}

    def format_response(self, response: str, docs: list):
        out = ""
        for doc in docs:
            out += f"## {doc.metadata.get('source', '')} -- {doc.metadata.get('title', '')}\n\n{doc.page_content}\n\n"
        out += f"\n\n{response}"
        return out
