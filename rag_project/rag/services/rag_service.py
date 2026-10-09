"""
RAG Service — Orchestrates the Retrieval-Augmented Generation flow via LangChain.

Flow:
    1. Retrieve the most relevant chunks using LangChain's vectorstore as_retriever.
    2. Construct the LCEL chain (Context + Question -> Prompt -> LLM).
    3. Return the answer along with the source filenames.
"""

import logging

from django.conf import settings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.services.llm_service import LLMService
from rag.services.vector_service import VectorService

logger = logging.getLogger(__name__)

# The specific prompt template requested for this project
PROMPT_TEMPLATE = """You are an AI assistant.
Only answer using the provided context.
If the answer is not present in the context, say "I could not find enough information in the uploaded documents."
Never hallucinate.

Context:
{context}

Question:
{question}

Answer:"""


class RAGService:
    """
    Orchestrates the chat flow using LangChain Expression Language (LCEL).
    """

    def __init__(self):
        self.top_k: int = settings.TOP_K_RESULTS
        self.vector_service = VectorService()
        self.vectorstore = self.vector_service.get_vectorstore()
        
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()
        
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_question(self, question: str) -> dict:
        """
        Process a user question, retrieve context, and generate an answer using LangChain.

        Args:
            question: The user's input question string.

        Returns:
            dict containing:
                - 'answer': The LLM's generated response string.
                - 'sources': List of unique filenames used as context.
        """
        logger.info("Received question: %s", question)

        # 1. Retrieve documents to extract sources
        docs = self.retriever.invoke(question)
        
        if not docs:
            logger.info("No relevant chunks found in the database.")
            return {
                "answer": "I could not find enough information in the uploaded documents.",
                "sources": [],
            }

        # Extract unique source filenames from metadata
        sources = list(set([doc.metadata.get('filename', 'Unknown') for doc in docs]))
        logger.info("Retrieved context from sources: %s", sources)

        # 2. Build the LCEL RAG chain
        rag_chain = (
            {"context": lambda x: self._format_docs(docs), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # 3. Invoke the chain
        answer = rag_chain.invoke(question)

        return {
            "answer": answer.strip(),
            "sources": sources,
        }
