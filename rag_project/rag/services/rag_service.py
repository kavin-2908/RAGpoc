"""
RAG Service — Orchestrates the Retrieval-Augmented Generation flow.

Flow:
    1. Embed the user's question.
    2. Perform vector similarity search to get top-K chunks.
    3. Construct the prompt with system rules, context, and question.
    4. Send the prompt to the LLM.
    5. Return the answer along with the source filenames.

This service coordinates EmbeddingService, VectorService, and LLMService.
It contains no DRF imports — pure Python.
"""

import logging

from django.conf import settings

from rag.services.embedding_service import EmbeddingService
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
    Orchestrates the chat flow for the Retrieval-Augmented Generation pipeline.
    """

    def __init__(self):
        self.top_k: int = settings.TOP_K_RESULTS
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()
        self.llm_service = LLMService()

    def answer_question(self, question: str) -> dict:
        """
        Process a user question, retrieve context, and generate an answer.

        Args:
            question: The user's input question string.

        Returns:
            dict containing:
                - 'answer': The LLM's generated response string.
                - 'sources': List of unique filenames used as context.
        """
        logger.info("Received question: %s", question)

        # Step 1: Generate embedding for the question
        query_embedding = self.embedding_service.embed_query(question)

        # Step 2: Retrieve the most relevant chunks from PostgreSQL
        chunks = self.vector_service.similarity_search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        if not chunks:
            logger.info("No relevant chunks found in the database.")
            return {
                "answer": "I could not find enough information in the uploaded documents. (The database is empty).",
                "sources": [],
            }

        # Extract unique source filenames from the retrieved chunks
        # Using a set preserves uniqueness, then we convert back to a list
        sources = list(set([chunk.document.filename for chunk in chunks]))
        logger.info("Retrieved context from sources: %s", sources)

        # Step 3: Build the prompt
        # Join all chunk contents with a double newline to clearly separate them
        context_text = "\n\n".join([chunk.content for chunk in chunks])
        
        prompt = PROMPT_TEMPLATE.format(
            context=context_text,
            question=question
        )

        # Step 4: Send prompt to Ollama
        answer = self.llm_service.generate(prompt)

        # Step 5: Return answer and sources
        return {
            "answer": answer,
            "sources": sources,
        }
