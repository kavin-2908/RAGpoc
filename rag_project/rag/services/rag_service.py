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

# Natural, efficient prompt template designed for conversational NLP synthesis
PROMPT_TEMPLATE = """You are an intelligent, articulate, and conversational AI assistant. Your goal is to synthesize the provided context into natural, fluent, and helpful answers for the user.

### INSTRUCTIONS:
1. **Conversational Synthesis**: Formulate your response in natural, fluent English using complete sentences. Do NOT output raw keyword fragments, pipe-separated lists, or unformatted database dumps unless specifically asked.
2. **Interpret Structured & Database Records**: The context may contain structured records from database tables or documents (e.g. key-value pairs, IDs, field attributes). Translate these records into natural, easy-to-read conversational statements (e.g., explain what each record represents in clear prose rather than echoing raw technical syntax).
3. **Strict Factual Grounding**: Base all facts, numbers, names, and entity attributes strictly on the provided Context. Do NOT invent, assume, or extrapolate facts outside the context.
4. **Structured Presentation**: Use clear paragraphs, bullet points, or bold highlights when explaining multiple items, entities, or steps to keep the answer clean and readable.
5. **Polite Fallback**: If the context does not contain enough information to answer the question, respond politely and conversationally: "I could not find sufficient information in the ingested database or uploaded documents to answer your question."

### CONTEXT:
{context}

### USER QUESTION:
{question}

### CONVERSATIONAL ANSWER:"""


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
        formatted_chunks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('filename', 'Unknown Source')
            table = doc.metadata.get('table')
            header = f"[Source {i}: {source} (Table: {table})]" if table else f"[Source {i}: {source}]"
            formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n".join(formatted_chunks)

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
                "answer": "I could not find sufficient information in the ingested database or uploaded documents to answer your question.",
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
