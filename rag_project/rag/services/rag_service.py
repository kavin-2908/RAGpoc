"""
RAG Service — Orchestrates the Retrieval-Augmented Generation flow via LangChain.

Flow:
    1. If the question is a direct conversational greeting or general conversational query,
       generate an answer directly using the LLM.
    2. Attempt to retrieve relevant chunks from the vectorstore.
    3. If no context is present in the dataset (or retrieval yields no documents):
       - If the question is general, query the LLM directly to generate the answer.
       - If the question is asking for missing dataset/document facts, inform politely.
    4. If context is retrieved:
       - Run RAG synthesis chain.
       - If the context answers the question, attribute the sources.
       - If the question is general and the context is irrelevant, answer directly from
         general knowledge and omit irrelevant sources.
"""

import logging
import re

from django.conf import settings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.services.llm_service import LLMService
from rag.services.vector_service import VectorService

logger = logging.getLogger(__name__)

# Direct prompt template used when no context is present in the dataset or for general conversation
DIRECT_PROMPT_TEMPLATE = """You are an intelligent, articulate, and conversational AI assistant. Your goal is to provide a helpful, natural, and accurate answer to the user.

### INSTRUCTIONS:
1. **General Questions & Conversation**: If the question is general (such as a greeting, general knowledge query, explanation, definition, coding problem, math calculation, or conversational topic), answer it accurately, politely, and comprehensively using your knowledge.
2. **Dataset / Document Inquiries**: If the user is specifically asking to look up or query private database records, uploaded documents, or internal files that require specific ingested dataset context (which is currently unavailable), politely inform them: "I could not find sufficient information in the ingested database or uploaded documents to answer your question."

### USER QUESTION:
{question}

### CONVERSATIONAL ANSWER:"""

# Natural, efficient prompt template designed for conversational NLP synthesis
PROMPT_TEMPLATE = """You are an intelligent, articulate, and conversational AI assistant. Your goal is to synthesize the provided context into natural, fluent, and helpful answers for the user.

### INSTRUCTIONS:
1. **Conversational Synthesis**: Formulate your response in natural, fluent English using complete sentences. Do NOT output raw keyword fragments, pipe-separated lists, or unformatted database dumps unless specifically asked.
2. **Interpret Structured & Database Records**: The context may contain structured records from database tables or documents (e.g. key-value pairs, IDs, field attributes). Translate these records into natural, easy-to-read conversational statements (e.g., explain what each record represents in clear prose rather than echoing raw technical syntax).
3. **Context Grounding & General Questions**:
   - If the provided Context contains information relevant to the user's question, base your response directly and factually on that Context.
   - If the provided Context does NOT contain information relevant to the question, and the question is a general question (e.g. greetings, general knowledge, explanations, definitions, coding, or everyday conversation), answer the question directly and conversationally using your general knowledge. Do not mention or reference the unrelated context chunks.
   - If the user specifically asks for document-specific, proprietary, or database records that are not in the Context, respond politely: "I could not find sufficient information in the ingested database or uploaded documents to answer your question."
4. **Structured Presentation**: Use clear paragraphs, bullet points, or bold highlights when explaining multiple items, entities, or steps to keep the answer clean and readable.
5. **Source Attribution Marker**:
   - If you used information from the provided Context to answer the question, append `<!-- USED_CONTEXT: YES -->` at the very end of your response.
   - If you answered a general question from your general knowledge because the Context was not relevant or lacked the answer, append `<!-- USED_CONTEXT: NO -->` at the very end of your response.

### CONTEXT:
{context}

### USER QUESTION:
{question}

### CONVERSATIONAL ANSWER:"""


class RAGService:
    """
    Orchestrates the chat flow using LangChain Expression Language (LCEL).
    Supports RAG context grounding as well as direct LLM generation for general queries.
    """

    def __init__(self):
        self.top_k: int = getattr(settings, 'TOP_K_RESULTS', 3)
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()

        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        self.direct_prompt = PromptTemplate.from_template(DIRECT_PROMPT_TEMPLATE)

        self.vectorstore = None
        self.retriever = None
        try:
            self.vector_service = VectorService()
            self.vectorstore = self.vector_service.get_vectorstore()
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        except Exception as e:
            logger.warning("Vector store retriever initialization skipped or unavailable: %s", e)

    @staticmethod
    def _is_conversational_greeting(question: str) -> bool:
        """
        Check if the input is a simple greeting or pleasantry that can be answered directly.
        """
        normalized = question.strip().lower().rstrip("?!. ")
        greetings = {
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
            "good evening", "how are you", "who are you", "what are you",
            "what can you do", "help", "thanks", "thank you", "bye", "goodbye"
        }
        return normalized in greetings

    def _generate_direct_answer(self, question: str) -> str:
        """
        Query the LLM directly without external document context.
        """
        logger.info("Generating direct answer from LLM for question: %s", question)
        direct_chain = (
            {"question": RunnablePassthrough()}
            | self.direct_prompt
            | self.llm
            | StrOutputParser()
        )
        return direct_chain.invoke(question).strip()

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
        Process a user question, retrieve context if available, and generate an answer.
        If there is no context present in the dataset and the question is general,
        or if the question is conversational, the LLM is asked directly.

        Args:
            question: The user's input question string.

        Returns:
            dict containing:
                - 'answer': The LLM's generated response string.
                - 'sources': List of unique filenames used as context (empty if answered directly).
        """
        clean_query = question.strip()
        logger.info("Received question: %s", clean_query)

        # 1. Quick bypass for direct greetings and pleasantries
        if self._is_conversational_greeting(clean_query):
            answer = self._generate_direct_answer(clean_query)
            return {
                "answer": answer,
                "sources": [],
            }

        # 2. Retrieve documents from vectorstore
        docs = []
        if self.retriever is not None:
            try:
                docs = self.retriever.invoke(clean_query)
            except Exception as e:
                logger.warning("Vector retrieval encountered an issue: %s. Falling back to direct LLM.", e)
                docs = []

        # 3. If no context is present in the dataset, ask the LLM directly
        if not docs:
            logger.info("No context present in dataset for question: '%s'. Querying LLM directly.", clean_query)
            answer = self._generate_direct_answer(clean_query)
            return {
                "answer": answer,
                "sources": [],
            }

        # Extract unique source filenames from metadata
        sources = list(set([doc.metadata.get('filename', 'Unknown') for doc in docs]))
        logger.info("Retrieved context from sources: %s", sources)

        # 4. Build and invoke the LCEL RAG chain
        rag_chain = (
            {"context": lambda x: self._format_docs(docs), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        raw_answer = rag_chain.invoke(clean_query)
        answer = raw_answer.strip()

        # 5. Check if LLM answered from general knowledge rather than the provided context
        if re.search(r'<!--\s*USED_CONTEXT:\s*NO\s*-->', answer, re.IGNORECASE):
            logger.info("LLM answered general question without using context chunks. Clearing sources.")
            sources = []
        elif "I could not find sufficient information in the ingested database" in answer:
            sources = []

        # Remove source attribution markers from final output
        clean_answer = re.sub(r'<!--\s*USED_CONTEXT:\s*(YES|NO)\s*-->', '', answer, flags=re.IGNORECASE).strip()

        return {
            "answer": clean_answer,
            "sources": sources,
        }
