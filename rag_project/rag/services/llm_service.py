"""
LLM Service — Communicates with the local Ollama instance via LangChain.
"""

import logging

from django.conf import settings
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles communication with the local Ollama LLM using LangChain.
    """

    def __init__(self):
        self.base_url = settings.OLLAMA_URL.rstrip('/')
        self.model_name = settings.MODEL_NAME
        
        # Initialize LangChain's Ollama wrapper
        self.llm = Ollama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=0  # For more deterministic RAG answers
        )

    def get_llm(self):
        """
        Return the LangChain LLM instance for use in chains.
        """
        return self.llm

    def generate(self, prompt: str) -> str:
        """
        Backward compatibility for direct generation if needed.
        """
        logger.info("Sending prompt to Ollama (model: %s)", self.model_name)
        response = self.llm.invoke(prompt)
        return response.strip()
