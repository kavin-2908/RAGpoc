import logging
from django.conf import settings

from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self):
        db_settings = settings.DATABASES['default']
        
        # Build connection string for langchain-postgres (psycopg3)
        self.connection_string = (
            f"postgresql+psycopg://{db_settings.get('USER', 'postgres')}:"
            f"{db_settings.get('PASSWORD', 'postgres')}@"
            f"{db_settings.get('HOST', 'localhost')}:"
            f"{db_settings.get('PORT', '5432')}/"
            f"{db_settings.get('NAME', 'rag_db')}"
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name="rag_documents",
            connection=self.connection_string,
            use_jsonb=True,
        )

    def get_vectorstore(self):
        return self.vectorstore
