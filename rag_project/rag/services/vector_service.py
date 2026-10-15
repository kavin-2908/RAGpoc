import logging
import urllib.parse
from django.conf import settings

from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class VectorService:
    _embeddings = None
    _vectorstore = None

    def __init__(self):
        db_settings = settings.DATABASES['default']
        
        user = urllib.parse.quote_plus(db_settings.get('USER', 'postgres'))
        password = urllib.parse.quote_plus(db_settings.get('PASSWORD', 'postgres'))
        host = db_settings.get('HOST', 'localhost')
        port = db_settings.get('PORT', '5432')
        name = db_settings.get('NAME', 'rag_db')
        
        # Build connection string for langchain-postgres (psycopg3)
        self.connection_string = (
            f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"
        )
        
        if VectorService._embeddings is None:
            logger.info("Initializing HuggingFace embedding model '%s' (cached for future requests)...", settings.EMBEDDING_MODEL_NAME)
            VectorService._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME
            )
        self.embeddings = VectorService._embeddings
        
        if VectorService._vectorstore is None:
            VectorService._vectorstore = PGVector(
                embeddings=self.embeddings,
                collection_name="rag_documents",
                connection=self.connection_string,
                use_jsonb=True,
            )
        self.vectorstore = VectorService._vectorstore

    def get_vectorstore(self):
        return self.vectorstore

