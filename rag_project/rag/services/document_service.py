import logging
import os
from pathlib import Path

from django.conf import settings

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.models import Document
from rag.services.vector_service import VectorService

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.upload_dir: Path = settings.UPLOAD_DIR
        self.chunk_size: int = settings.CHUNK_SIZE
        self.chunk_overlap: int = settings.CHUNK_OVERLAP
        
        self.vector_service = VectorService()
        self.vectorstore = self.vector_service.get_vectorstore()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process_document(self, file, filename: str) -> dict:
        file_type = self._get_file_type(filename)
        file_path = self._save_file(file, filename)
        
        try:
            # 1. Load document using LangChain Loaders
            loader = self._get_loader(file_path, file_type)
            docs = loader.load()
            
            if not docs:
                raise ValueError("No text could be extracted.")
                
            # Add filename to metadata
            for doc in docs:
                doc.metadata['filename'] = filename
                
            # 2. Split documents
            chunks = self.text_splitter.split_documents(docs)
            
            if not chunks:
                raise ValueError("Could not split document into chunks.")
                
            # 3. Store in LangChain PGVector
            self.vectorstore.add_documents(chunks)
            
            # 4. Save metadata to Django model
            document_record = Document.objects.create(
                filename=filename,
                file_type=file_type
            )
            
            return {
                "document_id": document_record.id,
                "chunks_created": len(chunks),
            }
        except Exception:
            self._delete_file(file_path)
            raise

    def process_database(self) -> dict:
        # In a full LangChain implementation, you could use SQLDatabaseLoader
        # But for now, we'll just raise NotImplementedError or build a simple loader
        from langchain_community.utilities import SQLDatabase
        from langchain_community.document_loaders.sqlalchemy import SQLAlchemyLoader
        
        # We store it under a virtual filename indicating it's from the database
        virtual_filename = f"mysql_db_{settings.SOURCE_DB_NAME}"
        
        raise NotImplementedError("Database ingestion using LangChain is pending implementation.")

    def _get_file_type(self, filename: str) -> str:
        if '.' not in filename:
            raise ValueError(f"Filename '{filename}' has no extension.")
        return filename.rsplit('.', 1)[-1].lower()

    def _get_loader(self, file_path: Path, file_type: str):
        path_str = str(file_path)
        if file_type == 'pdf':
            return PyPDFLoader(path_str)
        elif file_type == 'txt':
            return TextLoader(path_str)
        elif file_type == 'csv':
            return CSVLoader(path_str)
        elif file_type in ['docx', 'doc']:
            return UnstructuredWordDocumentLoader(path_str)
        elif file_type in ['xlsx', 'xls']:
            return UnstructuredExcelLoader(path_str)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _save_file(self, file, filename: str) -> Path:
        file_path = self.upload_dir / filename
        with open(file_path, 'wb') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        return file_path

    def _delete_file(self, file_path: Path) -> None:
        try:
            if file_path.exists():
                os.remove(file_path)
        except OSError as e:
            logger.warning("Failed to clean up file %s: %s", file_path, e)
