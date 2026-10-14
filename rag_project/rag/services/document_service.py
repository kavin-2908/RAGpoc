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
        """
        Extract data from the configured MySQL database, format each row into
        natural, semantically rich NLP descriptions, chunk it, embed it,
        and store it in PostgreSQL vector database.
        """
        import pymysql
        from langchain_core.documents import Document as LCDocument

        logger.info(
            "Connecting to source database %s on %s:%s",
            settings.SOURCE_DB_NAME,
            settings.SOURCE_DB_HOST,
            settings.SOURCE_DB_PORT,
        )

        virtual_filename = f"mysql_db_{settings.SOURCE_DB_NAME}"

        try:
            connection = pymysql.connect(
                host=settings.SOURCE_DB_HOST,
                user=settings.SOURCE_DB_USER,
                password=settings.SOURCE_DB_PASSWORD,
                database=settings.SOURCE_DB_NAME,
                port=int(settings.SOURCE_DB_PORT),
                cursorclass=pymysql.cursors.DictCursor,
            )
        except Exception as e:
            logger.error("Failed to connect to source database: %s", e)
            raise ValueError(f"Database connection failed: {e}") from e

        docs: list[LCDocument] = []

        try:
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                if not tables:
                    raise ValueError(f"No tables found in database '{settings.SOURCE_DB_NAME}'.")

                for table_dict in tables:
                    table_name = list(table_dict.values())[0]
                    logger.info("Ingesting table data from: %s", table_name)

                    cursor.execute(f"SELECT * FROM `{table_name}`")
                    rows = cursor.fetchall()

                    for row in rows:
                        # Convert database columns into clear, descriptive natural language sentences
                        field_items = [
                            f"{k.replace('_', ' ')} is {v}"
                            for k, v in row.items()
                            if v is not None and str(v).strip() != ""
                        ]
                        if not field_items:
                            continue

                        # Natural English representation of each record
                        natural_row_text = (
                            f"Record in table '{table_name}': "
                            + ", ".join(field_items)
                            + "."
                        )

                        docs.append(
                            LCDocument(
                                page_content=natural_row_text,
                                metadata={
                                    "filename": virtual_filename,
                                    "table": table_name,
                                    "source_type": "database",
                                },
                            )
                        )
        except Exception as e:
            logger.error("Failed to query source database: %s", e)
            raise ValueError(f"Database query failed: {e}") from e
        finally:
            connection.close()

        if not docs:
            raise ValueError("No table data could be extracted from the database.")

        chunks = self.text_splitter.split_documents(docs)
        if not chunks:
            raise ValueError("Database text could not be split into chunks.")

        self.vectorstore.add_documents(chunks)

        document_record, _ = Document.objects.get_or_create(
            filename=virtual_filename,
            defaults={"file_type": "db"},
        )

        return {
            "document_id": document_record.id,
            "chunks_created": len(chunks),
        }

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
