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
        
        self._vector_service = None
        self._vectorstore = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    @property
    def vector_service(self):
        if self._vector_service is None:
            self._vector_service = VectorService()
        return self._vector_service

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = self.vector_service.get_vectorstore()
        return self._vectorstore


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

    def _get_db_connection(self):
        import pymysql
        try:
            return pymysql.connect(
                host=settings.SOURCE_DB_HOST,
                user=settings.SOURCE_DB_USER,
                password=settings.SOURCE_DB_PASSWORD,
                database=settings.SOURCE_DB_NAME,
                port=int(settings.SOURCE_DB_PORT),
                cursorclass=pymysql.cursors.DictCursor,
            )
        except Exception as e:
            logger.error("Failed to connect to source database %s: %s", settings.SOURCE_DB_NAME, e)
            raise ValueError(f"Database connection failed: {e}") from e

    def get_available_tables(self) -> list[str]:
        """
        Fetch all table names from the configured source MySQL database.
        """
        connection = self._get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                if not tables:
                    return []
                return [list(row.values())[0] for row in tables]
        except Exception as e:
            logger.error("Failed to query tables from source database: %s", e)
            raise ValueError(f"Database query failed: {e}") from e
        finally:
            connection.close()

    def process_database(self, tables: str | list[str] | None = "*") -> dict:
        """
        Extract data from the specified table(s) or all tables in the configured
        MySQL database, format each row into natural NLP descriptions, chunk it,
        embed it, and store it in PostgreSQL vector database.

        Args:
            tables: Table name (str), comma-separated tables (str), list of tables,
                    or '*' / 'all table' / 'all' to ingest all tables.

        Returns:
            dict containing document_id, filename, tables_ingested, chunks_created.
        """
        from langchain_core.documents import Document as LCDocument

        # 1. Fetch available tables in the database
        available_tables = self.get_available_tables()
        if not available_tables:
            raise ValueError(f"No tables found in database '{settings.SOURCE_DB_NAME}'.")

        # 2. Determine target tables based on user input
        is_all = False
        raw_requested: list[str] = []

        if tables is None or tables == "" or tables == "*":
            is_all = True
        elif isinstance(tables, str):
            cleaned = tables.strip().lower()
            if cleaned in {"*", "all", "all table", "all tables"}:
                is_all = True
            elif "," in tables:
                raw_requested = [t.strip() for t in tables.split(",") if t.strip()]
            else:
                raw_requested = [tables.strip()]
        elif isinstance(tables, (list, tuple, set)):
            for item in tables:
                s = str(item).strip()
                if s.lower() in {"*", "all", "all table", "all tables"}:
                    is_all = True
                    break
                if s:
                    raw_requested.append(s)
        else:
            raw_requested = [str(tables).strip()]

        if is_all:
            target_tables = available_tables
        else:
            table_lookup = {t.lower(): t for t in available_tables}
            target_tables = []
            invalid_tables = []

            for req in raw_requested:
                req_lower = req.lower()
                if req_lower in table_lookup:
                    real_table = table_lookup[req_lower]
                    if real_table not in target_tables:
                        target_tables.append(real_table)
                else:
                    invalid_tables.append(req)

            if invalid_tables:
                raise ValueError(
                    f"Table(s) '{', '.join(invalid_tables)}' not found in database '{settings.SOURCE_DB_NAME}'. "
                    f"Available tables: {', '.join(available_tables)}"
                )

            if not target_tables:
                raise ValueError("No valid table names were specified for ingestion.")

        # 3. Determine virtual filename for Document tracking
        if is_all or set(target_tables) == set(available_tables):
            virtual_filename = f"mysql_db_{settings.SOURCE_DB_NAME}"
        elif len(target_tables) == 1:
            virtual_filename = f"mysql_db_{settings.SOURCE_DB_NAME}_{target_tables[0]}"
        else:
            suffix = "_".join(target_tables)
            if len(f"mysql_db_{settings.SOURCE_DB_NAME}_{suffix}") <= 240:
                virtual_filename = f"mysql_db_{settings.SOURCE_DB_NAME}_{suffix}"
            else:
                virtual_filename = f"mysql_db_{settings.SOURCE_DB_NAME}_{target_tables[0]}_and_{len(target_tables)-1}_more"

        logger.info(
            "Connecting to source database %s on %s:%s to ingest table(s): %s (virtual filename: %s)",
            settings.SOURCE_DB_NAME,
            settings.SOURCE_DB_HOST,
            settings.SOURCE_DB_PORT,
            target_tables,
            virtual_filename,
        )

        connection = self._get_db_connection()
        docs: list[LCDocument] = []

        try:
            with connection.cursor() as cursor:
                for table_name in target_tables:
                    logger.info("Ingesting table data from: %s", table_name)
                    cursor.execute(f"SELECT * FROM `{table_name}`")
                    rows = cursor.fetchall()
                    if not rows:
                        logger.warning("Table '%s' contains no rows, skipping.", table_name)
                        continue

                    for row in rows:
                        # Convert database columns into clear, descriptive natural language sentences
                        field_items = [
                            f"{k.replace('_', ' ')} is {v}"
                            for k, v in row.items()
                            if v is not None and str(v).strip() != ""
                        ]
                        if not field_items:
                            continue

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
            raise ValueError(f"No records could be extracted from table(s): {', '.join(target_tables)}.")

        # 4. Split documents into chunks
        chunks = self.text_splitter.split_documents(docs)
        if not chunks:
            raise ValueError("Database text could not be split into chunks.")

        # 5. Clean up any previous ingestion record with this filename to prevent duplicate embeddings
        existing_doc = Document.objects.filter(filename=virtual_filename).first()
        if existing_doc:
            existing_doc.delete()

        # 6. Store chunks in vectorstore
        self.vectorstore.add_documents(chunks)

        # 7. Create Document record
        document_record = Document.objects.create(
            filename=virtual_filename,
            file_type="db",
        )

        return {
            "document_id": document_record.id,
            "filename": virtual_filename,
            "tables_ingested": target_tables,
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
