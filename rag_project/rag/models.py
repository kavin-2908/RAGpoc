import logging
import os
from django.conf import settings
from django.db import connection, models
from django.db.models.signals import pre_delete
from django.dispatch import receiver

logger = logging.getLogger(__name__)


class Document(models.Model):
    """
    Represents an uploaded document.

    Stores metadata only. The actual text content and vectors
    will be managed by LangChain's PGVector.
    """

    filename = models.CharField(
        max_length=255,
        help_text="Original filename of the uploaded document.",
    )
    file_type = models.CharField(
        max_length=10,
        help_text="File extension without the dot (e.g., 'pdf', 'docx').",
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the document was uploaded.",
    )

    class Meta:
        ordering = ['-uploaded_at']
        db_table = 'documents'

    def __str__(self) -> str:
        return f"{self.filename} ({self.file_type})"


@receiver(pre_delete, sender=Document)
def cleanup_document_resources(sender, instance, **kwargs):
    """
    Clean up resources associated with a Document prior to deletion:
    1. Legacy 'chunks' table records (to prevent foreign key integrity violations).
    2. Vector embeddings in PGVector matching this document's filename.
    3. The uploaded physical file on disk.
    """
    try:
        with connection.cursor() as cursor:
            # 1. Clean up legacy 'chunks' table if it exists
            cursor.execute("""
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'chunks';
            """)
            if cursor.fetchone():
                cursor.execute("DELETE FROM chunks WHERE document_id = %s;", [instance.pk])
                logger.debug("Purged legacy chunks for document id %s", instance.pk)

            # 2. Clean up vectorstore embeddings if tables exist
            for table_name in ['langchain_pg_embedding', 'rag_embeddings']:
                cursor.execute("""
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s;
                """, [table_name])
                if cursor.fetchone():
                    cursor.execute(
                        f"DELETE FROM {table_name} WHERE cmetadata->>'filename' = %s;",
                        [instance.filename]
                    )
                    logger.debug("Purged embeddings from %s for filename %s", table_name, instance.filename)
    except Exception as e:
        logger.warning("Error cleaning up database resources for document %s: %s", instance.pk, e)

    # 3. Clean up physical file on disk
    if instance.filename:
        file_path = settings.UPLOAD_DIR / instance.filename
        if file_path.exists():
            try:
                os.remove(file_path)
                logger.info("Deleted file from disk: %s", file_path)
            except OSError as e:
                logger.warning("Could not delete file %s from disk: %s", file_path, e)
