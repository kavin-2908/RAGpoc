from django.db import models

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
