from rest_framework import serializers

from rag.models import Document


# ---------------------------------------------------------------------------
# Allowed file extensions for upload
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv', 'xlsx'}


class UploadSerializer(serializers.Serializer):
    """
    Validates file uploads.

    Checks that:
    - A file was actually provided.
    - The file extension is one of: pdf, docx, txt, csv, xlsx.
    """

    file = serializers.FileField(
        help_text="The document file to upload.",
    )

    def validate_file(self, file):
        """Validate the uploaded file's extension."""
        # Extract extension from filename (e.g., 'report.pdf' → 'pdf')
        filename = file.name
        if '.' not in filename:
            raise serializers.ValidationError(
                "File must have an extension (e.g., .pdf, .docx)."
            )

        extension = filename.rsplit('.', 1)[-1].lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise serializers.ValidationError(
                f"Unsupported file type: '.{extension}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )

        # Check file is not empty
        if file.size == 0:
            raise serializers.ValidationError("Uploaded file is empty.")

        return file


class ChatSerializer(serializers.Serializer):
    """
    Validates chat/question input.

    Checks that the question is a non-empty string.
    """

    question = serializers.CharField(
        max_length=2000,
        help_text="The question to ask about the uploaded documents.",
    )

    def validate_question(self, value: str) -> str:
        """Ensure the question is not just whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise serializers.ValidationError(
                "Question cannot be empty or whitespace only."
            )
        return cleaned


class DocumentSerializer(serializers.ModelSerializer):
    """
    Serializes Document model instances for the list API.

    Returns id, filename, file_type, and uploaded_at.
    """

    class Meta:
        model = Document
        fields = ['id', 'filename', 'file_type', 'uploaded_at']
        read_only_fields = fields
