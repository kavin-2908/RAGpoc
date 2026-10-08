import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from rag.models import Document
from rag.serializers import (
    ChatSerializer,
    DocumentSerializer,
    UploadSerializer,
)
from rag.services.document_service import DocumentService
from rag.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class UploadView(APIView):
    """
    POST /api/upload/

    Accepts a document file, processes it through the RAG pipeline:
    extract text → chunk → embed → store in PostgreSQL.
    """

    def post(self, request):
        serializer = UploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uploaded_file = serializer.validated_data['file']

        try:
            document_service = DocumentService()
            document_service.process_document(
                file=uploaded_file,
                filename=uploaded_file.name,
            )
            return Response(
                {"message": "Document indexed successfully."},
                status=status.HTTP_201_CREATED,
            )

        except ValueError as e:
            # Known errors: empty text, unsupported format, etc.
            logger.warning("Upload failed (ValueError): %s", e)
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except ConnectionError as e:
            # Embedding model or database connection issues
            logger.error("Upload failed (ConnectionError): %s", e)
            return Response(
                {"error": "Service temporarily unavailable. Please try again later."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            logger.exception("Upload failed (unexpected): %s", e)
            return Response(
                {"error": "An unexpected error occurred while processing the document."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

class DBIngestView(APIView):
    """
    POST /api/db-ingest/

    Connects to the source MySQL database, extracts all table data,
    processes it through the RAG pipeline, and stores it in PostgreSQL.
    """

    def post(self, request):
        try:
            document_service = DocumentService()
            result = document_service.process_database()
            return Response(
                {
                    "message": "Database ingested successfully.",
                    "document_id": result["document_id"],
                    "chunks_created": result["chunks_created"]
                },
                status=status.HTTP_201_CREATED,
            )

        except ValueError as e:
            logger.warning("DB ingestion failed (ValueError): %s", e)
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.exception("DB ingestion failed (unexpected): %s", e)
            return Response(
                {"error": "An unexpected error occurred while ingesting the database."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ChatView(APIView):
    """
    POST /api/chat/

    Accepts a question and returns an answer generated from
    relevant document chunks using RAG.
    """

    def post(self, request):
        serializer = ChatSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        question = serializer.validated_data['question']

        try:
            rag_service = RAGService()
            result = rag_service.answer_question(question)
            return Response(result, status=status.HTTP_200_OK)

        except ConnectionError as e:
            logger.error("Chat failed (ConnectionError): %s", e)
            return Response(
                {"error": "AI Service temporarily unavailable. Please try again later."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            logger.exception("Chat failed (unexpected): %s", e)
            return Response(
                {"error": "An unexpected error occurred while processing the question."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class DocumentListView(APIView):
    """
    GET /api/documents/

    Returns a list of all uploaded documents.
    """

    def get(self, request):
        documents = Document.objects.all()
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class DocumentDeleteView(APIView):
    """
    DELETE /api/documents/<id>/

    Deletes a document and all of its chunks (via CASCADE).
    """

    def delete(self, request, pk):
        try:
            document = Document.objects.get(pk=pk)
        except Document.DoesNotExist:
            return Response(
                {"error": f"Document with id {pk} not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        filename = document.filename
        document.delete()

        logger.info("Deleted document: %s (id=%s)", filename, pk)
        return Response(
            {"message": f"Document '{filename}' deleted successfully."},
            status=status.HTTP_200_OK,
        )
