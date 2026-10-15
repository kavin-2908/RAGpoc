import logging

from django.conf import settings
from django.db import transaction
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from rag.models import Document
from rag.serializers import (
    ChatSerializer,
    DBIngestSerializer,
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
    Ingests specific tables or all tables from the source MySQL database.
    Payload examples:
      - {"tables": "student"}
      - {"tables": ["student", "fees"]}
      - {"tables": "*"} or {"tables": "all table"}

    GET /api/db-ingest/
    Lists available tables from the source database.
    """

    def get(self, request):
        try:
            document_service = DocumentService()
            tables = document_service.get_available_tables()
            return Response(
                {
                    "database": settings.SOURCE_DB_NAME,
                    "tables": tables,
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            logger.exception("Failed to fetch available tables: %s", e)
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def post(self, request):
        serializer = DBIngestSerializer(data=request.data if request.data else {})
        serializer.is_valid(raise_exception=True)

        # Extract table input from validated data or fallback to query params
        validated = serializer.validated_data
        tables_input = (
            validated.get('tables')
            or validated.get('table')
            or validated.get('table_name')
            or request.query_params.get('tables')
            or request.query_params.get('table')
            or request.query_params.get('table_name')
            or '*'
        )

        try:
            document_service = DocumentService()
            result = document_service.process_database(tables=tables_input)
            table_names_str = ", ".join(result["tables_ingested"])
            return Response(
                {
                    "message": f"Successfully ingested {len(result['tables_ingested'])} table(s): {table_names_str}.",
                    "document_id": result["document_id"],
                    "filename": result["filename"],
                    "tables_ingested": result["tables_ingested"],
                    "chunks_created": result["chunks_created"],
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

    Deletes a document and all associated resources (chunks, embeddings, files).
    """

    @transaction.atomic
    def delete(self, request, pk):
        try:
            document = Document.objects.get(pk=pk)
        except Document.DoesNotExist:
            return Response(
                {"error": f"Document with id {pk} not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        filename = document.filename
        try:
            document.delete()
        except Exception as e:
            logger.exception("Failed to delete document id %s: %s", pk, e)
            return Response(
                {"error": f"Failed to delete document: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        logger.info("Deleted document: %s (id=%s)", filename, pk)
        return Response(
            {"message": f"Document '{filename}' deleted successfully."},
            status=status.HTTP_200_OK,
        )
