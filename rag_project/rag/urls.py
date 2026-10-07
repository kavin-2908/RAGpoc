from django.urls import path

from rag.views import (
    ChatView,
    DocumentDeleteView,
    DocumentListView,
    UploadView,
)

urlpatterns = [
    # Upload a document for indexing
    path('upload/', UploadView.as_view(), name='upload'),

    # Ask a question (RAG)
    path('chat/', ChatView.as_view(), name='chat'),

    # List all documents
    path('documents/', DocumentListView.as_view(), name='document-list'),

    # Delete a specific document
    path('documents/<int:pk>/', DocumentDeleteView.as_view(), name='document-delete'),
]
