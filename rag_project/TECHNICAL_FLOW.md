# RAG Project: Technical Flow & Architecture Document

This document provides a comprehensive technical overview of the Retrieval-Augmented Generation (RAG) backend. It breaks down the workflow into phases, explains how RAG is implemented here, and details the specific purpose of every file in the project.

---

## 1. How RAG Works in This Project

Retrieval-Augmented Generation (RAG) is a technique that gives a Large Language Model (LLM) access to external knowledge (your documents) to answer questions accurately and reduce hallucinations. 

In this project, it is broken down into two main phases:
1. **The Ingestion Phase (Upload):** Converting your documents into numerical representations (embeddings) and storing them.
2. **The Retrieval & Generation Phase (Chat):** Searching those stored representations for relevance to a user's question, and feeding that specific information to the LLM to generate an answer.

We enforce strict separation of concerns: **Views** handle HTTP requests, **Services** handle the core logic (embedding, LLM interaction, database searching), and LangChain handles the document ingestion and chains.

---

## 2. Phase-by-Phase Execution Flow

### Phase 1: Ingestion Flow (Document Upload)
When a user uploads a document to `/api/upload/`:

1. **API Endpoint (`views.py`):** Receives the HTTP POST request with the file. It passes the file to the `DocumentService`.
2. **File Parsing (`loaders/`):** Based on the file extension (e.g., PDF, CSV, DOCX), the appropriate loader reads the file and extracts raw text.
3. **Text Splitting (`utils/text_utils.py`):** The extracted text is too large for an LLM to process at once. It is split into smaller, overlapping chunks (e.g., 500 characters). Overlap ensures context isn't lost between chunk boundaries.
4. **Embedding Generation (`services/embedding_service.py`):** Each text chunk is passed to a local embedding model (`all-MiniLM-L6-v2` via `sentence-transformers`). This model converts the text chunk into a 384-dimensional vector (an array of floating-point numbers) representing its semantic meaning.
5. **Database Storage (`models.py` & `services/document_service.py`):** The original document metadata, the text chunks, and their corresponding vector embeddings are saved to PostgreSQL. The embeddings are stored using the `pgvector` extension.

### Phase 2: Retrieval & Generation Flow (Chat)
When a user asks a question via `/api/chat/`:

1. **API Endpoint (`views.py`):** Receives the user's question in JSON format and passes it to the `RagService`.
2. **Question Embedding (`services/embedding_service.py`):** The user's question is converted into a vector embedding using the *exact same* model used during ingestion.
3. **Vector Search (`services/vector_service.py`):** The system performs a "Cosine Similarity" search in PostgreSQL using `pgvector`. It compares the question's vector against all stored chunk vectors to find the most mathematically similar (semantically relevant) chunks. It retrieves the top `K` chunks (e.g., top 5).
4. **Prompt Construction (`services/rag_service.py`):** The retrieved chunks are stitched together into a strict prompt text. The prompt explicitly tells the LLM: *"Use the following context to answer the question. Do not use outside knowledge."*
5. **LLM Generation (`services/llm_service.py`):** The constructed prompt is sent to the local Ollama instance running the `gemma` model.
6. **Response (`views.py`):** The generated answer, along with the source filenames used, is returned to the user via the API.

---

## 3. Directory & File Breakdown

### Root Level
- **`manage.py`**: The standard Django command-line utility used to run the server, apply migrations, etc.
- **`requirements.txt`**: Lists all Python dependencies required to run the project.
- **`.env.example`**: Template for environment variables (like database credentials).
- **`README.md`**: The setup guide and general overview.

### `/config/` Directory
This is the core Django configuration folder.
- **`settings.py`**: Contains all Django settings, including database connections, installed apps, and REST framework configurations.
- **`urls.py`**: The main URL router for the entire project. It routes `/api/` traffic to the `rag` app.
- **`wsgi.py`**: The Web Server Gateway Interface entry point, used for deploying the application.

### `/rag/` Directory
This is the primary application containing all the RAG logic.

#### Core Django Files
- **`models.py`**: Defines the database schema. Contains `Document` (stores file metadata) and `Chunk` (stores the chunk text and its `pgvector` embedding field).
- **`views.py`**: Defines the API endpoints (`UploadView`, `ChatView`, `DocumentListView`). They validate incoming requests and return HTTP responses.
- **`serializers.py`**: Defines how Django converts complex data types (like QuerySets) to JSON and vice-versa.
- **`urls.py`**: Maps specific URL paths (like `/upload/` and `/chat/`) to the views defined in `views.py`.

#### `/rag/services/` (Business Logic)
This is where the heavy lifting happens.
- **`document_service.py`**: Orchestrates the entire Phase 1 (Ingestion) flow. It coordinates loaders, splitters, embedders, and database saving.
- **`embedding_service.py`**: Handles loading the `sentence-transformers` model and converting text strings into vectors.
- **`vector_service.py`**: Handles the database querying logic to perform cosine similarity searches using `pgvector`.
- **`llm_service.py`**: Handles communication with the local Ollama API to send prompts to the Gemma model and receive responses.
- **`rag_service.py`**: Orchestrates the entire Phase 2 (Chat) flow. Coordinates embedding the question, searching vectors, and calling the LLM service.

#### `/rag/loaders/` (File Parsing)
Responsible for extracting raw text from various file formats.
- **`pdf_loader.py`**: Uses libraries like `PyMuPDF` to extract text from PDFs.
- **`docx_loader.py`**: Extracts text from Word documents.
- **`csv_loader.py` / `excel_loader.py`**: Extracts tabular data into a readable text format.
- **`text_loader.py`**: Reads standard `.txt` files.

#### `/rag/utils/` (Helpers)
- **`text_utils.py`**: Contains the logic for splitting large texts into overlapping chunks safely (ensuring sentences or words aren't cut in half inappropriately).
