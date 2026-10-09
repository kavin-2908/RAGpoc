# Local RAG Backend

A production-quality, modular Retrieval-Augmented Generation (RAG) backend built from scratch to understand the internals of how RAG systems work.

**Built with LangChain, Django, PostgreSQL, and pgvector.**

---

## Architecture Overview

1. **Upload Flow**: 
   - Receives file (PDF, DOCX, TXT, CSV, XLSX).
   - Extracts text using format-specific loaders (`PyMuPDF`, `python-docx`, etc.).
   - Splits text into overlapping 500-character chunks.
   - Generates 384-dimensional embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
   - Stores chunks and embeddings in PostgreSQL using the `pgvector` extension.

2. **Chat Flow**:
   - Receives a user question.
   - Generates an embedding for the question.
   - Performs a cosine similarity search in PostgreSQL via `pgvector` to find the top 5 most relevant chunks.
   - Constructs a strict prompt combining the retrieved context and the user's question.
   - Sends the prompt to a local Gemma model running via **Ollama**.
   - Returns the answer and the source filenames.

---

## Prerequisites

1. **Python 3.10+**
2. **PostgreSQL 15+** with the **pgvector** extension installed.
3. **Ollama** installed on your system.

---

## Installation & Setup

### 1. Database Setup

Install PostgreSQL and the `pgvector` extension. Connect to your database server and run:

```sql
CREATE DATABASE rag_db;
\c rag_db;
CREATE EXTENSION vector;
```

### 2. Ollama Setup

Install Ollama from [ollama.com](https://ollama.com/). Then, pull and run the Gemma model:

```bash
ollama run gemma3:4b
```
Keep this running in the background, or ensure the Ollama service is active on `http://localhost:11434`.

### 3. Python Environment

Clone the repository and set up a virtual environment:

```bash
cd rag_project
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Configuration

Copy the example environment file and update the database credentials if necessary:

```bash
# On Windows (cmd):
copy .env.example .env
# On macOS/Linux/PowerShell:
cp .env.example .env
```

### 5. Run Migrations

Apply the Django migrations to create the `Document` and `Chunk` tables:

```bash
python manage.py makemigrations rag
python manage.py migrate
```
*(Note: The first time you run this, it will download the ~80MB `all-MiniLM-L6-v2` embedding model).*

### 6. Start the Server

```bash
python manage.py runserver
```
The API is now running at `http://localhost:8000`.

---

## API Documentation (Testing with Postman / cURL)

### 1. Upload a Document

**Endpoint:** `POST /api/upload/`  
**Content-Type:** `multipart/form-data`

*cURL Example:*
```bash
curl -X POST http://localhost:8000/api/upload/ \
  -F "file=@/path/to/your/document.pdf"
```

*Success Response (201):*
```json
{
    "message": "Document indexed successfully."
}
```

### 2. Ask a Question

**Endpoint:** `POST /api/chat/`  
**Content-Type:** `application/json`

*cURL Example:*
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

*Success Response (200):*
```json
{
    "answer": "The main topic of the document is...",
    "sources": [
        "document.pdf"
    ]
}
```

### 3. List Documents

**Endpoint:** `GET /api/documents/`

*cURL Example:*
```bash
curl http://localhost:8000/api/documents/
```

### 4. Delete a Document

**Endpoint:** `DELETE /api/documents/{id}/`

Deleting a document automatically deletes all of its associated chunks and embeddings via database CASCADE.

*cURL Example:*
```bash
curl -X DELETE http://localhost:8000/api/documents/1/
```

---

## Code Quality & Design Principles Enforced

- **Separation of Concerns:** Views only handle HTTP. Services handle business logic. Models handle data.
- **Framework:** Powered by LangChain for elegant, chain-based LLM orchestration and efficient document parsing.
- **Robust Database:** Transactions (`@transaction.atomic`) ensure partial uploads are never saved. `bulk_create` ensures fast inserts.
- **Strict Prompting:** The LLM is explicitly instructed to *only* use provided context, reducing hallucinations.
