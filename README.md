# Legal Document RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions using **only the information contained in the supplied PDF documents**. The application retrieves relevant document chunks from Qdrant and uses an OpenRouter free model to generate grounded answers with document-level citations.

## Architecture

```text
                    PDF Documents
                         │
                         ▼
                    PyMuPDF Parser
                         │
                         ▼
                 Text Chunking
                         │
                         ▼
             Sentence Transformers
              BAAI/bge-small-en-v1.5
                         │
                         ▼
                    Qdrant
                Vector Database
                         │
                  User Question
                         │
                         ▼
                 Query Embedding
                         │
                         ▼
              Similarity Retrieval
                         │
                         ▼
                 Relevant Chunks
                         │
                         ▼
                 OpenRouter LLM
                         │
                         ▼
              Answer + Citations
```

Each indexed chunk stores its **document name, page number, and text**, allowing the application to return the retrieved evidence along with every supported answer.

## Technology Stack

* **Python 3.11+**
* **PyMuPDF** — PDF text extraction
* **LangChain Text Splitters** — document chunking
* **Sentence Transformers** — embedding generation
* **Qdrant** — vector database and similarity search
* **OpenRouter** — LLM API
* **OpenAI Python SDK** — OpenRouter API integration
* **python-dotenv** — environment configuration
* **tqdm** — processing progress

## Embedding Model

The application uses:

```text
BAAI/bge-small-en-v1.5
```

Document chunks and user queries are converted into normalized embeddings and searched using **cosine similarity** in Qdrant.

## RAG Pipeline

1. All PDF files in `data/cases_pdf/` are discovered automatically.
2. PDF text is extracted page-by-page using PyMuPDF.
3. Text is split into overlapping chunks.
4. Each chunk is converted into an embedding.
5. Embeddings and metadata are stored in Qdrant.
6. A user's question is converted into an embedding.
7. Qdrant retrieves the most relevant chunks.
8. Retrieved chunks are provided to the OpenRouter model.
9. The model generates an answer using only the retrieved context.
10. The application displays the supporting document name, page number, and retrieved text.

## Grounding & Unknown Questions

The LLM is explicitly instructed not to use outside knowledge or guess.

If the retrieved documents do not contain enough information to answer the question, the application returns:

```text
Information is not available in the supplied documents.
```

This prevents unsupported answers and ensures that responses remain grounded in the supplied PDFs.

## Citations

For supported questions, the application provides:

```text
Document: <document name>
Page: <page number>
Retrieved Text:
<supporting text>
```

The citation text comes directly from the retrieved PDF chunk used as context for the answer.

## Assumptions

* All supplied PDFs are placed in `data/cases_pdf/`.
* Every `.pdf` file in that directory is processed automatically.
* Qdrant runs locally using Docker.
* An OpenRouter API key is required.
* A free OpenRouter model is used.
* The application should answer only from the supplied documents.

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_free_model

QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=legal_cases
```

### 3. Start Qdrant

```bash
docker compose up -d
```

### 4. Index the PDFs

```bash
python ingest.py
```

The ingestion script automatically processes all PDFs inside:

```text
data/cases_pdf/
```

### 5. Run the application

```bash
python query.py
```

Example:

```text
Question: What does Article 21 state?

Answer:
Article 21 states that no person shall be deprived of his
life or personal liberty except according to procedure
established by law.

Sources:
Document: ...
Page: ...
Retrieved Text: ...
```

Type `exit` to close the application.
