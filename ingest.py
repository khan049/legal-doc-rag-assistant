import uuid
from pathlib import Path

import fitz
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
)

from config import (
    DATA_FOLDER,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
)


# -------------------------------
# Initialize embedding model
# -------------------------------
embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

EMBEDDING_DIMENSION = embedding_model.get_sentence_embedding_dimension()


# -------------------------------
# Connect Qdrant
# -------------------------------
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


# -------------------------------
# Create Collection
# -------------------------------
try:
    client.delete_collection(COLLECTION_NAME)
    print("Existing collection deleted.")
except Exception:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_DIMENSION,
        distance=Distance.COSINE,
    ),
)

print(f"Created collection: {COLLECTION_NAME}")

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="document",
    field_schema=PayloadSchemaType.KEYWORD,
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="page",
    field_schema=PayloadSchemaType.INTEGER,
)


# -------------------------------
# Text Splitter
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

def extract_pdf(pdf_path: Path):

    document = fitz.open(pdf_path)

    pages = []

    for page_number in range(len(document)):

        page = document.load_page(page_number)

        text = page.get_text().strip()

        if text:

            pages.append(
                {
                    "document": pdf_path.name,
                    "page": page_number + 1,
                    "text": text,
                }
            )

    document.close()

    return pages

def chunk_pages(pages):

    chunks = []

    for page in pages:

        texts = splitter.split_text(page["text"])

        for chunk in texts:

            chunks.append(
                {
                    "document": page["document"],
                    "page": page["page"],
                    "text": chunk,
                }
            )

    return chunks

def embed_chunks(chunks):

    texts = [chunk["text"] for chunk in chunks]

    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return vectors

def upload_chunks(chunks, vectors):
    points = []

    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "document": chunk["document"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                },
            )
        )

    BATCH_SIZE = 100  # Increase/decrease if needed

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )

        print(f"Uploaded {min(i + BATCH_SIZE, len(points))}/{len(points)} chunks")

    print("All chunks uploaded successfully.")
     
def main():

    pdfs = sorted(Path(DATA_FOLDER).glob("*.pdf"))

    print(f"Found {len(pdfs)} PDF(s).\n")

    all_chunks = []

    for pdf in tqdm(pdfs):

        pages = extract_pdf(pdf)

        chunks = chunk_pages(pages)

        all_chunks.extend(chunks)

    print(f"\nGenerated {len(all_chunks)} chunks.")

    vectors = embed_chunks(all_chunks)

    upload_chunks(all_chunks, vectors)

    print("\nIndexing Complete!")


if __name__ == "__main__":
    main()
    
