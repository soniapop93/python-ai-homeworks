import chromadb
import pymupdf4llm
from pathlib import Path
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "final_data_sources"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def extract_pages(file_path: Path) -> list[dict]:
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return pymupdf4llm.to_markdown(
            str(file_path),
            page_chunks=True
        )

    if extension == ".txt":
        text = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        return [{
            "metadata": {"page_number": 1},
            "text": text
        }]

    return []


def index_file(collection, model, file_path: Path):
    pages = extract_pages(file_path)
    ids = []
    documents = []
    metadata = []

    collection.delete(where={"source": file_path.name})

    for page in pages:
        page_number = page["metadata"].get("page_number", 1)
        page_text = page["text"]

        for chunk_number, chunk in enumerate(
            chunk_text(page_text), start=1
        ):
            ids.append(
                f"{file_path.name}:p{page_number}:c{chunk_number}"
            )

            documents.append(chunk)

            metadata.append({
                "source": file_path.name,
                "page": page_number,
                "chunk": chunk_number,
            })

    if not documents:
        print(f"No text found in {file_path.name}.")
        return

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadata,
        embeddings=embeddings,
    )

    print(f"{file_path.name}: {len(documents)} chunks indexed.")


def index_folder(collection, model):
    if not DATA_DIR.is_dir():
        print(f"Directory not found: {DATA_DIR}")
        return

    supported_extensions = {".pdf", ".txt"}

    files = sorted(
        file_path
        for file_path in DATA_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in supported_extensions
    )

    if not files:
        print("No PDF or TXT files found in final_data_sources.")
        return

    for file_path in files:
        try:
            index_file(collection, model, file_path)
        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")


def main():
    print("Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    try:
        client = chromadb.HttpClient(
            host="localhost",
            port=8000
        )

        client.heartbeat()

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    except Exception as error:
        print(f"Could not connect to ChromaDB: {error}")
        print(
            "Start the ChromaDB server with: "
            "uv run chroma run --path chroma_db --port 8000"
        )
        return

    print("\n=== DOCUMENT INDEXING ===")
    index_folder(collection, model)

    print(f"\nTotal indexed chunks: {collection.count()}")


if __name__ == "__main__":
    main()
