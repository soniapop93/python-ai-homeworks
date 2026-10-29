import os
import json
from pathlib import Path
import chromadb
import pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data_sources"
DB_DIR = PROJECT_DIR / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 800
SEARCH_RESULT_COUNT = 8
CHUNK_OVERLAP = 120

class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(
        description="Exact filename in data_sources, for example StarMaker.pdf"
    )
    question: str = Field(
        description="Question about the document contents"
    )


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

    if extension in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8")

        return [{
            "metadata": {"page_number": 1},
            "text": text
        }]

    return []


def index_file(collection, model, file_path: Path):
    pages = extract_pages(file_path)
    records = []

    for page in pages:
        page_number = page["metadata"]["page_number"]
        page_text = page["text"]

        for chunk_number, chunk in enumerate(chunk_text(page_text)):
            records.append((chunk, page_number, chunk_number))

    collection.delete(where={"source": file_path.name})

    if not records:
        print(f"No text found in {file_path.name}.")
        return

    ids = []
    documents = []
    metadata = []

    for chunk, page_number, chunk_number in records:
        ids.append(
            f"{file_path.name}:p{page_number}:c{chunk_number}"
        )

        documents.append(chunk)

        metadata.append({
            "source": file_path.name,
            "page": page_number,
            "chunk": chunk_number,
        })

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

    supported_extensions = {".pdf", ".txt", ".md"}

    files = sorted(
        file_path
        for file_path in DATA_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in supported_extensions
    )

    if not files:
        print("No PDF, TXT, or MD files found in data_sources.")
        return

    for file_path in files:
        try:
            index_file(collection, model, file_path)
        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")


def search_document(collection, model, args: str) -> str:
    try:
        request = SearchRequest.model_validate(json.loads(args))
    except (json.JSONDecodeError, ValidationError) as error:
        return f"Invalid request: {error}"

    filename = request.filename.strip()

    if not filename or filename != Path(filename).name:
        return (
            "Invalid filename. Provide only the filename, "
            "for example, StarMaker.pdf."
        )

    existing = collection.get(
        where={"source": filename},
        limit=1,
        include=["metadatas"]
    )

    if not existing["ids"]:
        available_files = sorted({
            metadata["source"]
            for metadata in collection.get(
                include=["metadatas"]
            )["metadatas"]
            if metadata and "source" in metadata
        })

        if available_files:
            return (
                f"File '{filename}' was not found.\n"
                f"Available files: {', '.join(available_files)}"
            )

        return "No indexed files are available."

    query_embedding = model.encode(
        request.question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=SEARCH_RESULT_COUNT,
        where={"source": filename},
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return "No relevant passages found in the selected file."

    context_parts = []

    print("\nRetrieved passages:")

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        source = metadata["source"]
        page = metadata["page"]

        print(
            f"File: {source} | "
            f"Page: {page} | "
            f"Distance: {distance:.4f}"
        )

        context_parts.append(
            f"{document}\nSource: {source}, page {page}"
        )

    return "\n\n---\n\n".join(context_parts)


def complete(client: OpenAI, messages: list, tools: list):
    return client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    ).choices[0].message


def main():
    load_dotenv(BASE_DIR.parent / ".env", override=True)

    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    model_name = os.getenv("OPEN_ROUTER_MODEL_NAME")

    if not api_key or not model_name:
        print(
            "Configure OPEN_ROUTER_API_KEY and "
            "OPEN_ROUTER_MODEL_NAME in the .env file."
        )
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print("Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    db = chromadb.PersistentClient(path=str(DB_DIR))

    collection = db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("\n=== DOCUMENT ===")
    index_folder(collection, model)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_document",
                "description": (
                    "Search a specific file in data_sources. "
                    "Always use this tool for document questions. "
                    "Provide the exact filename and question. "
                    "Results include source filenames and page numbers."
                ),
                "parameters": SearchRequest.model_json_schema(),
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a RAG agent. Answer in English. "
                "The user must provide a filename and a question. "
                "Ask for the filename if it is missing. "
                "Always use search_document for document questions. "
                "Use only the retrieved passages. Never invent facts. "
                "Cite filenames and page numbers. "
                "If the answer is unavailable, state this clearly."
            ),
        }
    ]

    print("\n=== RAG SEARCH ===")
    print("Example: StarMaker.pdf, What is this document about?")
    print("Type 'quit' or 'exit' to quit.")

    while True:
        user_input = input("\nyou> ").strip()

        if user_input.lower() in {"quit", "exit"}:
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input,
        })

        try:
            response = complete(client, messages, tools)

            messages.append(
                response.model_dump(exclude_none=True)
            )

            while response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.name == "search_document":
                        output = search_document(
                            collection,
                            model,
                            tool_call.function.arguments,
                        )
                    else:
                        output = f"Unknown tool: {tool_call.function.name}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    })

                response = complete(client, messages, tools)

                messages.append(
                    response.model_dump(exclude_none=True)
                )

            print("\nAgent:", response.content or "")

        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()