import os
import json
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"

SEARCH_RESULT_COUNT = 10


class UserQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        description="Question asked by the user"
    )


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        description="Question to search for in the indexed documents"
    )


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: str = Field(
        description="Text context retrieved from ChromaDB"
    )


def search_document(collection, model, args: str) -> str:
    try:
        request = SearchRequest.model_validate(json.loads(args))
    except (json.JSONDecodeError, ValidationError) as error:
        return f"Invalid request: {error}"

    question = request.question.strip()

    if not question:
        return "Invalid request: question cannot be empty."

    if collection.count() == 0:
        return "No indexed documents are available."

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(SEARCH_RESULT_COUNT, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []

    print("\nRetrieved passages:")

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        source = metadata["source"]
        page = metadata["page"]
        chunk = metadata["chunk"]

        print(
            f"File: {source} | "
            f"Page: {page} | "
            f"Chunk: {chunk} | "
            f"Distance: {distance:.4f}"
        )

        context_parts.append(
            f"{document}\n"
            f"Source: {source}, page {page}, chunk {chunk}"
        )

    context = "\n\n---\n\n".join(context_parts)

    try:
        response = SearchResponse.model_validate({
            "context": context
        })
        return response.context
    except ValidationError as error:
        return f"Invalid search response: {error}"


def complete(client: OpenAI, messages: list, tools: list):
    return client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    ).choices[0].message


def main():
    load_dotenv()

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

    try:
        db = chromadb.HttpClient(
            host="localhost",
            port=8000
        )

        db.heartbeat()

        collection = db.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as error:
        print(f"Could not connect to ChromaDB: {error}")
        print(
            "Start the ChromaDB server with: "
            "uv run chroma run --path chroma_db --port 8000"
        )
        print("Run index_documents.py before starting the agent.")
        return

    tools = [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": (
                    "Search the indexed documents using semantic search. "
                    "Always use this tool to answer questions about "
                    "the user's documents. The tool returns relevant "
                    "passages with source filenames and page numbers."
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
                "Always use rag_search to answer questions about "
                "the indexed documents. Use only the retrieved context. "
                "Do not invent facts or use your own knowledge to answer "
                "document questions. Cite the source filename and page "
                "number when possible. If the context does not contain "
                "the answer, clearly state that the information was not "
                "found in the indexed documents."
            ),
        }
    ]

    print("\n=== RAG AGENT ===")
    print("Ask a question about the indexed documents.")
    print("Type 'quit' or 'exit' to quit.")

    while True:
        user_input = input("\nyou> ").strip()

        if user_input.lower() in {"quit", "exit"}:
            break

        try:
            request = UserQuestion.model_validate({
                "question": user_input
            })
        except ValidationError as error:
            print(f"Invalid question: {error}")
            continue

        messages.append({
            "role": "user",
            "content": request.question,
        })

        try:
            response = complete(client, messages, tools)

            messages.append(
                response.model_dump(exclude_none=True)
            )

            while response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.name == "rag_search":
                        output = search_document(
                            collection,
                            model,
                            tool_call.function.arguments,
                        )
                    else:
                        output = (
                            f"Unknown tool: "
                            f"{tool_call.function.name}"
                        )

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
