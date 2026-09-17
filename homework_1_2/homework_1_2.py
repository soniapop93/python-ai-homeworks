import os
import subprocess

from dotenv import load_dotenv
from openai import OpenAI


def get_files():
    result = subprocess.run(
        ["cmd", "/c", "dir", "/s", "/b", "/a-d", ".."],
        capture_output=True,
        text=True
    )
    return result.stdout


def find_text(word):
    result = subprocess.run(
        ["cmd", "/c", "findstr", "/s", "/i", "/m", word, "*.*"],
        capture_output=True,
        text=True
    )
    return result.stdout


def main():
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY")
    )

    messages = [
        {
            "role": "system",
            "content": """Cauta in repository folder, dar intai listeaza fisierele, apoi cauta keyword-ul pytest iar la final afiseaza fisierele in care apare"""
        },
        {
            "role": "user",
            "content": "Cauta pytest in proiect"
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "Listeaza fisierele din repository folder",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_keyword",
                "description": "Cauta un keyword in fisiere",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Keyword-ul cautat"
                        }
                    },
                    "required": ["keyword"]
                }
            }
        }
    ]

    files = get_files()

    messages.append({
        "role": "user",
        "content": files
    })

    response = client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto"
    ).choices[0].message

    while response.tool_calls:
        messages.append(response)

        for call in response.tool_calls:
            name = call.function.name

            if name == "list_files":
                result = files
            elif name == "search_keyword":
                result = find_text("pytest")
            else:
                result = "Tool not found"

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result
            })

        response = client.chat.completions.create(
            messages=messages,
            model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
            tools=tools,
            tool_choice="auto"
        ).choices[0].message

    return response.content


if __name__ == "__main__":
    main()