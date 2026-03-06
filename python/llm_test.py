"""
llm_test.py — Run LLaMA 3 locally via Ollama REST API.

Ollama must be running (it is — confirmed on port 11434).
Model: llama3:latest
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3:latest"

def chat(message: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


if __name__ == "__main__":
    prompt = "In two sentences, explain what cognitive behavioral therapy is."
    print(f"Prompt: {prompt}\n")
    print("Response:")
    print(chat(prompt))
