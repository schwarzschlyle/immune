import os

from openai import OpenAI

import immune

immune.init()

providers = {
    "ollama": OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    "gemini": OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ.get("GEMINI_API_KEY", ""),
    ),
}

for name, client in providers.items():
    model = "llama3.2" if name == "ollama" else "gemini-3-flash"
    reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say hello."}])
    print(name, reply.choices[0].message.content, immune.verdict(reply).action.value)
