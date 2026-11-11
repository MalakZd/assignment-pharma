import os
from dotenv import load_dotenv, dotenv_values

print("Variable names in .env:", list(dotenv_values(".env").keys()))

load_dotenv(override=True)  # .env wins over system variables

from urllib.parse import urlparse
base = os.getenv("OPENAI_BASE_URL")
print("OPENAI_BASE_URL host:", urlparse(base).netloc if base else "NOT SET")
print("Chat model:", os.getenv("OPENAI_MODEL"))
print("Embedding model:", os.getenv("OPENAI_EMBEDDING_MODEL"))

from openai import OpenAI
client = OpenAI()

try:
    print("Models:", [m.id for m in client.models.list().data])
except Exception as e:
    print("models.list failed (some proxies block it):", type(e).__name__)

emb = client.embeddings.create(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    input=["warfarin"],
)
print("Embedding dims:", len(emb.data[0].embedding))

chat = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL"),
    messages=[{"role": "user", "content": "In one sentence, what is a contraindication?"}],
)
print(chat.choices[0].message.content)