import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


def get_llm(temperature: float = 0.4) -> ChatOpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=temperature)
