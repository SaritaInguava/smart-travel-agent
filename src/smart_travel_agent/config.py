import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")


def get_llm(temperature: float = 0.4, model: str = DEFAULT_MODEL) -> ChatOpenAI:
    _require_api_key()
    return ChatOpenAI(model=model, temperature=temperature)


def get_embeddings() -> OpenAIEmbeddings:
    _require_api_key()
    return OpenAIEmbeddings()


def get_expedia_rapidapi_key() -> str:
    api_key = os.environ.get("EXPEDIA_RAPIDAPI_KEY")
    if not api_key:
        raise RuntimeError("EXPEDIA_RAPIDAPI_KEY is not set. Copy .env.example to .env and fill it in.")
    return api_key
