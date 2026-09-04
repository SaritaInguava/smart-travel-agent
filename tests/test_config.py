import pytest

from smart_travel_agent.config import get_expedia_rapidapi_key, get_llm


def test_get_llm_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm()


def test_get_expedia_rapidapi_key_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("EXPEDIA_RAPIDAPI_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EXPEDIA_RAPIDAPI_KEY"):
        get_expedia_rapidapi_key()
