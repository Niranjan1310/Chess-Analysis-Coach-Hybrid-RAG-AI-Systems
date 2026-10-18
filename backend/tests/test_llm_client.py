"""Tests for app.rag.llm_client."""

import pytest
from app.rag.llm_client import LLMClient, FakeLLMClient, GeminiClient


def test_fake_llm_client_default():
    fake = FakeLLMClient()
    response = fake.generate(system="sys prompt", user="user prompt")
    assert fake.call_count == 1
    assert fake.last_system == "sys prompt"
    assert fake.last_user == "user prompt"
    assert "played move was a mistake" in response


def test_fake_llm_client_custom_response():
    fake = FakeLLMClient(response="Custom explanation text.")
    response = fake.generate(system="sys", user="user")
    assert response == "Custom explanation text."
    assert fake.call_count == 1


def test_llm_client_initialization_defaults(monkeypatch):
    monkeypatch.setattr("app.config.settings.gemini_api_key", "test-key-123")
    monkeypatch.setattr("app.config.settings.gemini_model", "test-model-custom")
    client = LLMClient()
    assert client._api_key == "test-key-123"
    assert client._model == "test-model-custom"


def test_llm_client_initialization_override():
    client = LLMClient(api_key="override-key", model="gemini-test")
    assert client._api_key == "override-key"
    assert client._model == "gemini-test"


def test_gemini_client_error_handling(monkeypatch):
    client = GeminiClient(api_key="bad-key")
    # Calling generate without valid credentials / mocking google.generativeai failure
    with pytest.raises(RuntimeError, match="LLM generation failed"):
        client.generate(system="sys", user="user")
