"""
Thin LLM client wrapper around the Gemini API.

Provides a thin ``LLMClient`` wrapper around Gemini.
The API key is read from settings (``GEMINI_API_KEY`` env var).

For tests, inject a ``FakeLLMClient`` instead.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol that any LLM client must satisfy."""

    def generate(self, system: str, user: str) -> str:
        """Generate a response given system and user prompts."""
        ...


class GeminiClient:
    """Real Gemini API client.

    Args:
        api_key: Gemini API key.
        model: Model name (default: gemini-2.0-flash).
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, system: str, user: str) -> str:
        """Call the Gemini API and return the text response.

        Raises:
            RuntimeError: if the API call fails.
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel(
                self._model,
                system_instruction=system,
            )
            response = model.generate_content(user)
            return response.text
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            raise RuntimeError(f"LLM generation failed: {e}") from e


class LLMClient(GeminiClient):
    """Thin wrapper around GeminiClient that reads credentials from settings (.env)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key if api_key is not None else settings.gemini_api_key,
            model=model if model is not None else settings.gemini_model,
        )


class FakeLLMClient:
    """Deterministic fake client for tests.

    Returns a canned response that includes the played move and best
    move so that move-faithfulness checks can be tested.
    """

    def __init__(self, response: str | None = None) -> None:
        self._response = response
        self.last_system: str = ""
        self.last_user: str = ""
        self.call_count: int = 0

    def generate(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        self.call_count += 1
        if self._response is not None:
            return self._response
        # Default: echo back the played move and best move from the prompt
        return (
            "The played move was a mistake because it loses material. "
            "The best move maintains the advantage. "
            "Concept to study: tactical awareness. "
            "Sources: [source-1]"
        )
