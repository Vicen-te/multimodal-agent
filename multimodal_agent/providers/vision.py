"""Vision-language model: local Ollama VL, Google Gemini, or an OpenAI-compatible API."""

from __future__ import annotations

import base64
import binascii

from .._content import content_to_text
from ..config import Settings


class OllamaVisionModel:
    """Describe or answer questions about an image using an Ollama VL model."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self._host = host
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self._host)
        return self._client

    def describe(self, image_base64: str, query: str) -> str:
        """Run the VL model on a base64-encoded image with a textual query."""
        client = self._ensure_client()
        prompt = query or "Describe this image in detail."
        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image_base64]}],
        )
        # The ollama client returns a dict on older versions and a ChatResponse
        # object on newer ones; both expose message.content.
        message = response["message"] if isinstance(response, dict) else response.message
        return message["content"] if isinstance(message, dict) else message.content


def _image_mime(image_base64: str) -> str:
    """Sniff the image mime type from its leading bytes, defaulting to PNG."""
    try:
        header = base64.b64decode(image_base64[:32])
    except (binascii.Error, ValueError):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"GIF8"):
        return "image/gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


class GeminiVisionModel:
    """Describe an image using a multimodal Google Gemini model."""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._client = ChatGoogleGenerativeAI(
                model=self.model, google_api_key=self._api_key, temperature=0
            )
        return self._client

    def describe(self, image_base64: str, query: str) -> str:
        """Run the multimodal model on a base64-encoded image with a query."""
        from langchain_core.messages import HumanMessage

        client = self._ensure_client()
        prompt = query or "Describe this image in detail."
        data_url = f"data:{_image_mime(image_base64)};base64,{image_base64}"
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": data_url},
            ]
        )
        return content_to_text(client.invoke([message]).content)


class OpenAIVisionModel:
    """Describe an image through a vision-capable model on an OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.model, api_key=self._api_key, base_url=self._base_url, temperature=0
            )
        return self._client

    def describe(self, image_base64: str, query: str) -> str:
        """Run the model on a base64-encoded image with a query."""
        from langchain_core.messages import HumanMessage

        client = self._ensure_client()
        prompt = query or "Describe this image in detail."
        data_url = f"data:{_image_mime(image_base64)};base64,{image_base64}"
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        return content_to_text(client.invoke([message]).content)


def build_vision_model(settings: Settings):
    """Return the vision backend selected by ``llm_provider``."""
    if settings.llm_provider == "openai":
        return OpenAIVisionModel(
            settings.openai_model, settings.openai_api_key, settings.openai_base_url
        )
    if settings.llm_provider == "gemini":
        return GeminiVisionModel(settings.gemini_vision_model, settings.google_api_key)
    return OllamaVisionModel(settings.vision_model, settings.ollama_host)
