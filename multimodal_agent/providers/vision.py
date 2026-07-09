"""Vision-language model backed by a local Ollama multimodal model."""

from __future__ import annotations

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


def build_vision_model(settings: Settings) -> OllamaVisionModel:
    return OllamaVisionModel(settings.vision_model, settings.ollama_host)
