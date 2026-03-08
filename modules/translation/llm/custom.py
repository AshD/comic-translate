from typing import Any
import logging
import sys
import numpy as np

from .gpt import GPTTranslation


logger = logging.getLogger(__name__)


def _safe_console_write(*parts: object) -> None:
    text = " ".join("" if part is None else str(part) for part in parts)
    stream = getattr(sys, 'stdout', None)
    if stream is None:
        logger.info(text)
        return
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, 'encoding', None) or 'utf-8'
        safe_text = text.encode(encoding, errors='replace').decode(encoding, errors='replace')
        try:
            stream.write(safe_text + "\n")
        except OSError:
            logger.info(safe_text)
    except OSError:
        logger.info(text)


class CustomTranslation(GPTTranslation):
    """Translation engine using custom LLM configurations with OpenAI-compatible API."""

    def __init__(self):
        super().__init__()

    def initialize(self, settings: Any, source_lang: str, target_lang: str, tr_key: str, **kwargs) -> None:
        """
        Initialize custom translation engine.

        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
        """
        super(GPTTranslation, self).initialize(settings, source_lang, target_lang, **kwargs)

        credentials = settings.get_credentials(settings.ui.tr(tr_key))
        self.api_key = credentials.get('api_key', '')
        self.model = credentials.get('model', '')
        self.api_base_url = credentials.get('api_url', '').rstrip('/')
        self.timeout = 120

    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        logger.info(
            "Custom LLM request endpoint=%s model=%s image_input=%s image_shape=%s",
            self.api_base_url,
            self.model,
            self.img_as_llm_input,
            getattr(image, 'shape', None),
        )
        _safe_console_write()
        _safe_console_write("=== Custom LLM Request ===")
        _safe_console_write(f"Endpoint: {self.api_base_url}")
        _safe_console_write(f"Model: {self.model}")
        _safe_console_write(f"Image input enabled: {self.img_as_llm_input}")
        _safe_console_write("System prompt:")
        _safe_console_write(system_prompt)
        _safe_console_write("User prompt:")
        _safe_console_write(user_prompt)

        response_text = super()._perform_translation(user_prompt, system_prompt, image)

        logger.info(
            "Custom LLM response endpoint=%s model=%s chars=%d",
            self.api_base_url,
            self.model,
            len(response_text or ""),
        )
        _safe_console_write("=== Custom LLM Response ===")
        _safe_console_write(response_text)
        _safe_console_write("=== End Custom LLM Exchange ===")
        _safe_console_write()

        return response_text
