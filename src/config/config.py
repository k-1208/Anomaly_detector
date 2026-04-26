"""Configuration and prompt policy for Gemini narration."""

from __future__ import annotations

import os
from typing import Any

try:
	import google.generativeai as genai  # pyright: ignore[reportMissingImports]
except ImportError:
	genai = None

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

GEMINI_PROMPT_INSTRUCTIONS = (
	"You are an infrastructure anomaly analyst. "
	"Explain the following anomaly in clear, concise language. "
	"Focus on what the metrics mean, why it looks abnormal, "
	"and what action should be taken. "
	"If there are security concerns, mention them explicitly."
)

GEMINI_PROMPT_OUTPUT_FORMAT = (
	"Return:\n"
	"1. short summary\n"
	"2. why it matters\n"
	"3. suggested action\n"
	"4. security note if relevant"
)


def get_gemini_api_key() -> str | None:
	"""Read Gemini API key from environment."""
	return os.getenv("GEMINI_API_KEY")


def create_gemini_model(
	api_key: str | None = None,
	model_name: str = DEFAULT_GEMINI_MODEL,
) -> Any:
	"""Create and return a configured Gemini model client."""
	if genai is None:
		raise ImportError(
			"google-generativeai is not installed. Install it to use Gemini support."
		)

	resolved_api_key = api_key or get_gemini_api_key()
	if not resolved_api_key:
		raise ValueError("Gemini API key is required to initialize Gemini model")

	genai.configure(api_key=resolved_api_key)
	return genai.GenerativeModel(model_name)
