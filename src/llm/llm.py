"""Simple helpers for turning anomaly data into Gemini prompts."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from config.config import (
	DEFAULT_GEMINI_MODEL,
	GEMINI_PROMPT_INSTRUCTIONS,
	GEMINI_PROMPT_OUTPUT_FORMAT,
	create_gemini_model,
)
from llm.tools import SystemInfoTool, SystemSnapshot
from models.llm import LLMAnomalyRequest

class GeminiAnomalyNarrator:
	"""Builds prompts for Gemini and optionally calls the Gemini API."""

	def __init__(self, api_key: str | None = None, model_name: str = DEFAULT_GEMINI_MODEL):
		self.api_key = api_key
		self.model_name = model_name

	@staticmethod
	def build_prompt(
		anomaly: LLMAnomalyRequest | dict[str, Any],
		system_snapshot: SystemSnapshot | dict[str, Any] | None = None,
	) -> str:
		"""Convert anomaly data into a concise prompt for Gemini."""
		payload = asdict(anomaly) if isinstance(anomaly, LLMAnomalyRequest) else anomaly
		if system_snapshot is None:
			system_snapshot = SystemInfoTool().collect_system_snapshot()
		elif isinstance(system_snapshot, SystemSnapshot):
			system_snapshot = asdict(system_snapshot)

		return (
			f"{GEMINI_PROMPT_INSTRUCTIONS}\n\n"
			f"Anomaly data:\n{json.dumps(payload, indent=2)}\n\n"
			f"System context:\n{json.dumps(system_snapshot, indent=2)}\n\n"
			f"{GEMINI_PROMPT_OUTPUT_FORMAT}"
		)

	def generate_explanation(
		self,
		anomaly: LLMAnomalyRequest | dict[str, Any],
		system_snapshot: SystemSnapshot | dict[str, Any] | None = None,
	) -> str:
		"""Send the anomaly prompt to Gemini and return the response text."""
		if not self.api_key:
			raise ValueError("Gemini API key is required to generate an explanation")

		model = create_gemini_model(api_key=self.api_key, model_name=self.model_name)
		prompt = self.build_prompt(anomaly, system_snapshot=system_snapshot)
		response = model.generate_content(prompt)

		return getattr(response, "text", str(response))


def build_gemini_payload(
	resource_id: str,
	is_anomalous: bool,
	anomaly_type: str,
	confidence: float,
	final_score: float,
	reason: str,
	rule_findings: list[str] | None = None,
	ml_type: str | None = None,
	ml_score: float | None = None,
	security_note: str | None = None,
) -> dict[str, Any]:
	"""Convenience helper to build a Gemini-ready anomaly payload."""
	return asdict(
		LLMAnomalyRequest(
			resource_id=resource_id,
			is_anomalous=is_anomalous,
			anomaly_type=anomaly_type,
			confidence=confidence,
			final_score=final_score,
			reason=reason,
			rule_findings=rule_findings,
			ml_type=ml_type,
			ml_score=ml_score,
			security_note=security_note,
		)
	)
