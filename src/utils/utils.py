"""Utility classes for payload ingestion and validation."""

from __future__ import annotations

import json
from typing import Any, Iterable

from models.metrics import ProcessedFeatures, ResourceMetrics


class PayloadValidationError(ValueError):
	"""Raised when the incoming payload has invalid shape or values."""


class PayloadIngestor:
	"""Parses and validates resource metric payloads."""

	@staticmethod
	def _require_keys(item: dict[str, Any], required: Iterable[str], idx: int) -> None:
		missing = [k for k in required if k not in item]
		if missing:
			raise PayloadValidationError(
				f"Item {idx}: missing required field(s): {', '.join(missing)}"
			)

	@staticmethod
	def _validate_item(item: Any, idx: int) -> ResourceMetrics:
		if not isinstance(item, dict):
			raise PayloadValidationError(f"Item {idx}: each resource must be an object")

		required_fields = [
			"resource_id",
			"cpu_avg",
			"cpu_p95",
			"memory_avg",
			"network_pct",
			"internet_facing",
			"identity_attached",
		]
		PayloadIngestor._require_keys(item, required_fields, idx)

		resource_id = item["resource_id"]
		if not isinstance(resource_id, str) or not resource_id.strip():
			raise PayloadValidationError(
				f"Item {idx}: 'resource_id' must be a non-empty string"
			)

		cpu_avg = item["cpu_avg"]
		cpu_p95 = item["cpu_p95"]
		memory_avg = item["memory_avg"]
		network_pct = item["network_pct"]
		internet_facing = item["internet_facing"]
		identity_attached = item["identity_attached"]

		return ResourceMetrics(
			resource_id=resource_id.strip(),
			cpu_avg=cpu_avg,
			cpu_p95=cpu_p95,
			memory_avg=memory_avg,
			network_pct=network_pct,
			internet_facing=internet_facing,
			identity_attached=identity_attached,
		)

	@staticmethod
	def ingest_payload(payload: str | bytes | list[dict[str, Any]]) -> list[ResourceMetrics]:
		"""Parse and validate payload into strongly typed resource metrics."""
		if isinstance(payload, bytes):
			payload = payload.decode("utf-8")

		if isinstance(payload, str):
			try:
				parsed = json.loads(payload)
			except json.JSONDecodeError as exc:
				raise PayloadValidationError(f"Malformed JSON payload: {exc}") from exc
		else:
			parsed = payload

		if not isinstance(parsed, list):
			raise PayloadValidationError("Payload must be a list of resource objects")

		validated: list[ResourceMetrics] = []
		for idx, item in enumerate(parsed):
			validated.append(PayloadIngestor._validate_item(item, idx))
		return validated
	

class FeatureProcessor:
	"""Builds feature vectors from validated resource metrics."""

	@staticmethod
	def _safe_denom(value: float) -> float:
		return max(float(value), 0.01)

	@staticmethod
	def process_one(resource: ResourceMetrics) -> ProcessedFeatures:
		cpu_avg = float(resource.cpu_avg)
		cpu_p95 = float(resource.cpu_p95)
		memory_avg = float(resource.memory_avg)
		network_pct = float(resource.network_pct)

		net_cpu_ratio = network_pct / FeatureProcessor._safe_denom(cpu_avg)
		mem_cpu_ratio = memory_avg / FeatureProcessor._safe_denom(cpu_avg)
		cpu_p95_delta = cpu_p95 - cpu_avg
		utilisation = (cpu_avg + memory_avg) / 2
		net_mem_ratio = network_pct / FeatureProcessor._safe_denom(memory_avg)

		return ProcessedFeatures(
			resource_id=resource.resource_id,
			net_cpu_ratio=net_cpu_ratio,
			mem_cpu_ratio=mem_cpu_ratio,
			cpu_p95_delta=cpu_p95_delta,
			utilisation=utilisation,
			net_mem_ratio=net_mem_ratio,
		)

	@staticmethod
	def process(resources: list[ResourceMetrics]) -> list[ProcessedFeatures]:
		return [FeatureProcessor.process_one(resource) for resource in resources]


# Backward compatibility for existing imports.
featureProcessor = FeatureProcessor

