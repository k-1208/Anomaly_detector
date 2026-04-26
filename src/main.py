"""End-to-end anomaly pipeline: ingest -> hybrid score -> LLM enrichment."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from config.config import get_gemini_api_key
from llm import GeminiAnomalyNarrator, SystemInfoTool, build_gemini_payload
from models.metrics import ResourceMetrics
from utils import HybridAnomaly
from utils.ml_model import MLAnomalyScorer
from utils import PayloadIngestor, PayloadValidationError


# Only trigger LLM enrichment for actionable anomalies.
# Medium-only efficiency findings at very low score are kept structured but not narrated.
HIGH_SEVERITY_RULE_IDS = {"R2", "R5", "R7", "R8", "R9", "R10", "R12", "R13"}
LLM_MIN_FINAL_SCORE = 0.40


def _should_run_llm(result: Any) -> bool:
	"""Return True only when anomaly signal is strong or high-severity rules fire."""
	if not result.is_anomalous:
		return False

	has_high_severity_rule = any(
		rule_id in HIGH_SEVERITY_RULE_IDS for rule_id in result.rule_findings
	)
	return has_high_severity_rule or float(result.final_score) >= LLM_MIN_FINAL_SCORE


def _load_system_payload() -> list[dict[str, Any]]:
	"""Build a payload from live local system/process metrics."""
	system_tool = SystemInfoTool(max_processes=20)
	snapshot = system_tool.collect_system_snapshot()

	host_entry = {
		"resource_id": f"host:{snapshot.hostname}",
		"cpu_avg": float(snapshot.cpu_percent or 0.0),
		"cpu_p95": float(snapshot.cpu_percent or 0.0),
		"memory_avg": float(snapshot.memory_percent or 0.0),
		"network_pct": 0.0,
		"internet_facing": False,
		"identity_attached": True,
	}

	process_entries: list[dict[str, Any]] = []
	for proc in snapshot.processes:
		cpu_avg = float(proc.cpu_percent or 0.0)
		memory_avg = float(proc.memory_percent or 0.0)
		connections = int(proc.connections or 0)

		process_entries.append(
			{
				"resource_id": f"proc:{proc.name}:{proc.pid}",
				"cpu_avg": cpu_avg,
				"cpu_p95": min(100.0, cpu_avg * 1.15),
				"memory_avg": memory_avg,
				"network_pct": min(100.0, float(connections * 5)),
				"internet_facing": connections > 0,
				"identity_attached": bool(proc.username),
			}
		)

	return [host_entry, *process_entries]


def _read_payload_from_file(input_file: str) -> list[dict[str, Any]]:
	"""Read and parse payload JSON from a local file path."""
	with open(input_file, "r", encoding="utf-8") as f:
		loaded = json.load(f)

	if not isinstance(loaded, list):
		raise ValueError("Input file must contain a JSON array of resource objects")

	return loaded


def _write_output_file(output: dict[str, Any], output_file: str) -> None:
	"""Persist pipeline output JSON to disk."""
	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(output, f, indent=2)




def _build_baseline(resources: list[ResourceMetrics]) -> list[ResourceMetrics]:
	"""Build a baseline for ML fitting; currently uses current batch as seed data."""
	return resources


def run_pipeline(input_file: str | None = None) -> dict[str, Any]:
	"""Run full anomaly pipeline and return structured results.

	If input_file is provided, the payload is read from that JSON file.
	Otherwise, a payload is synthesized from local system metrics.
	"""
	if input_file:
		payload = _read_payload_from_file(input_file)
	else:
		payload = _load_system_payload()

	resources = PayloadIngestor.ingest_payload(payload)

	ml_scorer = MLAnomalyScorer()
	ml_scorer.fit(_build_baseline(resources))

	hybrid = HybridAnomaly(ml_scorer)
	hybrid_results = hybrid.score_batch(resources)

	api_key = get_gemini_api_key()
	narrator = GeminiAnomalyNarrator(api_key=api_key)
	system_tool = SystemInfoTool(max_processes=10)
	snapshot = system_tool.collect_system_snapshot()

	results: list[dict[str, Any]] = []
	for result in hybrid_results:
		item = asdict(result)

		if _should_run_llm(result):
			llm_payload = build_gemini_payload(
				resource_id=result.resource_id,
				is_anomalous=result.is_anomalous,
				anomaly_type=result.anomaly_type,
				confidence=result.confidence,
				final_score=result.final_score,
				reason=result.reason,
				rule_findings=result.rule_findings,
				ml_type=result.ml_type,
				ml_score=result.ml_score,
			)

			if api_key:
				item["llm_explanation"] = narrator.generate_explanation(
					llm_payload,
					system_snapshot=snapshot,
				)
			else:
				item["llm_prompt_preview"] = narrator.build_prompt(
					llm_payload,
					system_snapshot=snapshot,
				)

		results.append(item)

	return {
		"input_count": len(resources),
		"anomalies_found": sum(1 for r in hybrid_results if r.is_anomalous),
		"used_gemini": bool(api_key),
		"results": results,
	}


if __name__ == "__main__":
	# By default this uses live local system data; provide a JSON file path to override.
	INPUT_FILE = None
	OUTPUT_FILE = "pipeline_output.json"

	try:
		output = run_pipeline(input_file=INPUT_FILE)
		_write_output_file(output, OUTPUT_FILE)
		print(f"Saved pipeline output to: {OUTPUT_FILE}")
		print(
			f"Summary: input_count={output['input_count']}, "
			f"anomalies_found={output['anomalies_found']}, "
			f"used_gemini={output['used_gemini']}"
		)
		print(json.dumps(output, indent=2))
	except (PayloadValidationError, FileNotFoundError, ValueError, ImportError) as err:
		print(f"Pipeline failed: {err}")
