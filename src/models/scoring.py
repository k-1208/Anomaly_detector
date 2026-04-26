"""Dataclasses for ML and hybrid scoring outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScorerOutput:
	resource_id: str
	final_score: float
	anomaly_type: str
	is_anomalous: bool
	confidence: float
	method_scores: dict
	triggered_signals: list
	reason: str


@dataclass
class HybridAnomalyOutput:
	resource_id: str
	is_anomalous: bool
	final_score: float
	anomaly_type: str
	confidence: float
	rule_findings: list[str]
	ml_score: float
	ml_type: str
	reason: str
