"""Hybrid anomaly scorer combining deterministic rules + ML model."""

from __future__ import annotations

from models.metrics import ResourceMetrics
from models.rules import ResourceRuleResult
from models.scoring import HybridAnomalyOutput, ScorerOutput
from utils.utils import featureProcessor
from utils.rule_engine import RuleEngine
from utils.ml_model.model import MLAnomalyScorer


class HybridAnomaly:
	"""Combines rule-based checks with ML anomaly detection."""

	def __init__(self, ml_scorer: MLAnomalyScorer):
		self.ml_scorer = ml_scorer

	@staticmethod
	def _severity_rank(severity: str) -> int:
		"""Rank severity for priority ordering."""
		rank_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
		return rank_map.get(severity, 0)

	@staticmethod
	def _is_actionable_rule_severity(severity: str) -> bool:
		return severity in {"high", "critical"}

	def score_one(self, resource: ResourceMetrics) -> HybridAnomalyOutput:
		"""Score a single resource using both rule engine and ML model."""
		
		# Get rule engine findings
		features = featureProcessor.process_one(resource)
		rule_result: ResourceRuleResult = RuleEngine.evaluate_one(resource, features)
		
		# Get ML model score
		ml_output: ScorerOutput = self.ml_scorer.score(resource)
		
		rule_ids = [f.rule_id for f in rule_result.findings]
		has_rule_match = len(rule_result.findings) > 0
		has_actionable_rule_match = any(
			self._is_actionable_rule_severity(f.severity) for f in rule_result.findings
		)
		
		# Determine anomaly status: true if ML flags it or actionable rules are matched.
		# Medium/low efficiency rules remain findings but do not force anomaly state.
		is_anomalous = has_actionable_rule_match or ml_output.is_anomalous
		
		# Pick anomaly type with priority:
		# 1. Critical/high severity rules override ML classification
		# 2. Otherwise use ML classification
		anomaly_type = ml_output.anomaly_type
		if has_actionable_rule_match:
			# Find the highest severity finding
			max_finding = max(
				rule_result.findings, 
				key=lambda f: self._severity_rank(f.severity)
			)
			anomaly_type = max_finding.finding
		
		# Combine scores: if rules triggered, boost score; else use ML
		if has_actionable_rule_match:
			# Rules caught something: use max(rule_signal, ml_score)
			# Interpret rules as 0.6+ anomaly signal
			final_score = max(0.6, ml_output.final_score)
		else:
			final_score = ml_output.final_score
		
		# Confidence: average of ML confidence and rule strength
		if has_actionable_rule_match:
			# Rule match gives high confidence (0.8)
			confidence = (ml_output.confidence + 0.8) / 2
		else:
			confidence = ml_output.confidence
		reason_parts = []
		
		if has_rule_match:
			rule_desc = ", ".join(
				f"{f.rule_id}({f.severity[0].upper()})"
				for f in sorted(rule_result.findings, key=lambda f: self._severity_rank(f.severity), reverse=True)
			)
			reason_parts.append(f"Deterministic rules matched: {rule_desc}.")
		
		reason_parts.append(f"ML model score: {ml_output.final_score:.4f}.")
		reason_parts.append(ml_output.reason)
		
		if is_anomalous:
			final_reason = " ".join(reason_parts)
		else:
			final_reason = f"No anomalies detected. {' '.join(reason_parts)}"
		
		return HybridAnomalyOutput(
			resource_id=resource.resource_id,
			is_anomalous=is_anomalous,
			final_score=final_score,
			anomaly_type=anomaly_type,
			confidence=round(confidence, 4),
			rule_findings=rule_ids,
			ml_score=ml_output.final_score,
			ml_type=ml_output.anomaly_type,
			reason=final_reason,
		)

	def score_batch(self, resources: list[ResourceMetrics]) -> list[HybridAnomalyOutput]:
		"""Score a batch of resources."""
		return [self.score_one(resource) for resource in resources]
