"""Rule-based checks for infrastructure anomaly and inefficiency signals."""

from __future__ import annotations

from models.metrics import ProcessedFeatures, ResourceMetrics
from models.rules import ResourceRuleResult, RuleFinding
from utils.utils import featureProcessor


class RuleEngine:
	"""Applies deterministic checks over metrics and derived features."""

	@staticmethod
	def evaluate_one(resource: ResourceMetrics, features: ProcessedFeatures) -> ResourceRuleResult:
		findings: list[RuleFinding] = []

		cpu_avg = float(resource.cpu_avg)
		cpu_p95 = float(resource.cpu_p95)
		memory_avg = float(resource.memory_avg)
		network_pct = float(resource.network_pct)
		internet_facing = bool(resource.internet_facing)
		identity_attached = bool(resource.identity_attached)
		is_externally_exposed = internet_facing and network_pct >= 30

		# Group 1 - CPU rules
		if cpu_avg < 5:
			findings.append(
				RuleFinding(
					rule_id="R1",
					finding="idle_resource",
					severity="medium",
					reason="Instance is running but doing essentially no compute work",
				)
			)

		if cpu_avg > 85:
			findings.append(
				RuleFinding(
					rule_id="R2",
					finding="compute_thrash",
					severity="high",
					reason="Sustained high CPU - under-provisioned or runaway process",
				)
			)

		if cpu_p95 > 90:
			findings.append(
				RuleFinding(
					rule_id="R3",
					finding="cpu_p95_spike",
					severity="medium",
					reason="Extreme bursts even if average looks normal",
				)
			)

		if (cpu_p95 - cpu_avg) > 40:
			findings.append(
				RuleFinding(
					rule_id="R4",
					finding="bursty_workload",
					severity="low",
					reason="Highly intermittent load - hard to right-size, auto-scale candidate",
				)
			)

		# Group 2 - Memory rules
		if memory_avg > 85:
			findings.append(
				RuleFinding(
					rule_id="R5",
					finding="memory_pressure",
					severity="high",
					reason="OOM risk; swap likely active",
				)
			)

		if memory_avg < 15 and cpu_avg < 10:
			findings.append(
				RuleFinding(
					rule_id="R6",
					finding="over_provisioned",
					severity="medium",
					reason="Both memory and CPU idle - instance is wasteful",
				)
			)

		# Group 3 - Network rules
		if network_pct > 70 and cpu_avg < 20:
			findings.append(
				RuleFinding(
					rule_id="R7",
					finding="network_anomaly",
					severity="high",
					reason=(
						"High traffic with no compute work - exfiltration, relay, "
						"or misconfigured proxy"
					),
				)
			)

		if network_pct > 90:
			findings.append(
				RuleFinding(
					rule_id="R8",
					finding="network_saturation",
					severity="high",
					reason="Bandwidth ceiling - drops and retransmits likely",
				)
			)

		# Group 4 - Security rules
		if is_externally_exposed and identity_attached:
			findings.append(
				RuleFinding(
					rule_id="R9",
					finding="high_exposure_risk",
					severity="critical",
					reason=(
						"Internet-reachable with cloud credentials attached - breach "
						"pivot point"
					),
				)
			)

		if cpu_avg < 5 and is_externally_exposed:
			findings.append(
				RuleFinding(
					rule_id="R10",
					finding="idle_exposed",
					severity="high",
					reason=(
						"Dormant but publicly reachable - forgotten instance, "
						"unnecessary attack surface"
					),
				)
			)

		if cpu_avg < 5 and identity_attached:
			findings.append(
				RuleFinding(
					rule_id="R11",
					finding="credentialed_idle",
					severity="medium",
					reason=(
						"Instance holds cloud credentials but is doing nothing - "
						"credentials could be harvested"
					),
				)
			)

		if cpu_avg < 10 and is_externally_exposed and identity_attached:
			findings.append(
				RuleFinding(
					rule_id="R12",
					finding="over_provisioned_with_risk",
					severity="critical",
					reason=(
						"Wasteful AND dangerous - strongest combined signal in "
						"your dataset"
					),
				)
			)

		# Group 5 - Cross-metric compound rules
		if float(features.net_cpu_ratio) > 5 and network_pct > 20:
			findings.append(
				RuleFinding(
					rule_id="R13",
					finding="network_anomaly",
					severity="high",
					reason=(
						"Network far exceeds compute - the ratio matters more than "
						"either raw value"
					),
				)
			)

		if float(features.mem_cpu_ratio) > 10:
			findings.append(
				RuleFinding(
					rule_id="R14",
					finding="over_provisioned",
					severity="medium",
					reason=(
						"Memory loaded relative to CPU - wrong instance class or "
						"idle workload"
					),
				)
			)

		return ResourceRuleResult(resource_id=resource.resource_id, findings=findings)

	@staticmethod
	def evaluate(
		resources: list[ResourceMetrics],
		features: list[ProcessedFeatures] | None = None,
	) -> list[ResourceRuleResult]:
		if features is None:
			features = featureProcessor.process(resources)

		feature_by_id = {f.resource_id: f for f in features}
		results: list[ResourceRuleResult] = []

		for resource in resources:
			feature = feature_by_id.get(resource.resource_id)
			if feature is None:
				feature = featureProcessor.process_one(resource)
			results.append(RuleEngine.evaluate_one(resource, feature))

		return results
