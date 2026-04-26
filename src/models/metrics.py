"""Dataclasses for base resource metrics and derived features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourceMetrics:
	resource_id: str
	cpu_avg: float
	cpu_p95: float
	memory_avg: float
	network_pct: float
	internet_facing: bool
	identity_attached: bool


@dataclass
class ProcessedFeatures:
	resource_id: str
	net_cpu_ratio: float
	mem_cpu_ratio: float
	cpu_p95_delta: float
	utilisation: float
	net_mem_ratio: float
