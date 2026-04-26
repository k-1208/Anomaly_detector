"""Dataclasses for LLM requests and system snapshots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMAnomalyRequest:
	resource_id: str
	is_anomalous: bool
	anomaly_type: str
	confidence: float
	final_score: float
	reason: str
	rule_findings: list[str] | None = None
	ml_type: str | None = None
	ml_score: float | None = None
	security_note: str | None = None


@dataclass
class ProcessSnapshot:
	pid: int
	name: str
	cpu_percent: float | None
	memory_percent: float | None
	status: str | None
	username: str | None
	create_time: float | None
	cmdline: list[str] | None
	open_files: list[str] | None = None
	connections: int | None = None


@dataclass
class SystemSnapshot:
	timestamp_utc: str
	hostname: str
	os: str
	python_version: str
	cpu_count_logical: int | None
	cpu_percent: float | None
	memory_total: int | None
	memory_used: int | None
	memory_percent: float | None
	disk_total: int | None
	disk_used: int | None
	disk_percent: float | None
	network_bytes_sent: int | None
	network_bytes_recv: int | None
	processes: list[ProcessSnapshot]
	notes: list[str]
