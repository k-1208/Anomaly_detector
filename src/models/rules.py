"""Dataclasses for deterministic rule engine results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuleFinding:
	rule_id: str
	finding: str
	severity: str
	reason: str


@dataclass
class ResourceRuleResult:
	resource_id: str
	findings: list[RuleFinding]
