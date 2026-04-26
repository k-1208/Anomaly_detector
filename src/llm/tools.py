"""Utilities for collecting system and process context for LLM analysis."""

from __future__ import annotations

import importlib
import json
import platform
import socket
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from models.llm import ProcessSnapshot, SystemSnapshot

psutil = importlib.import_module("psutil")


class SystemInfoTool:
	"""Collects runtime system information and process context."""

	def __init__(self, max_processes: int = 10):
		self.max_processes = max_processes

	@staticmethod
	def _safe_get_open_files(proc: Any) -> list[str] | None:
		try:
			return [f.path for f in proc.open_files()]
		except Exception:
			return None

	@staticmethod
	def _safe_get_connections(proc: Any) -> int | None:
		try:
			return len(proc.connections())
		except Exception:
			return None

	@staticmethod
	def _collect_process_snapshot(proc: Any) -> ProcessSnapshot:
		try:
			return ProcessSnapshot(
				pid=proc.pid,
				name=proc.name(),
				cpu_percent=proc.cpu_percent(interval=None),
				memory_percent=proc.memory_percent(),
				status=proc.status(),
				username=proc.username(),
				create_time=proc.create_time(),
				cmdline=proc.cmdline(),
				open_files=SystemInfoTool._safe_get_open_files(proc),
				connections=SystemInfoTool._safe_get_connections(proc),
			)
		except Exception:
			return ProcessSnapshot(
				pid=getattr(proc, "pid", -1),
				name=getattr(proc, "name", lambda: "unknown")(),
				cpu_percent=None,
				memory_percent=None,
				status=None,
				username=None,
				create_time=None,
				cmdline=None,
			)

	def _collect_processes(self) -> list[ProcessSnapshot]:
		processes: list[ProcessSnapshot] = []
		for proc in psutil.process_iter([
			"pid",
			"name",
			"cpu_percent",
			"memory_percent",
			"status",
			"username",
			"create_time",
			"cmdline",
		]):
			try:
				processes.append(self._collect_process_snapshot(proc))
			except Exception:
				continue

		processes.sort(
			key=lambda p: ((p.cpu_percent or 0.0), (p.memory_percent or 0.0)),
			reverse=True,
		)
		return processes[: self.max_processes]

	def collect_system_snapshot(self) -> SystemSnapshot:
		"""Capture a point-in-time system snapshot with top processes."""
		notes: list[str] = []
		processes: list[ProcessSnapshot] = []

		# Prime cpu_percent so the next call returns a meaningful value.
		psutil.cpu_percent(interval=None)
		for proc in psutil.process_iter():
			try:
				proc.cpu_percent(interval=None)
			except Exception:
				continue
		processes = self._collect_processes()

		memory_total = memory_used = memory_percent = None
		disk_total = disk_used = disk_percent = None
		network_bytes_sent = network_bytes_recv = None
		cpu_count_logical = None
		cpu_percent = None

		if psutil is not None:
			try:
				memory = psutil.virtual_memory()
				memory_total = int(memory.total)
				memory_used = int(memory.used)
				memory_percent = float(memory.percent)
			except Exception:
				notes.append("Unable to read memory snapshot")

			try:
				disk = psutil.disk_usage("/")
				disk_total = int(disk.total)
				disk_used = int(disk.used)
				disk_percent = float(disk.percent)
			except Exception:
				notes.append("Unable to read disk snapshot")

			try:
				net = psutil.net_io_counters()
				network_bytes_sent = int(net.bytes_sent)
				network_bytes_recv = int(net.bytes_recv)
			except Exception:
				notes.append("Unable to read network snapshot")

			try:
				cpu_count_logical = psutil.cpu_count(logical=True)
				cpu_percent = float(psutil.cpu_percent(interval=0.1))
			except Exception:
				notes.append("Unable to read CPU snapshot")

		return SystemSnapshot(
			timestamp_utc=datetime.now(timezone.utc).isoformat(),
			hostname=socket.gethostname(),
			os=f"{platform.system()} {platform.release()}",
			python_version=platform.python_version(),
			cpu_count_logical=cpu_count_logical,
			cpu_percent=cpu_percent,
			memory_total=memory_total,
			memory_used=memory_used,
			memory_percent=memory_percent,
			disk_total=disk_total,
			disk_used=disk_used,
			disk_percent=disk_percent,
			network_bytes_sent=network_bytes_sent,
			network_bytes_recv=network_bytes_recv,
			processes=processes,
			notes=notes,
		)

	def build_llm_payload(
		self,
		anomaly_data: dict[str, Any],
		system_snapshot: SystemSnapshot | None = None,
	) -> dict[str, Any]:
		"""Combine anomaly details with system context for LLM consumption."""
		if system_snapshot is None:
			system_snapshot = self.collect_system_snapshot()

		return {
			"anomaly": anomaly_data,
			"system_snapshot": asdict(system_snapshot),
			"serialized": json.dumps(
				{
					"anomaly": anomaly_data,
					"system_snapshot": asdict(system_snapshot),
				},
				indent=2,
			),
		}

	def collect_process_by_pid(self, pid: int) -> ProcessSnapshot:
		"""Collect a detailed snapshot for a single process."""
		proc = psutil.Process(pid)
		return self._collect_process_snapshot(proc)

	def collect_processes_by_name(self, name: str) -> list[ProcessSnapshot]:
		"""Collect all processes matching a name substring."""
		matches: list[ProcessSnapshot] = []
		for proc in psutil.process_iter():
			try:
				proc_name = proc.name()
				if name.lower() in proc_name.lower():
					matches.append(self._collect_process_snapshot(proc))
			except Exception:
				continue
		return matches


def snapshot_to_json(snapshot: SystemSnapshot | ProcessSnapshot | dict[str, Any]) -> str:
	"""Serialize a snapshot or payload to JSON for logging or LLM transport."""
	if isinstance(snapshot, (SystemSnapshot, ProcessSnapshot)):
		snapshot = asdict(snapshot)
	return json.dumps(snapshot, indent=2)
