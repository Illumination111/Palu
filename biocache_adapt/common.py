"""Dependency-free accounting, timing, JSONL, budget, and manifest helpers."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Mapping

SCHEMA_VERSION = "biocache-proxy-v1"
BUDGET_RATIOS = (0.125, 0.25, 0.5)
PHASES = ("compress", "recover", "generate", "network")


def recursive_nbytes(obj: Any, seen: set[int] | None = None) -> int:
    """Count owned payload bytes recursively, deduplicating aliased objects."""
    if obj is None:
        return 0
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return int(obj.nbytes if isinstance(obj, memoryview) else len(obj))
    if isinstance(obj, (str, int, float, bool)):
        return 0
    if hasattr(obj, "element_size") and hasattr(obj, "nelement"):
        return int(obj.element_size()) * int(obj.nelement())
    if hasattr(obj, "nbytes") and not isinstance(obj, Mapping):
        try:
            return int(obj.nbytes)
        except (TypeError, ValueError):
            pass
    if isinstance(obj, Mapping):
        return sum(recursive_nbytes(k, seen) + recursive_nbytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(recursive_nbytes(v, seen) for v in obj)
    if dataclasses.is_dataclass(obj):
        return sum(recursive_nbytes(getattr(obj, f.name), seen) for f in dataclasses.fields(obj))
    for names in (("key_cache", "value_cache"), ("keys", "values")):
        if any(hasattr(obj, name) for name in names):
            return sum(recursive_nbytes(getattr(obj, name, None), seen) for name in names)
    if hasattr(obj, "__dict__"):
        return recursive_nbytes(vars(obj), seen)
    return 0


class PhaseTimer:
    def __init__(self) -> None:
        self.ms = {name: 0.0 for name in PHASES}

    @contextmanager
    def measure(self, phase: str):
        if phase not in self.ms:
            raise ValueError(f"unknown phase: {phase}")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[phase] += (time.perf_counter() - start) * 1000.0

    def reset(self) -> None:
        for phase in self.ms:
            self.ms[phase] = 0.0


def normalize_budget_ratio(value: float | str) -> float:
    text = str(value).strip()
    ratio = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    if ratio > 1:
        ratio /= 100.0
    if not any(abs(ratio - allowed) < 1e-12 for allowed in BUDGET_RATIOS):
        raise ValueError("budget must be one of 12.5%, 25%, or 50%")
    return ratio


def budget_report(ratio: float | str, dense_state: Any, persistent_state: Any) -> dict[str, Any]:
    ratio = normalize_budget_ratio(ratio)
    dense_bytes = recursive_nbytes(dense_state)
    actual_bytes = recursive_nbytes(persistent_state)
    budget_bytes = int(dense_bytes * ratio)
    return {
        "budget_ratio": ratio,
        "dense_reference_bytes": dense_bytes,
        "budget_bytes": budget_bytes,
        "actual_persistent_bytes": actual_bytes,
        "within_budget": actual_bytes <= budget_bytes,
    }


def result_record(
    *,
    method: str,
    suite: str,
    sample_id: str,
    adapter: Any,
    response: str = "",
    target: str = "",
    context_length: int = 0,
    needle_position: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if suite not in {"needle", "length_scan"}:
        raise ValueError("suite must be needle or length_scan")
    persistent = recursive_nbytes(adapter.persistent_state())
    static = recursive_nbytes(adapter.static_extra_state())
    temporary = recursive_nbytes(adapter.temporary_state())
    timings = {name: float(adapter.timings_ms.get(name, 0.0)) for name in PHASES}
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "suite": suite,
        "sample_id": str(sample_id),
        "context_length": int(context_length),
        "needle_position": needle_position,
        "budget_ratio": adapter.budget_ratio,
        "budget_bytes": adapter.last_budget.get("budget_bytes", 0),
        "actual_persistent_bytes": persistent,
        "actual_static_extra_bytes": static,
        "actual_temporary_bytes": temporary,
        "peak_temporary_bytes": int(adapter.peak_temporary_bytes),
        "timings_ms": timings,
        "response": response,
        "target": target,
        "exact_match": response.strip() == target.strip() if target else None,
        "metadata": dict(metadata or {}),
    }


def append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def collect_manifest(repo_root: str, config: Mapping[str, Any]) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", repo_root, *args], capture_output=True, text=True, timeout=5, check=False
            ).stdout.strip() or None
        except OSError:
            return None

    hashes = {}
    package_dir = os.path.dirname(__file__)
    for name in sorted(os.listdir(package_dir)):
        path = os.path.join(package_dir, name)
        if name.endswith(".py") and os.path.isfile(path):
            with open(path, "rb") as handle:
                hashes[name] = hashlib.sha256(handle.read()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "python": sys.version,
        "config": dict(config),
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "dirty": bool(git("status", "--porcelain"))},
        "source_sha256": hashes,
    }


def write_manifest(path: str, manifest: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, ensure_ascii=False, indent=2)


def dry_run_cli(adapter_cls: type, method: str, argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=f"{method} BioCache text proxy (no-model dry-run)")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--budget", default="25%", choices=("12.5%", "25%", "50%"))
    parser.add_argument("--suite", default="needle", choices=("needle", "length_scan"))
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    adapter = adapter_cls(budget_ratio=args.budget, dry_run=True)
    adapter.reset()
    adapter.prefill_archive("prefix needle=314159 suffix", segment_id="dry-0")
    response = adapter.query("What is the needle?")
    record = result_record(
        method=method, suite=args.suite, sample_id="dry-0", adapter=adapter,
        response=str(response), target="314159", context_length=27, needle_position=7,
        metadata={"dry_run": True},
    )
    if args.output:
        append_jsonl(args.output, record)
    if args.manifest:
        write_manifest(args.manifest, collect_manifest(os.path.dirname(os.path.dirname(__file__)), vars(args)))
    print(json.dumps(record, ensure_ascii=False))
    return record
