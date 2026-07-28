#!/usr/bin/env python3
"""Fail-closed validator for local multi-GPU qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

SCHEMA_VERSION = "mold.local.multi-gpu.qualification.v1"
REQUIRED_CHECKS = {
    "both_devices_discovered",
    "both_devices_executed",
    "busy_disable_drained",
    "queue_replanned_after_disable",
    "all_disabled_maintenance",
    "queued_cancellation",
    "restart_persistence",
    "legacy_rollback",
    "selector_matrix",
    "client_projection",
    "models_tree_unchanged",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def validate_against_schema(report: object) -> None:
    """Reuse the repository's hermetic JSON-Schema subset implementation."""
    helper_path = pathlib.Path(__file__).with_name(
        "validate-cuda-qualification-report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mold_cuda_qualification_schema_helper", helper_path
    )
    if spec is None or spec.loader is None:
        fail(f"cannot load hermetic schema helper: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    schema_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "docs"
        / "qualification"
        / "local-multi-gpu-report.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        helper.audit_schema_keywords(schema)
        helper.validate_schema(report, schema, schema)
    except helper.ValidationFailure as error:
        fail(f"schema validation failed: {error}")


def validate(report_path: pathlib.Path, require_passing: bool) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_against_schema(report)
    if report.get("schema_version") != SCHEMA_VERSION:
        fail(f"unsupported schema_version: {report.get('schema_version')!r}")

    expected = report.get("host", {}).get("expected_gpu_uuids")
    if not isinstance(expected, list) or len(expected) < 2:
        fail("host.expected_gpu_uuids must contain at least two devices")
    if len(expected) != len(set(expected)):
        fail("host.expected_gpu_uuids contains duplicates")
    if not all(isinstance(value, str) and GPU_UUID_RE.fullmatch(value) for value in expected):
        fail("host.expected_gpu_uuids contains an invalid NVIDIA GPU UUID")

    devices = report.get("host", {}).get("devices")
    if not isinstance(devices, list):
        fail("host.devices must be an array")
    observed = [device.get("uuid") for device in devices if isinstance(device, dict)]
    if set(observed) != set(expected) or len(observed) != len(expected):
        fail("host.devices is not the exact expected GPU UUID inventory")

    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECKS:
        fail("checks must contain exactly the required acceptance gates")
    for name, check in checks.items():
        if not isinstance(check, dict):
            fail(f"check {name} must be an object")
        if check.get("status") not in {"passed", "failed"}:
            fail(f"check {name} has an invalid status")
        if not isinstance(check.get("summary"), str) or not check["summary"]:
            fail(f"check {name} is missing a summary")
        if not isinstance(check.get("evidence_labels"), list):
            fail(f"check {name} is missing evidence labels")

    evidence = report.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        fail("evidence must be a non-empty array")
    labels: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"label", "path", "sha256"}:
            fail("every evidence item must contain exactly label/path/sha256")
        label = item["label"]
        if label in labels:
            fail(f"duplicate evidence label: {label}")
        labels.add(label)
        path = pathlib.Path(item["path"])
        if not path.is_absolute():
            path = (report_path.parent / path).resolve()
        if not path.is_file():
            fail(f"evidence file does not exist: {path}")
        expected_sha = item["sha256"]
        if not SHA256_RE.fullmatch(expected_sha):
            fail(f"evidence {label} has an invalid SHA-256")
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            fail(f"evidence {label} hash mismatch: {actual_sha} != {expected_sha}")

    for name, check in checks.items():
        unknown = set(check["evidence_labels"]) - labels
        if unknown:
            fail(f"check {name} references unknown evidence labels: {sorted(unknown)}")

    candidate = report.get("candidate", {})
    binary = pathlib.Path(candidate.get("path", ""))
    if not binary.is_file():
        fail(f"candidate binary no longer exists: {binary}")
    if sha256(binary) != candidate.get("sha256"):
        fail("candidate binary hash no longer matches the report")
    server_pid = candidate.get("server_pid")
    if server_pid is not None and (
        not isinstance(server_pid, int)
        or isinstance(server_pid, bool)
        or server_pid <= 0
    ):
        fail("candidate.server_pid must be null or identify an exact qualification process")
    port = report.get("isolation", {}).get("port")
    if not isinstance(port, int) or isinstance(port, bool) or port > 65535:
        fail("isolation.port must be a valid non-privileged TCP port")

    qualified = report.get("hardware_qualified")
    all_passed = all(check["status"] == "passed" for check in checks.values())
    if qualified is not all_passed:
        fail("hardware_qualified must equal the conjunction of all required checks")
    if qualified and server_pid is None:
        fail("passing qualification requires the exact candidate server PID")
    if require_passing and not qualified:
        fail("report is valid failure evidence, not passing hardware qualification")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="validate report integrity without requiring hardware_qualified=true",
    )
    args = parser.parse_args()
    try:
        report = validate(args.report.resolve(), not args.allow_failure)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"invalid local multi-GPU report: {error}", file=sys.stderr)
        return 1
    print(
        f"validated {report['schema_version']}: "
        f"hardware_qualified={str(report['hardware_qualified']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
