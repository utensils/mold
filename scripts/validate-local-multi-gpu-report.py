#!/usr/bin/env python3
"""Fail-closed validator for local 2x RTX 3090 qualification evidence."""

from __future__ import annotations

import argparse
import binascii
import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import re
import struct
import sys
import zlib

SCHEMA_VERSION = "mold.local.multi-gpu.qualification.v2"
EVIDENCE_SCHEMA_VERSION = "mold.local.multi-gpu.evidence.v2"
QUALIFICATION_PROFILE = "local-2x-rtx3090-sm86"
RTX3090_NAME = "NVIDIA GeForce RTX 3090"
RTX3090_MEMORY_MIB = 24576
RTX3090_COMPUTE_CAPABILITY = "8.6"
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
CHECK_EVIDENCE = {
    "both_devices_discovered": {"nvidia-inventory", "initial-api-projection"},
    "both_devices_executed": {"parallel-runtime-samples", "parallel-results"},
    "busy_disable_drained": {"parallel-runtime-samples", "parallel-results"},
    "queue_replanned_after_disable": {
        "parallel-runtime-samples",
        "parallel-results",
    },
    "all_disabled_maintenance": {"all-disabled-maintenance"},
    "queued_cancellation": {"queued-cancellation"},
    "restart_persistence": {"restart-persistence", "restart-server-log"},
    "legacy_rollback": {"legacy-rollback", "legacy-server-log"},
    "selector_matrix": {
        "selector-matrix",
        "ambiguous-selector-source-contract",
    },
    "client_projection": {"client-projection", "initial-api-projection"},
    "models_tree_unchanged": {"models-tree-before", "models-tree-after"},
}
MANDATORY_EVIDENCE = {
    "normalized-request",
    "model-artifacts",
    "source-provenance",
    "candidate-version",
    "nvidia-inventory",
    "initial-api-projection",
    "client-projection",
    "parallel-runtime-samples",
    "parallel-results",
    "queued-cancellation",
    "all-disabled-maintenance",
    "restart-persistence",
    "legacy-rollback",
    "selector-matrix",
    "ambiguous-selector-source-contract",
    "models-tree-before",
    "models-tree-after",
    "primary-command",
    "restart-command",
    "legacy-command",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
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


def parse_inventory(text: str) -> list[dict[str, object]]:
    devices = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        row = [part.strip() for part in raw.split(",")]
        if len(row) != 6:
            fail(f"nvidia-inventory has malformed row: {row!r}")
        index, gpu_uuid, name, memory, capability, driver = row
        try:
            devices.append(
                {
                    "index": int(index),
                    "uuid": gpu_uuid,
                    "name": name,
                    "memory_total_mib": int(memory),
                    "compute_capability": capability,
                    "driver_version": driver,
                }
            )
        except ValueError as error:
            fail(f"nvidia-inventory has non-numeric fields: {row!r}")
    return devices


def validate_hardware_profile(
    devices: list[dict[str, object]], expected: list[str]
) -> None:
    if len(expected) != 2 or len(set(expected)) != 2:
        fail("local-2x-rtx3090-sm86 requires exactly two unique expected UUIDs")
    if not all(isinstance(value, str) and GPU_UUID_RE.fullmatch(value) for value in expected):
        fail("host.expected_gpu_uuids contains an invalid NVIDIA GPU UUID")
    observed = [device.get("uuid") for device in devices]
    if len(devices) != 2 or observed != list(dict.fromkeys(observed)):
        fail("local-2x-rtx3090-sm86 requires exactly two unique devices")
    if set(observed) != set(expected):
        fail("host.devices is not the exact expected GPU UUID inventory")
    for device in devices:
        if device.get("name") != RTX3090_NAME:
            fail("qualification device is not NVIDIA GeForce RTX 3090")
        if device.get("memory_total_mib") != RTX3090_MEMORY_MIB:
            fail("qualification device does not have exactly 24576 MiB")
        if device.get("compute_capability") != RTX3090_COMPUTE_CAPABILITY:
            fail("qualification device is not compute capability 8.6")


def validate_png(path: pathlib.Path, width: int, height: int) -> None:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        fail(f"output is not a PNG: {path}")
    offset = 8
    ihdr = None
    idat = bytearray()
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            fail(f"PNG has a truncated chunk: {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            fail(f"PNG chunk extends beyond file: {path}")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        if binascii.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            fail(f"PNG CRC mismatch: {path}")
        if kind == b"IHDR":
            if ihdr is not None or length != 13:
                fail(f"PNG has invalid IHDR: {path}")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            if length != 0 or end != len(data):
                fail(f"PNG has invalid IEND or trailing bytes: {path}")
            saw_iend = True
            break
        offset = end
    if ihdr is None or not idat or not saw_iend:
        fail(f"PNG is missing required chunks: {path}")
    actual_width, actual_height, depth, color_type, compression, filtering, interlace = ihdr
    if (actual_width, actual_height) != (width, height):
        fail(f"PNG dimensions do not match request: {path}")
    if depth != 8 or compression != 0 or filtering != 0 or interlace != 0:
        fail(f"PNG is not non-interlaced 8-bit output: {path}")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        fail(f"PNG has unsupported color type: {path}")
    try:
        decoded = zlib.decompress(bytes(idat))
    except zlib.error as error:
        fail(f"PNG IDAT cannot be decompressed: {path}: {error}")
    if len(decoded) != actual_height * (1 + actual_width * channels):
        fail(f"PNG decoded byte count does not match dimensions: {path}")
    for row in range(actual_height):
        if decoded[row * (1 + actual_width * channels)] > 4:
            fail(f"PNG has an invalid row filter: {path}")


def has_typed_cancelled_sse(value: object) -> bool:
    if not isinstance(value, str):
        return False
    event_name = None
    data_lines: list[str] = []

    def contains_cancelled(item: object) -> bool:
        if item == "cancelled":
            return True
        if isinstance(item, dict):
            return any(contains_cancelled(child) for child in item.values())
        if isinstance(item, list):
            return any(contains_cancelled(child) for child in item)
        return False

    for raw in [*value.splitlines(), ""]:
        if raw == "":
            if event_name == "error" and data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    payload = None
                if contains_cancelled(payload):
                    return True
            event_name = None
            data_lines = []
        elif raw.startswith("event:"):
            event_name = raw.removeprefix("event:").strip()
        elif raw.startswith("data:"):
            data_lines.append(raw.removeprefix("data:").lstrip())
    return False


def load_evidence(
    report_path: pathlib.Path, items: object
) -> tuple[dict[str, pathlib.Path], dict[str, object]]:
    if not isinstance(items, list) or not items:
        fail("evidence must be a non-empty array")
    root = pathlib.Path(str(report_path) + ".d").resolve()
    paths: dict[str, pathlib.Path] = {}
    values: dict[str, object] = {}
    used_paths: set[pathlib.Path] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "label",
            "path",
            "sha256",
            "kind",
        }:
            fail("every evidence item must contain exactly label/path/sha256/kind")
        label = item["label"]
        if not isinstance(label, str) or not label or label in paths:
            fail(f"duplicate or invalid evidence label: {label!r}")
        path = pathlib.Path(item["path"])
        if not path.is_absolute():
            path = report_path.parent / path
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            fail(f"evidence path escapes exact report evidence directory: {path}")
        if path in used_paths:
            fail(f"multiple evidence labels reuse one path: {path}")
        used_paths.add(path)
        if not path.is_file():
            fail(f"evidence file does not exist: {path}")
        expected_sha = item["sha256"]
        if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
            fail(f"evidence {label} has an invalid SHA-256")
        if sha256(path) != expected_sha:
            fail(f"evidence {label} hash mismatch")
        kind = item["kind"]
        if kind == "json":
            try:
                values[label] = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                fail(f"evidence {label} is not typed JSON: {error}")
        elif kind == "jsonl":
            rows = []
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                fail(f"evidence {label} is not typed JSONL: {error}")
            values[label] = rows
        elif kind == "text":
            values[label] = path.read_text(encoding="utf-8", errors="strict")
        elif kind == "png":
            values[label] = None
        else:
            fail(f"evidence {label} has unsupported kind {kind!r}")
        paths[label] = path
    return paths, values


def require_evidence_version(label: str, value: object) -> dict:
    if not isinstance(value, dict):
        fail(f"evidence {label} must be an object")
    if value.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        fail(f"evidence {label} has the wrong evidence schema version")
    return value


def validate_sandbox_argv(
    label: str, argv: object, candidate_path: pathlib.Path | None = None
) -> None:
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        fail(f"{label} sandbox argv is not a string array")
    if argv[:2] != ["bwrap", "--die-with-parent"]:
        fail(f"{label} did not use the Bubblewrap launcher")
    if not any(
        argv[index : index + 3] == ["--ro-bind", "/", "/"]
        for index in range(len(argv) - 2)
    ):
        fail(f"{label} did not mount the host root read-only")
    try:
        separator = argv.index("--")
    except ValueError:
        fail(f"{label} Bubblewrap argv lacks a command separator")
    if candidate_path is not None:
        if separator + 1 >= len(argv):
            fail(f"{label} Bubblewrap argv lacks the candidate command")
        if pathlib.Path(argv[separator + 1]).resolve() != candidate_path:
            fail(f"{label} Bubblewrap argv does not execute the exact candidate")


def validate_passing_evidence(
    report: dict,
    report_path: pathlib.Path,
    paths: dict[str, pathlib.Path],
    values: dict[str, object],
) -> None:
    missing = MANDATORY_EVIDENCE - set(paths)
    if missing:
        fail(f"passing report is missing mandatory typed evidence: {sorted(missing)}")

    expected = report["host"]["expected_gpu_uuids"]
    validate_hardware_profile(report["host"]["devices"], expected)
    inventory = parse_inventory(str(values["nvidia-inventory"]))
    if inventory != report["host"]["devices"]:
        fail("typed nvidia inventory does not exactly match report host devices")

    request_path = pathlib.Path(report["request"]["path"]).resolve()
    if request_path != paths["normalized-request"]:
        fail("request.path is not the normalized-request evidence path")
    if sha256(request_path) != report["request"]["sha256"]:
        fail("request SHA-256 does not match normalized request")
    request = values["normalized-request"]
    if not isinstance(request, dict) or request.get("model") != report["request"]["model"]:
        fail("normalized request model does not match report")
    if request.get("output_format", "png") != "png":
        fail("passing qualification requires PNG output")
    width, height = request.get("width"), request.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        fail("normalized request lacks integer dimensions")

    artifacts = report["request"]["artifacts"]
    if not artifacts:
        fail("passing qualification requires exact model artifacts")
    models_dir = pathlib.Path(report["isolation"]["models_dir"]).resolve()
    for artifact in artifacts:
        path = pathlib.Path(artifact["path"]).resolve()
        try:
            path.relative_to(models_dir)
        except ValueError:
            fail(f"model artifact escapes models_dir: {path}")
        if not path.is_file() or path.is_symlink():
            fail(f"model artifact is not a regular file: {path}")
        if path.stat().st_size != artifact["size"] or sha256(path) != artifact["sha256"]:
            fail(f"model artifact changed: {path}")
    artifact_evidence = require_evidence_version(
        "model-artifacts", values["model-artifacts"]
    )
    if (
        artifact_evidence.get("model") != report["request"]["model"]
        or artifact_evidence.get("artifacts") != artifacts
    ):
        fail("model-artifacts evidence is not exact")

    source = require_evidence_version("source-provenance", values["source-provenance"])
    if source.get("commit") != report["source_commit"]:
        fail("source provenance does not match report commit")
    version = require_evidence_version("candidate-version", values["candidate-version"])
    binary_path = pathlib.Path(report["candidate"]["path"]).resolve()
    validate_sandbox_argv("candidate version", version.get("argv"), binary_path)
    if (
        version.get("binary_sha256") != report["candidate"]["sha256"]
        or version.get("version") != report["candidate"]["version"]
        or version.get("sandboxed") is not True
    ):
        fail("candidate version evidence is not sandboxed and binary-bound")

    initial = values["initial-api-projection"]
    if not isinstance(initial, dict):
        fail("initial API projection must be an object")
    if initial.get("status", {}).get("hostname") != report["host"]["hostname"]:
        fail("server status hostname does not match report host identity")
    api_devices = initial.get("devices", {}).get("devices", [])
    api_mapping = {
        device.get("id"): device.get("nvml_uuid")
        for device in api_devices
        if isinstance(device, dict)
    }
    if len(api_mapping) != 2 or set(api_mapping.values()) != set(expected):
        fail("initial API device mapping is not the exact two-GPU inventory")
    if not all(
        device.get("desired_enabled") is True
        and device.get("admin_state") == "enabled"
        and device.get("schedulable") is True
        for device in api_devices
    ):
        fail("initial API devices are not all enabled and schedulable")
    expected_projection = {
        (device.get("ordinal"), device.get("name")) for device in api_devices
    }
    status_projection = {
        (device.get("ordinal"), device.get("name"))
        for device in initial.get("status", {}).get("gpus", [])
    }
    resource_projection = {
        (device.get("ordinal"), device.get("name"))
        for device in initial.get("resources", {}).get("gpus", [])
    }
    if status_projection != expected_projection:
        fail("legacy status does not exactly project API ordinal/name identity")
    if resource_projection != expected_projection:
        fail("resource telemetry does not exactly project API ordinal/name identity")
    capabilities = initial.get("capabilities", {})
    if not (
        capabilities.get("devices", {}).get("available")
        and capabilities.get("devices", {}).get("lifecycle")
        and capabilities.get("devices", {}).get("planned_lanes")
        and capabilities.get("dispatch", {}).get("v2_authoritative")
        and capabilities.get("dispatch", {}).get("active_mode") == "v2"
    ):
        fail("initial API projection is not authoritative Scheduler V2")

    client = require_evidence_version("client-projection", values["client-projection"])
    validate_sandbox_argv("client projection", client.get("argv"), binary_path)
    if client.get("server_pid") != report["candidate"]["server_pid"]:
        fail("client projection is not bound to the primary server PID")
    client_ids = [row.get("id") for row in client.get("gpu_list", {}).get("devices", [])]
    if client_ids != [device.get("id") for device in api_devices]:
        fail("client projection IDs differ from the API")

    parallel = require_evidence_version("parallel-results", values["parallel-results"])
    pid = report["candidate"]["server_pid"]
    if parallel.get("server_pid") != pid:
        fail("parallel evidence is not bound to the exact primary PID")
    if set(parallel.get("observed_active_uuids", [])) != set(expected):
        fail("parallel evidence omitted active work on a qualification UUID")
    if set(parallel.get("observed_compute_uuids", [])) != set(expected):
        fail("parallel evidence omitted exact-PID compute context on a UUID")
    decisive = False
    for sample in parallel.get("decisive_samples", []):
        active = sample.get("active", [])
        active_uuids = {row.get("gpu_uuid") for row in active}
        work_ids = {row.get("work_id") for row in active}
        compute_uuids = {
            row.get("gpu_uuid")
            for row in sample.get("compute_apps", [])
            if row.get("pid") == pid
        }
        if (
            sample.get("server_pid") == pid
            and active_uuids == set(expected)
            and len(work_ids) == 2
            and active_uuids <= compute_uuids
        ):
            decisive = True
            break
    if not decisive:
        fail("no same-sample PID/work-ID/UUID execution proof exists")

    ordinal_to_uuid = {
        device.get("ordinal"): device.get("nvml_uuid") for device in api_devices
    }
    outputs = parallel.get("results", [])
    if len(outputs) != report["request"]["job_count"]:
        fail("parallel output count does not match requested job count")
    work_ids: set[str] = set()
    output_uuids: set[str] = set()
    output_by_work: dict[str, dict] = {}
    execution_bindings = parallel.get("execution_bindings")
    if not isinstance(execution_bindings, dict):
        fail("parallel evidence lacks exact work-ID execution bindings")
    for output in outputs:
        index = output.get("index")
        label = f"parallel-output-{index}"
        if not isinstance(index, int) or label not in paths:
            fail("parallel output lacks mandatory typed PNG evidence")
        if output.get("status") != 200 or output.get("content_type") != "image/png":
            fail(f"parallel output {index} was not a successful PNG response")
        if output.get("path") != str(paths[label]) or output.get("sha256") != sha256(
            paths[label]
        ):
            fail(f"parallel output {index} path/hash binding is invalid")
        if output.get("size") != paths[label].stat().st_size:
            fail(f"parallel output {index} size binding is invalid")
        validate_png(paths[label], width, height)
        if output.get("decoded") != {"decoded": True, "width": width, "height": height}:
            fail(f"parallel output {index} decoded metadata is invalid")
        work_id = output.get("work_id")
        if not isinstance(work_id, str) or not work_id or work_id in work_ids:
            fail("parallel outputs do not have unique exact work IDs")
        work_ids.add(work_id)
        output_by_work[work_id] = output
        ordinal = output.get("gpu_ordinal")
        gpu_uuid = output.get("gpu_uuid")
        if ordinal_to_uuid.get(ordinal) != gpu_uuid:
            fail("output GPU ordinal does not map to its claimed UUID")
        if execution_bindings.get(work_id) != gpu_uuid:
            fail("output work ID is not bound to exact-PID execution on its UUID")
        output_uuids.add(gpu_uuid)
    if output_uuids != set(expected):
        fail("validated outputs do not cover both exact GPU UUIDs")

    disabled_id = parallel.get("disabled_id")
    pre = parallel.get("pre_disable_assignments")
    post = parallel.get("post_disable_assignments")
    if (
        not isinstance(pre, dict)
        or not pre
        or not isinstance(post, dict)
        or set(pre) != set(post)
        or set(parallel.get("replanned_work_ids", [])) != set(pre)
        or any(value != disabled_id for value in pre.values())
        or any(value == disabled_id for value in post.values())
    ):
        fail("same queued work IDs were not proven to move after disable")
    for work_id, device_id in post.items():
        output = output_by_work.get(work_id)
        if output is None or output.get("gpu_uuid") != api_mapping.get(device_id):
            fail("replanned work output did not execute on its new exact UUID")
    if (
        parallel.get("disable_response", {}).get("status") != 202
        or parallel.get("disable_response", {}).get("body", {}).get("admin_state")
        != "draining"
        or parallel.get("drained", {}).get("admin_state") != "disabled"
        or parallel.get("reenabled", {}).get("admin_state") != "enabled"
    ):
        fail("busy drain/disable/re-enable evidence is incomplete")

    cancellation = require_evidence_version(
        "queued-cancellation", values["queued-cancellation"]
    )
    cancellation_job = cancellation.get("job_id")
    cancellation_work = cancellation.get("work_id")
    queue_before = cancellation.get("queue_before_cancel")
    devices_before = cancellation.get("devices_before_cancel")
    queue_after = cancellation.get("queue_after")
    devices_after = cancellation.get("devices_after")
    if not all(
        isinstance(value, dict)
        for value in (queue_before, devices_before, queue_after, devices_after)
    ):
        fail("queued cancellation lacks typed before/after scheduler evidence")
    before_entries = [
        entry
        for entry in queue_before.get("entries", [])
        if isinstance(entry, dict) and entry.get("id") == cancellation_job
    ]
    before_work = [
        item
        for item in (queue_before.get("plan") or {}).get("work_items", [])
        if isinstance(item, dict)
        and item.get("parent_id") == cancellation_job
        and item.get("work_id") == cancellation_work
    ]
    before_active = any(
        device.get("active_work_id") == cancellation_work
        for device in devices_before.get("devices", [])
        if isinstance(device, dict)
    )
    after_queued = any(
        entry.get("id") == cancellation_job
        for entry in queue_after.get("entries", [])
        if isinstance(entry, dict)
    )
    after_planned = any(
        item.get("work_id") == cancellation_work
        for item in (queue_after.get("plan") or {}).get("work_items", [])
        if isinstance(item, dict)
    )
    after_active = any(
        device.get("active_work_id") == cancellation_work
        for device in devices_after.get("devices", [])
        if isinstance(device, dict)
    )
    output_before = cancellation.get("output_tree_before")
    output_after = cancellation.get("output_tree_after")
    if (
        cancellation.get("server_pid") != pid
        or cancellation.get("cancel_status") != 204
        or cancellation.get("resume_status") != 200
        or cancellation.get("stream_http_status") != 200
        or cancellation.get("typed_cancelled") is not True
        or cancellation.get("never_active") is not True
        or cancellation.get("output_tree_unchanged") is not True
        or cancellation.get("queue_was_paused") is not True
        or not cancellation_job
        or not cancellation_work
        or queue_before.get("paused") is not True
        or len(before_entries) != 1
        or before_entries[0].get("state") != "queued"
        or len(before_work) != 1
        or before_work[0].get("activity_phase")
        not in {"queued", "planned", "blocked"}
        or before_active
        or after_queued
        or after_planned
        or after_active
        or not isinstance(output_before, list)
        or output_after != output_before
        or not has_typed_cancelled_sse(cancellation.get("stream_tail"))
    ):
        fail("queued cancellation lacks typed no-inference proof")

    maintenance = require_evidence_version(
        "all-disabled-maintenance", values["all-disabled-maintenance"]
    )
    if (
        maintenance.get("server_pid") != pid
        or maintenance.get("status") not in {409, 503}
        or "maintenance" not in str(maintenance.get("body", "")).lower()
        or not all(
            device.get("admin_state") == "disabled"
            for device in maintenance.get("devices", {}).get("devices", [])
        )
    ):
        fail("all-disabled maintenance evidence is invalid")

    restart = require_evidence_version("restart-persistence", values["restart-persistence"])
    if (
        restart.get("old_pid") != pid
        or not isinstance(restart.get("new_pid"), int)
        or restart.get("new_pid") == pid
        or restart.get("before_mapping") != api_mapping
        or restart.get("after_mapping") != api_mapping
        or restart.get("disabled_uuid") != api_mapping.get(restart.get("disabled_id"))
    ):
        fail("restart persistence is not exact-PID/stable-ID/UUID bound")
    persisted = {
        device.get("id"): device
        for device in restart.get("persisted_devices", {}).get("devices", [])
    }
    if persisted.get(restart.get("disabled_id"), {}).get("desired_enabled") is not False:
        fail("disabled preference did not persist across restart")
    if not all(
        device.get("admin_state") == "enabled"
        for device in restart.get("restored_devices", {}).get("devices", [])
    ):
        fail("persisted device was not restored")

    legacy = require_evidence_version("legacy-rollback", values["legacy-rollback"])
    legacy_caps = legacy.get("snapshot", {}).get("capabilities", {})
    if (
        legacy.get("device_mapping") != api_mapping
        or legacy.get("patch_status") != 409
        or legacy_caps.get("dispatch", {}).get("active_mode") != "legacy"
        or legacy_caps.get("dispatch", {}).get("v2_authoritative")
        or legacy_caps.get("devices", {}).get("lifecycle")
    ):
        fail("legacy rollback evidence is invalid")

    selector = require_evidence_version("selector-matrix", values["selector-matrix"])
    for scenario in selector.get("scenarios", []):
        validate_sandbox_argv(
            f"selector {scenario.get('label')}",
            scenario.get("argv"),
            binary_path,
        )
    labels = {scenario.get("label") for scenario in selector.get("scenarios", [])}
    required_labels = {
        "empty",
        "all",
        "ordinal-one",
        "stable-id",
        "nvidia-uuid",
        "none",
        "reordered-stable",
        "missing",
    }
    if labels != required_labels:
        fail("selector matrix is incomplete")
    selector_pids = [
        scenario.get("pid")
        for scenario in selector.get("scenarios", [])
        if scenario.get("label") != "missing"
    ]
    if (
        len(selector_pids) != 7
        or not all(isinstance(value, int) and value > 0 for value in selector_pids)
        or len(set(selector_pids)) != len(selector_pids)
    ):
        fail("selector scenarios are not bound to unique exact candidate PIDs")
    contract = require_evidence_version(
        "ambiguous-selector-source-contract",
        values["ambiguous-selector-source-contract"],
    )
    validate_sandbox_argv("ambiguous selector source contract", contract.get("argv"))
    if contract.get("exit_code") != 0 or contract.get("hardware_claimed") is not False:
        fail("ambiguous selector source contract is invalid")

    if values["models-tree-before"] != values["models-tree-after"]:
        fail("models tree changed during qualification")
    for name, labels in CHECK_EVIDENCE.items():
        if set(report["checks"][name]["evidence_labels"]) != labels:
            fail(f"check {name} does not cite its exact mandatory evidence")
    for label in (
        "primary-command",
        "restart-command",
        "legacy-command",
    ):
        command = values[label]
        if not isinstance(command, dict):
            fail(f"candidate command {label} was not Bubblewrap isolated")
        validate_sandbox_argv(label, command.get("argv"), binary_path)


def validate(report_path: pathlib.Path, require_passing: bool) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_against_schema(report)
    if report.get("schema_version") != SCHEMA_VERSION:
        fail(f"unsupported schema_version: {report.get('schema_version')!r}")
    if report.get("qualification_profile") != QUALIFICATION_PROFILE:
        fail("qualification_profile is not the exact local 2x RTX 3090 profile")
    if not COMMIT_RE.fullmatch(report.get("source_commit", "")):
        fail("source_commit is not an exact Git commit")
    started = dt.datetime.fromisoformat(report["started_at"].replace("Z", "+00:00"))
    finished = dt.datetime.fromisoformat(report["finished_at"].replace("Z", "+00:00"))
    if finished < started:
        fail("finished_at precedes started_at")

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

    paths, values = load_evidence(report_path, report.get("evidence"))
    for name, check in checks.items():
        unknown = set(check["evidence_labels"]) - set(paths)
        if unknown:
            fail(f"check {name} references unknown evidence labels: {sorted(unknown)}")

    candidate = report.get("candidate", {})
    binary = pathlib.Path(candidate.get("path", ""))
    if not binary.is_file() or sha256(binary) != candidate.get("sha256"):
        fail("candidate binary no longer exists or does not match the report")
    server_pid = candidate.get("server_pid")
    if server_pid is not None and (
        not isinstance(server_pid, int)
        or isinstance(server_pid, bool)
        or server_pid <= 0
    ):
        fail("candidate.server_pid must be null or identify an exact qualification process")
    port = report.get("isolation", {}).get("port")
    if (
        not isinstance(port, int)
        or isinstance(port, bool)
        or not (1024 <= port <= 65535)
        or port == 7680
    ):
        fail("isolation.port must be non-privileged and must not be reserved port 7680")

    qualified = report.get("hardware_qualified")
    all_passed = all(check["status"] == "passed" for check in checks.values())
    if qualified is not all_passed:
        fail("hardware_qualified must equal the conjunction of all required checks")
    if qualified and server_pid is None:
        fail("passing qualification requires the exact candidate server PID")
    if qualified:
        validate_passing_evidence(report, report_path, paths, values)
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
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"invalid local multi-GPU report: {error}", file=sys.stderr)
        return 1
    print(
        f"validated {report['schema_version']}: "
        f"hardware_qualified={str(report['hardware_qualified']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
