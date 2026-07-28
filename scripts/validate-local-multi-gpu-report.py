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
import posixpath
import re
import stat
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
    "live_service_unchanged",
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
    "live_service_unchanged": {"live-service-before", "live-service-after"},
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
    "live-service-before",
    "live-service-after",
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


def canonical_path(
    label: str,
    value: object,
    *,
    require_exists: bool = False,
    require_directory: bool = False,
    require_regular: bool = False,
    require_char_device: bool = False,
) -> pathlib.Path:
    if not isinstance(value, str) or not value.startswith("/"):
        fail(f"{label} must be an absolute path")
    if "\x00" in value or (value != "/" and "//" in value):
        fail(f"{label} has empty or ambiguous path components")
    components = [] if value == "/" else value.split("/")[1:]
    if any(component in {"", ".", ".."} for component in components):
        fail(f"{label} is not a normalized lexical path")
    if posixpath.normpath(value) != value:
        fail(f"{label} is not a canonical lexical path")
    path = pathlib.Path(value)
    current = pathlib.Path("/")
    for component in components:
        current /= component
        if current.is_symlink():
            fail(f"{label} contains a symlink alias: {current}")
    if require_exists and not path.exists():
        fail(f"{label} does not exist")
    if require_directory and not path.is_dir():
        fail(f"{label} is not a directory")
    if require_regular and (
        not path.is_file() or not stat.S_ISREG(path.stat().st_mode)
    ):
        fail(f"{label} is not a regular file")
    if require_char_device and (
        not path.exists() or not stat.S_ISCHR(path.stat().st_mode)
    ):
        fail(f"{label} is not an actual character device")
    return path


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


def embedded_git_sha_matches_source(embedded: object, source_commit: object) -> bool:
    if not isinstance(embedded, str) or not isinstance(source_commit, str):
        return False
    if len(source_commit) != 40 or not COMMIT_RE.fullmatch(source_commit):
        return False
    return (
        7 <= len(embedded) <= len(source_commit)
        and bool(re.fullmatch(r"[0-9a-f]+", embedded))
        and source_commit.startswith(embedded)
    )


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
        path = canonical_path(
            f"evidence {label} path",
            item["path"],
            require_exists=True,
            require_regular=True,
        )
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
    expected_prefix = [
        "bwrap",
        "--die-with-parent",
        "--unshare-user",
        "--disable-userns",
        "--unshare-pid",
        "--unshare-ipc",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/",
        "/",
    ]
    if argv[: len(expected_prefix)] != expected_prefix:
        fail(f"{label} did not use the exact read-only-root/PID Bubblewrap prefix")
    try:
        separator = argv.index("--")
    except ValueError:
        fail(f"{label} Bubblewrap argv lacks a command separator")
    if argv.count("--") != 1:
        fail(f"{label} Bubblewrap argv has ambiguous command separators")
    options = argv[:separator]
    root_mounts = [
        options[index : index + 3]
        for index in range(len(options) - 2)
        if options[index]
        in {"--ro-bind", "--ro-bind-try", "--bind", "--bind-try"}
        and options[index + 2] == "/"
    ]
    if root_mounts != [["--ro-bind", "/", "/"]]:
        fail(f"{label} has a writable, duplicate, or ambiguous root mount")
    if any(option in options for option in ("--overlay", "--tmp-overlay")):
        fail(f"{label} uses a writable root overlay")
    if "--dev-bind" in options and any(
        options[index : index + 3] == ["--dev-bind", "/dev", "/dev"]
        for index in range(len(options) - 2)
    ):
        fail(f"{label} exposes the entire host /dev writable")
    cursor = len(expected_prefix)
    stages: list[int] = []
    binds: list[list[str]] = []
    directories: list[str] = []
    device_binds: list[list[str]] = []
    tmpfs: list[str] = []
    dev_count = 0
    proc_count = 0
    while cursor < separator:
        option = options[cursor]
        if option == "--dev" and cursor + 1 < separator:
            dev_count += 1
            stages.append(0)
            if options[cursor + 1] != "/dev":
                fail(f"{label} uses an unexpected private device mount")
            cursor += 2
        elif option == "--proc" and cursor + 1 < separator:
            proc_count += 1
            stages.append(1)
            if options[cursor + 1] != "/proc":
                fail(f"{label} uses an unexpected proc mount")
            cursor += 2
        elif option == "--bind" and cursor + 2 < separator:
            stages.append(2)
            binds.append(options[cursor : cursor + 3])
            cursor += 3
        elif option == "--dir" and cursor + 1 < separator:
            stages.append(3)
            directories.append(options[cursor + 1])
            cursor += 2
        elif option == "--dev-bind-try" and cursor + 2 < separator:
            stages.append(4)
            device_binds.append(options[cursor : cursor + 3])
            cursor += 3
        elif option == "--tmpfs" and cursor + 1 < separator:
            stages.append(5)
            tmpfs.append(options[cursor + 1])
            cursor += 2
        else:
            fail(f"{label} uses unsupported or malformed Bubblewrap option {option!r}")
    if stages != sorted(stages) or dev_count != 1 or proc_count != 1:
        fail(f"{label} Bubblewrap option ordering/count is not canonical")
    if len(binds) != 2:
        fail(f"{label} must expose exactly one runtime and its /tmp")
    runtime_source = canonical_path(
        f"{label} runtime bind source",
        binds[0][1],
        require_exists=True,
        require_directory=True,
    )
    runtime_destination = canonical_path(
        f"{label} runtime bind destination", binds[0][2]
    )
    tmp_source = canonical_path(
        f"{label} temporary bind source",
        binds[1][1],
        require_exists=True,
        require_directory=True,
    )
    tmp_destination = canonical_path(
        f"{label} temporary bind destination", binds[1][2]
    )
    runtime = runtime_source
    if (
        binds[0][1] != binds[0][2]
        or runtime_destination != runtime
        or tmp_source != runtime / "tmp"
        or tmp_destination != pathlib.Path("/tmp")
    ):
        fail(f"{label} writable mounts do not match the isolated runtime")
    canonical_directories = [
        str(
            canonical_path(
                f"{label} device directory",
                directory,
                require_exists=True,
                require_directory=True,
            )
        )
        for directory in directories
    ]
    if len(canonical_directories) != len(set(canonical_directories)) or not set(
        canonical_directories
    ) <= {
        "/dev/nvidia-caps",
        "/dev/dri",
    }:
        fail(f"{label} creates unexpected device directories")
    for _, source, destination in device_binds:
        source_path = canonical_path(
            f"{label} device source",
            source,
            require_exists=True,
            require_char_device=True,
        )
        destination_path = canonical_path(f"{label} device destination", destination)
        allowed = (
            source_path.parent == pathlib.Path("/dev")
            and source_path.name.startswith("nvidia")
        ) or source_path.parent in {
            pathlib.Path("/dev/nvidia-caps"),
            pathlib.Path("/dev/dri"),
        }
        if source != destination or source_path != destination_path or not allowed:
            fail(f"{label} exposes an unexpected host device")
    if len(tmpfs) > 1 or any(
        not re.fullmatch(
            r"/run/user/[0-9]+",
            str(canonical_path(f"{label} runtime mask", destination)),
        )
        for destination in tmpfs
    ):
        fail(f"{label} masks an unexpected runtime path")
    if candidate_path is not None:
        if separator + 1 >= len(argv):
            fail(f"{label} Bubblewrap argv lacks the candidate command")
        candidate_command = canonical_path(
            f"{label} candidate command",
            argv[separator + 1],
            require_exists=True,
            require_regular=True,
        )
        if candidate_command != candidate_path or argv[separator + 1] != str(
            candidate_path
        ):
            fail(f"{label} Bubblewrap argv does not execute the exact candidate")


def validate_isolation_environment(
    label: str,
    value: object,
    *,
    report: dict,
    evidence_root: pathlib.Path,
    runtime: pathlib.Path,
    expected_runtime: pathlib.Path,
    client: bool = False,
) -> None:
    if not isinstance(value, dict):
        fail(f"{label} environment evidence is missing")
    inherited_home = report["isolation"]["inherited_home"]
    if value.get("HOME") != inherited_home:
        fail(f"{label} did not preserve the exact inherited HOME")
    expected_host = (
        f"http://127.0.0.1:{report['isolation']['port']}" if client else None
    )
    if value.get("MOLD_HOST") != expected_host:
        fail(f"{label} environment targets an unexpected host or reserved service")
    validate_role_runtime(label, runtime, evidence_root, expected_runtime)
    expected_paths = {
        "TMPDIR": runtime / "tmp",
        "XDG_CACHE_HOME": runtime / "cache" / "xdg",
        "XDG_CONFIG_HOME": runtime / "config",
        "XDG_DATA_HOME": runtime / "data",
        "CUDA_CACHE_PATH": runtime / "cache" / "cuda",
        "HF_HOME": runtime / "cache" / "huggingface",
        "MOLD_HOME": runtime / "home",
        "MOLD_DB_PATH": runtime / "mold.db",
        "MOLD_OUTPUT_DIR": runtime / "output",
    }
    for key, expected_path in expected_paths.items():
        observed_path = canonical_path(
            f"{label} environment {key}",
            value.get(key),
            require_exists=key
            in {
                "TMPDIR",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "CUDA_CACHE_PATH",
                "HF_HOME",
                "MOLD_HOME",
                "MOLD_OUTPUT_DIR",
            },
        )
        if observed_path != expected_path:
            fail(f"{label} environment {key} is not the exact isolated path")
        try:
            expected_path.relative_to(evidence_root)
        except ValueError:
            fail(f"{label} runtime escapes the evidence directory")


def validate_probe_environment(
    label: str,
    value: object,
    *,
    inherited_home: str,
    runtime: pathlib.Path,
    evidence_root: pathlib.Path,
    expected_runtime: pathlib.Path,
) -> None:
    required_keys = {
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "MOLD_HOST",
    }
    if not isinstance(value, dict) or set(value) != required_keys:
        fail(f"{label} environment is not the exact recorded probe environment")
    if value["HOME"] != inherited_home or value["MOLD_HOST"] is not None:
        fail(f"{label} inherited HOME or host target is invalid")
    validate_role_runtime(label, runtime, evidence_root, expected_runtime)
    expected = {
        "TMPDIR": runtime / "tmp",
        "XDG_CACHE_HOME": runtime / "cache" / "xdg",
        "XDG_CONFIG_HOME": runtime / "config",
        "XDG_DATA_HOME": runtime / "data",
    }
    for key, expected_path in expected.items():
        observed = canonical_path(
            f"{label} environment {key}",
            value[key],
            require_exists=True,
            require_directory=True,
        )
        if observed != expected_path:
            fail(f"{label} environment {key} is not the exact probe path")


def validate_role_runtime(
    label: str,
    runtime: pathlib.Path,
    evidence_root: pathlib.Path,
    expected_runtime: pathlib.Path,
) -> None:
    observed = canonical_path(
        f"{label} writable runtime",
        str(runtime),
        require_exists=True,
        require_directory=True,
    )
    root = canonical_path(
        f"{label} evidence root",
        str(evidence_root),
        require_exists=True,
        require_directory=True,
    )
    expected = canonical_path(
        f"{label} expected runtime",
        str(expected_runtime),
        require_exists=True,
        require_directory=True,
    )
    try:
        observed.relative_to(root)
        expected.relative_to(root)
    except ValueError:
        fail(f"{label} writable runtime escapes the exact evidence root")
    if observed != expected:
        fail(f"{label} writable runtime is not the exact role-specific directory")


def sandbox_runtime(argv: list[str]) -> pathlib.Path:
    separator = argv.index("--")
    options = argv[:separator]
    for index in range(len(options) - 2):
        if options[index] == "--bind" and options[index + 1] == options[index + 2]:
            return canonical_path(
                "Bubblewrap runtime",
                options[index + 1],
                require_exists=True,
                require_directory=True,
            )
    fail("Bubblewrap argv lacks its exact writable runtime mount")


def sandbox_inner(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1 :]


def validate_process_identity(
    label: str,
    value: object,
    *,
    pid: int,
    binary_path: pathlib.Path,
    binary_sha256: str,
) -> dict:
    binary_stat = binary_path.stat()
    if not isinstance(value, dict):
        fail(f"{label} process identity is missing")
    if (
        value.get("pid") != pid
        or value.get("executable") != str(binary_path)
        or value.get("binary_sha256") != binary_sha256
        or not isinstance(value.get("start_ticks"), int)
        or value.get("start_ticks") <= 0
        or not isinstance(value.get("executable_device"), int)
        or not isinstance(value.get("executable_inode"), int)
        or value.get("executable_device") != binary_stat.st_dev
        or value.get("executable_inode") != binary_stat.st_ino
        or not re.fullmatch(r"pid:\[[0-9]+\]", str(value.get("pid_namespace")))
        or not isinstance(value.get("nspid"), list)
        or len(value["nspid"]) < 2
        or value["nspid"][0] != pid
        or any(
            not isinstance(namespace_pid, int) or namespace_pid <= 0
            for namespace_pid in value["nspid"]
        )
        or value["nspid"][-1] == pid
    ):
        fail(f"{label} process identity is not exact PID/start/executable/namespace bound")
    canonical_path(
        f"{label} process executable",
        value["executable"],
        require_exists=True,
        require_regular=True,
    )
    return value


def validate_live_service_snapshot(label: str, value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"port", "listeners", "owners"}:
        fail(f"{label} live-service snapshot has an invalid shape")
    if value["port"] != 7680 or not isinstance(value["listeners"], list) or not isinstance(
        value["owners"], list
    ):
        fail(f"{label} live-service snapshot has invalid listener fields")
    listener_inodes: set[str] = set()
    for row in value["listeners"]:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"family", "local_port", "remote_port", "state", "inode"}
            or row["family"] not in {"tcp", "tcp6"}
            or row["local_port"] != 7680
            or row["remote_port"] != 0
            or row["state"] != "0A"
            or not isinstance(row["inode"], str)
            or not row["inode"].isdigit()
            or row["inode"] in listener_inodes
        ):
            fail(f"{label} live-service listener row is invalid")
        listener_inodes.add(row["inode"])
    owned_inodes: set[str] = set()
    for row in value["owners"]:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"pid", "start_ticks", "executable", "socket_inodes"}
            or not isinstance(row["pid"], int)
            or row["pid"] <= 0
            or not isinstance(row["start_ticks"], int)
            or row["start_ticks"] <= 0
            or not isinstance(row["socket_inodes"], list)
        ):
            fail(f"{label} live-service owner row is invalid")
        canonical_path(f"{label} live-service executable", row["executable"])
        if any(
            not isinstance(inode, str) or not inode.isdigit()
            for inode in row["socket_inodes"]
        ):
            fail(f"{label} live-service owner socket inode is invalid")
        row_inodes = set(row["socket_inodes"])
        if (
            len(row_inodes) != len(row["socket_inodes"])
            or not row_inodes
            or not row_inodes <= listener_inodes
            or row_inodes & owned_inodes
        ):
            fail(f"{label} live-service owner/socket relationship is invalid")
        owned_inodes.update(row_inodes)
    if bool(listener_inodes) != bool(value["owners"]) or owned_inodes != listener_inodes:
        fail(f"{label} live-service listeners are not exactly owner-bound")
    return value


def validate_passing_evidence(
    report: dict,
    report_path: pathlib.Path,
    paths: dict[str, pathlib.Path],
    values: dict[str, object],
) -> None:
    evidence_root = canonical_path(
        "evidence root",
        str(report_path) + ".d",
        require_exists=True,
        require_directory=True,
    )
    missing = MANDATORY_EVIDENCE - set(paths)
    if missing:
        fail(f"passing report is missing mandatory typed evidence: {sorted(missing)}")

    expected = report["host"]["expected_gpu_uuids"]
    validate_hardware_profile(report["host"]["devices"], expected)
    inventory = parse_inventory(str(values["nvidia-inventory"]))
    if inventory != report["host"]["devices"]:
        fail("typed nvidia inventory does not exactly match report host devices")

    request_path = canonical_path(
        "request path",
        report["request"]["path"],
        require_exists=True,
        require_regular=True,
    )
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
    artifact_roots = report["request"]["artifact_roots"]
    if not artifacts:
        fail("passing qualification requires exact model artifacts")
    if not artifact_roots:
        fail("passing qualification requires resolved companion artifact roots")
    models_dir = canonical_path(
        "models_dir",
        report["isolation"]["models_dir"],
        require_exists=True,
        require_directory=True,
    )
    resolved_roots: list[pathlib.Path] = []
    for raw_root in artifact_roots:
        artifact_root = canonical_path(
            "artifact root",
            raw_root,
            require_exists=True,
            require_directory=True,
        )
        try:
            artifact_root.relative_to(models_dir)
        except ValueError:
            fail(f"artifact root escapes models_dir: {artifact_root}")
        if not artifact_root.is_dir():
            fail(f"artifact root no longer exists: {artifact_root}")
        resolved_roots.append(artifact_root)
    actual_paths = sorted(
        {
            path.resolve()
            for artifact_root in resolved_roots
            for path in artifact_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        },
        key=str,
    )
    reported_paths = sorted(
        (
            canonical_path(
                "artifact path",
                artifact["path"],
                require_exists=True,
                require_regular=True,
            )
            for artifact in artifacts
        ),
        key=str,
    )
    if actual_paths != reported_paths:
        fail("reported artifacts are not the complete resolved companion inventory")
    for artifact in artifacts:
        path = canonical_path(
            "artifact path",
            artifact["path"],
            require_exists=True,
            require_regular=True,
        )
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
        or artifact_evidence.get("artifact_roots") != artifact_roots
    ):
        fail("model-artifacts evidence is not exact")

    source = require_evidence_version("source-provenance", values["source-provenance"])
    if source.get("commit") != report["source_commit"]:
        fail("source provenance does not match report commit")
    source_root = canonical_path(
        "source provenance root",
        source.get("source_root"),
        require_exists=True,
        require_directory=True,
    )
    version = require_evidence_version("candidate-version", values["candidate-version"])
    binary_path = canonical_path(
        "candidate path",
        report["candidate"]["path"],
        require_exists=True,
        require_regular=True,
    )
    validate_sandbox_argv("candidate version", version.get("argv"), binary_path)
    if sandbox_inner(version["argv"]) != [str(binary_path), "version"]:
        fail("candidate version invocation has unexpected inner arguments")
    validate_probe_environment(
        "candidate version",
        version.get("environment"),
        inherited_home=report["isolation"]["inherited_home"],
        runtime=sandbox_runtime(version["argv"]),
        evidence_root=evidence_root,
        expected_runtime=evidence_root / "candidate-probe-runtime",
    )
    build_identity = version.get("build_identity")
    if (
        version.get("binary_sha256") != report["candidate"]["sha256"]
        or version.get("binary") != str(binary_path)
        or version.get("version") != report["candidate"]["version"]
        or version.get("sandboxed") is not True
        or build_identity
        != {
            "source_commit": report["source_commit"],
            "binary_sha256": report["candidate"]["sha256"],
            "version": report["candidate"]["version"],
        }
    ):
        fail("candidate version evidence is not sandboxed and binary-bound")

    initial = values["initial-api-projection"]
    if not isinstance(initial, dict):
        fail("initial API projection must be an object")
    if initial.get("status", {}).get("hostname") != report["host"]["hostname"]:
        fail("server status hostname does not match report host identity")
    if not embedded_git_sha_matches_source(
        initial.get("status", {}).get("git_sha"), report["source_commit"]
    ):
        fail("candidate embedded git SHA does not match source provenance")
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
    if sandbox_inner(client["argv"]) != [
        str(binary_path),
        "gpu",
        "list",
        "--json",
    ]:
        fail("client projection invocation has unexpected inner arguments")
    validate_isolation_environment(
        "client projection",
        client.get("environment"),
        report=report,
        evidence_root=evidence_root,
        runtime=sandbox_runtime(client["argv"]),
        expected_runtime=evidence_root / "primary-runtime",
        client=True,
    )
    if client.get("server_pid") != report["candidate"]["server_pid"]:
        fail("client projection is not bound to the primary server PID")
    client_ids = [row.get("id") for row in client.get("gpu_list", {}).get("devices", [])]
    if client_ids != [device.get("id") for device in api_devices]:
        fail("client projection IDs differ from the API")

    parallel = require_evidence_version("parallel-results", values["parallel-results"])
    pid = report["candidate"]["server_pid"]
    if parallel.get("server_pid") != pid:
        fail("parallel evidence is not bound to the exact primary PID")
    primary_identity = validate_process_identity(
        "parallel candidate",
        parallel.get("candidate_identity"),
        pid=pid,
        binary_path=binary_path,
        binary_sha256=report["candidate"]["sha256"],
    )
    raw_samples = values["parallel-runtime-samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        fail("parallel runtime samples are empty")
    raw_bindings: dict[str, str] = {}
    raw_active_uuids: set[str] = set()
    raw_compute_uuids: set[str] = set()
    raw_decisive: list[dict[str, object]] = []
    raw_times: list[dt.datetime] = []
    raw_plan_versions: list[int] = []
    for sample in raw_samples:
        if not isinstance(sample, dict) or sample.get("server_pid") != pid:
            fail("raw runtime sample is not bound to the exact candidate PID")
        if sample.get("candidate_identity") != primary_identity:
            fail("raw runtime sample has a different candidate process identity")
        try:
            sampled_at = dt.datetime.fromisoformat(
                sample["at"].replace("Z", "+00:00")
            )
        except (KeyError, AttributeError, ValueError):
            fail("raw runtime sample has an invalid timestamp")
        plan_version = (sample.get("queue", {}).get("plan") or {}).get(
            "plan_version"
        )
        if sampled_at.utcoffset() is None:
            fail("raw runtime sample timestamp lacks a timezone")
        if (
            not isinstance(plan_version, int)
            or isinstance(plan_version, bool)
            or plan_version < 0
        ):
            fail("raw runtime sample omits an exact plan version")
        raw_times.append(sampled_at)
        raw_plan_versions.append(plan_version)
        if sample.get("reserved_service_connections") != []:
            fail("candidate sampled a connection to the reserved live service")
        devices_in_sample = sample.get("devices")
        compute_in_sample = sample.get("compute_apps")
        if not isinstance(devices_in_sample, list) or not isinstance(
            compute_in_sample, list
        ):
            fail("raw runtime sample lacks typed device/compute observations")
        exact_compute = {
            row.get("gpu_uuid")
            for row in compute_in_sample
            if isinstance(row, dict) and row.get("pid") == pid
        }
        raw_compute_uuids.update(
            value for value in exact_compute if isinstance(value, str)
        )
        active = [
            device
            for device in devices_in_sample
            if isinstance(device, dict)
            and device.get("active_work_id")
            and device.get("activity") not in {"idle", "stopping"}
        ]
        for device in active:
            work_id = device.get("active_work_id")
            gpu_uuid = device.get("nvml_uuid")
            if gpu_uuid not in exact_compute:
                continue
            if (
                not isinstance(work_id, str)
                or not isinstance(gpu_uuid, str)
                or gpu_uuid not in expected
            ):
                fail("raw active work has invalid work-ID/GPU identity")
            existing = raw_bindings.get(work_id)
            if existing is not None and existing != gpu_uuid:
                fail("raw samples move an active work ID across physical GPUs")
            raw_bindings[work_id] = gpu_uuid
            raw_active_uuids.add(gpu_uuid)
        active_projection = [
            {
                "device_id": device.get("id"),
                "gpu_uuid": device.get("nvml_uuid"),
                "work_id": device.get("active_work_id"),
            }
            for device in active
        ]
        if (
            {row["gpu_uuid"] for row in active_projection} == set(expected)
            and len({row["work_id"] for row in active_projection}) == 2
            and set(expected) <= exact_compute
        ):
            raw_decisive.append(
                {
                    "server_pid": pid,
                    "active": active_projection,
                    "compute_apps": [
                        row
                        for row in compute_in_sample
                        if isinstance(row, dict) and row.get("pid") == pid
                    ],
                }
            )
    if any(
        later <= earlier for earlier, later in zip(raw_times, raw_times[1:])
    ):
        fail("raw runtime sample timeline is not strictly increasing")
    if any(
        later < earlier
        for earlier, later in zip(raw_plan_versions, raw_plan_versions[1:])
    ):
        fail("raw runtime plan-version timeline regressed")
    if set(parallel.get("observed_active_uuids", [])) != raw_active_uuids:
        fail("parallel active UUID summary does not derive from raw samples")
    if set(parallel.get("observed_compute_uuids", [])) != raw_compute_uuids:
        fail("parallel compute UUID summary does not derive from raw samples")
    if not raw_decisive or parallel.get("decisive_samples") != raw_decisive:
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
        expected_prompt = f"{request['prompt']} variation {index}"
        expected_seed = int(request.get("seed", 0)) + index - 1
        headers = output.get("headers")
        if (
            output.get("prompt") != expected_prompt
            or output.get("seed") != expected_seed
            or not isinstance(headers, dict)
            or headers.get("x-mold-seed-used") != str(expected_seed)
            or headers.get("x-mold-gpu") != str(output.get("gpu_ordinal"))
        ):
            fail(f"parallel output {index} is inconsistent with the request/result headers")
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
        if raw_bindings.get(work_id) != gpu_uuid:
            fail("output work ID is not present in raw exact-PID/GPU samples")
        output_uuids.add(gpu_uuid)
    if output_uuids != set(expected):
        fail("validated outputs do not cover both exact GPU UUIDs")
    if set(range(1, report["request"]["job_count"] + 1)) != {
        output.get("index") for output in outputs
    }:
        fail("parallel output indices are not the exact requested sequence")
    if execution_bindings != {
        work_id: output["gpu_uuid"] for work_id, output in output_by_work.items()
    }:
        fail("execution binding summary contains ghost or missing work IDs")

    disabled_id = parallel.get("disabled_id")
    disabled_uuid = parallel.get("disabled_uuid")
    pre = parallel.get("pre_disable_assignments")
    post = parallel.get("post_disable_assignments")
    if (
        disabled_id not in api_mapping
        or disabled_uuid != api_mapping.get(disabled_id)
        or not isinstance(pre, dict)
        or not pre
        or not isinstance(post, dict)
        or set(pre) != set(post)
        or set(parallel.get("replanned_work_ids", [])) != set(pre)
        or any(value != disabled_id for value in pre.values())
        or any(value == disabled_id for value in post.values())
    ):
        fail("same queued work IDs were not proven to move after disable")
    raw_plans = [
        {
            str(item.get("work_id")): str(item.get("planned_device_id"))
            for item in (sample.get("queue", {}).get("plan") or {}).get(
                "work_items", []
            )
            if isinstance(item, dict)
            and item.get("work_id")
            and item.get("planned_device_id")
        }
        for sample in raw_samples
    ]
    pre_indices = [
        index
        for index, plan in enumerate(raw_plans)
        if all(plan.get(work_id) == lane for work_id, lane in pre.items())
    ]
    post_indices = [
        index
        for index, plan in enumerate(raw_plans)
        if all(plan.get(work_id) == lane for work_id, lane in post.items())
    ]
    if not pre_indices:
        fail("disabled lane pre-plan is not present in raw queue samples")
    if not post_indices:
        fail("disabled lane post-plan is not present in raw queue samples")
    chronological_replan = next(
        (
            (pre_index, post_index)
            for pre_index in pre_indices
            for post_index in post_indices
            if post_index > pre_index
            and raw_plan_versions[post_index] > raw_plan_versions[pre_index]
        ),
        None,
    )
    if chronological_replan is None:
        fail("queue replan is not a later, strictly advanced raw plan version")
    for work_id, device_id in post.items():
        output = output_by_work.get(work_id)
        if output is None or output.get("gpu_uuid") != api_mapping.get(device_id):
            fail("replanned work output did not execute on its new exact UUID")
    if (
        parallel.get("disable_response", {}).get("status") != 202
        or parallel.get("disable_response", {}).get("body", {}).get("id")
        != disabled_id
        or parallel.get("disable_response", {}).get("body", {}).get("nvml_uuid")
        != disabled_uuid
        or parallel.get("disable_response", {}).get("body", {}).get(
            "admin_state"
        )
        != "draining"
        or parallel.get("drained", {}).get("id") != disabled_id
        or parallel.get("drained", {}).get("nvml_uuid") != disabled_uuid
        or parallel.get("drained", {}).get("admin_state") != "disabled"
        or parallel.get("reenabled", {}).get("id") != disabled_id
        or parallel.get("reenabled", {}).get("nvml_uuid") != disabled_uuid
        or parallel.get("reenabled", {}).get("admin_state") != "enabled"
    ):
        fail("busy drain/disable/re-enable evidence is incomplete")

    cancellation = require_evidence_version(
        "queued-cancellation", values["queued-cancellation"]
    )
    cancellation_job = cancellation.get("job_id")
    cancellation_work = cancellation.get("work_id")
    pause_response = cancellation.get("pause_response")
    queue_before = cancellation.get("queue_before_cancel")
    devices_before = cancellation.get("devices_before_cancel")
    queue_after = cancellation.get("queue_after")
    devices_after = cancellation.get("devices_after")
    queue_terminal = cancellation.get("queue_after_cancel_before_resume")
    devices_terminal = cancellation.get("devices_after_cancel_before_resume")
    if not all(
        isinstance(value, dict)
        for value in (
            queue_before,
            devices_before,
            queue_terminal,
            devices_terminal,
            queue_after,
            devices_after,
        )
    ):
        fail("queued cancellation lacks typed before/after scheduler evidence")
    terminal_at = dt.datetime.fromisoformat(
        cancellation["terminal_observed_at"].replace("Z", "+00:00")
    )
    resume_at = dt.datetime.fromisoformat(
        cancellation["resume_at"].replace("Z", "+00:00")
    )
    cancellation_request = cancellation.get("request")
    expected_cancel_request = dict(request)
    expected_cancel_request["prompt"] = f"{request['prompt']} queued cancellation"
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
    terminal_queued = any(
        entry.get("id") == cancellation_job
        for entry in queue_terminal.get("entries", [])
        if isinstance(entry, dict)
    )
    terminal_planned = any(
        item.get("work_id") == cancellation_work
        for item in (queue_terminal.get("plan") or {}).get("work_items", [])
        if isinstance(item, dict)
    )
    terminal_active = any(
        device.get("active_work_id") == cancellation_work
        for device in devices_terminal.get("devices", [])
        if isinstance(device, dict)
    )
    if (
        cancellation.get("server_pid") != pid
        or not isinstance(pause_response, dict)
        or pause_response.get("status") != 200
        or not isinstance(pause_response.get("body"), dict)
        or pause_response["body"].get("paused") is not True
        or cancellation.get("cancel_status") != 204
        or cancellation.get("resume_status") != 200
        or cancellation.get("stream_http_status") != 200
        or cancellation.get("typed_cancelled") is not True
        or cancellation.get("never_active") is not True
        or cancellation.get("output_tree_unchanged") is not True
        or cancellation.get("queue_was_paused") is not True
        or not cancellation_job
        or not cancellation_work
        or len(before_entries) != 1
        or before_entries[0].get("state") != "queued"
        or len(before_work) != 1
        or before_work[0].get("activity_phase") != "blocked"
        or before_work[0].get("blocked_reason") != "queue_paused"
        or before_active
        or after_queued
        or after_planned
        or after_active
        or not isinstance(output_before, list)
        or output_after != output_before
        or not has_typed_cancelled_sse(cancellation.get("stream_tail"))
        or terminal_at > resume_at
        or cancellation_request != expected_cancel_request
        or terminal_queued
        or terminal_planned
        or terminal_active
    ):
        fail("queued cancellation lacks typed no-inference proof")

    maintenance = require_evidence_version(
        "all-disabled-maintenance", values["all-disabled-maintenance"]
    )
    maintenance_devices = maintenance.get("devices", {}).get("devices", [])
    maintenance_mapping = {
        device.get("id"): device.get("nvml_uuid")
        for device in maintenance_devices
        if isinstance(device, dict)
    }
    if (
        maintenance.get("server_pid") != pid
        or maintenance.get("status") not in {409, 503}
        or "maintenance" not in str(maintenance.get("body", "")).lower()
        or len(maintenance_devices) != 2
        or maintenance_mapping != api_mapping
        or not all(
            device.get("admin_state") == "disabled"
            for device in maintenance_devices
        )
    ):
        fail("all-disabled maintenance device evidence is invalid")

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
    restart_old_identity = validate_process_identity(
        "restart old candidate",
        restart.get("old_process_identity"),
        pid=pid,
        binary_path=binary_path,
        binary_sha256=report["candidate"]["sha256"],
    )
    if restart_old_identity != primary_identity:
        fail("restart old process identity differs from primary evidence")
    validate_process_identity(
        "restart new candidate",
        restart.get("new_process_identity"),
        pid=restart["new_pid"],
        binary_path=binary_path,
        binary_sha256=report["candidate"]["sha256"],
    )
    persisted_devices = restart.get("persisted_devices", {}).get("devices", [])
    persisted = {
        device.get("id"): device
        for device in persisted_devices
        if isinstance(device, dict)
    }
    if (
        len(persisted_devices) != 2
        or {stable_id: row.get("nvml_uuid") for stable_id, row in persisted.items()}
        != api_mapping
        or persisted.get(restart.get("disabled_id"), {}).get("desired_enabled")
        is not False
        or any(
            row.get("desired_enabled") is not True
            for stable_id, row in persisted.items()
            if stable_id != restart.get("disabled_id")
        )
    ):
        fail("disabled preference did not persist across restart")
    restored_devices = restart.get("restored_devices", {}).get("devices", [])
    restored_mapping = {
        device.get("id"): device.get("nvml_uuid")
        for device in restored_devices
        if isinstance(device, dict)
    }
    if (
        len(restored_devices) != 2
        or restored_mapping != api_mapping
        or not all(
            device.get("desired_enabled") is True
            and device.get("admin_state") == "enabled"
            for device in restored_devices
        )
    ):
        fail("persisted device was not restored")

    legacy = require_evidence_version("legacy-rollback", values["legacy-rollback"])
    legacy_caps = legacy.get("snapshot", {}).get("capabilities", {})
    legacy_devices = legacy.get("snapshot", {}).get("devices", {}).get("devices", [])
    legacy_mapping = {
        device.get("id"): device.get("nvml_uuid")
        for device in legacy_devices
        if isinstance(device, dict)
    }
    if (
        legacy.get("device_mapping") != api_mapping
        or legacy_mapping != api_mapping
        or len(legacy_devices) != 2
        or any(
            device.get("admin_state") == "startup_excluded"
            for device in legacy_devices
        )
        or legacy.get("patch_target") not in api_mapping
        or legacy.get("patch_status") != 409
        or legacy.get("patch_body", {}).get("code")
        != "DEVICE_LIFECYCLE_MODE_CONFLICT"
        or "authoritative scheduler V2" not in legacy.get("patch_body", {}).get(
            "error", ""
        )
        or legacy_caps.get("dispatch", {}).get("active_mode") != "legacy"
        or legacy_caps.get("dispatch", {}).get("v2_authoritative")
        or legacy_caps.get("devices", {}).get("lifecycle")
    ):
        fail("legacy rollback evidence is invalid")
    validate_process_identity(
        "legacy candidate",
        legacy.get("process_identity"),
        pid=legacy.get("server_pid"),
        binary_path=binary_path,
        binary_sha256=report["candidate"]["sha256"],
    )

    selector = require_evidence_version("selector-matrix", values["selector-matrix"])
    scenarios = selector.get("scenarios", [])
    physical_order = [
        str(device["uuid"])
        for device in sorted(report["host"]["devices"], key=lambda row: int(row["index"]))
    ]
    stable_by_uuid = {gpu_uuid: stable_id for stable_id, gpu_uuid in api_mapping.items()}
    expected_selectors = {
        "empty": (None, set(expected)),
        "all": ("all", set(expected)),
        "ordinal-one": ("1", {physical_order[1]}),
        "stable-id": (stable_by_uuid[physical_order[0]], {physical_order[0]}),
        "nvidia-uuid": (physical_order[1], {physical_order[1]}),
        "none": ("none", set()),
        "reordered-stable": (
            stable_by_uuid[physical_order[0]],
            {physical_order[0]},
        ),
    }
    server_base_argv = [
        str(binary_path),
        "serve",
        "--bind",
        "127.0.0.1",
        "--port",
        str(report["isolation"]["port"]),
        "--models-dir",
        str(models_dir),
        "--queue-size",
        "64",
        "--log-format",
        "json",
    ]
    for scenario in scenarios:
        label = scenario.get("label")
        validate_sandbox_argv(
            f"selector {label}",
            scenario.get("argv"),
            binary_path,
        )
        validate_isolation_environment(
            f"selector {label}",
            scenario.get("environment"),
            report=report,
            evidence_root=evidence_root,
            runtime=sandbox_runtime(scenario["argv"]),
            expected_runtime=evidence_root / "selector-runtimes" / str(label),
        )
        if label == "missing":
            if (
                scenario.get("selector") != "GPU-ffffffff"
                or scenario.get("expected_uuids") != []
                or not isinstance(scenario.get("exit_code"), int)
                or scenario.get("exit_code") == 0
                or "did not match" not in scenario.get("expected_failure", "")
                or sandbox_inner(scenario["argv"])
                != [*server_base_argv, "--gpus", "GPU-ffffffff"]
            ):
                fail("missing selector scenario is not a typed failure")
            continue
        validate_process_identity(
            f"selector {label}",
            scenario.get("process_identity"),
            pid=scenario.get("pid"),
            binary_path=binary_path,
            binary_sha256=report["candidate"]["sha256"],
        )
        if label not in expected_selectors:
            fail(f"unknown selector scenario {label!r}")
        expected_selector, expected_selected = expected_selectors[label]
        if (
            scenario.get("selector") != expected_selector
            or set(scenario.get("expected_uuids", [])) != expected_selected
        ):
            fail(f"selector {label} input/expected UUID claim is wrong")
        devices_payload = scenario.get("devices")
        if not isinstance(devices_payload, dict):
            fail(f"selector {label} lacks device results")
        selector_devices = devices_payload.get("devices", [])
        selector_mapping = {
            device.get("id"): device.get("nvml_uuid")
            for device in selector_devices
            if isinstance(device, dict)
        }
        if len(selector_devices) != 2 or selector_mapping != api_mapping:
            fail(f"selector {label} omitted or changed qualification devices")
        selected = {
            device.get("nvml_uuid")
            for device in selector_devices
            if isinstance(device, dict)
            and device.get("admin_state") != "startup_excluded"
        }
        if selected != expected_selected:
            fail(f"selector {label} result UUIDs do not match its input")
        if any(
            (device.get("nvml_uuid") in expected_selected)
            == (device.get("admin_state") == "startup_excluded")
            for device in selector_devices
        ):
            fail(f"selector {label} exclusion states do not match its input")
        argv = scenario["argv"]
        inner = sandbox_inner(argv)
        expected_inner = list(server_base_argv)
        if expected_selector is not None:
            expected_inner.extend(["--gpus", expected_selector])
        if inner != expected_inner:
            fail(f"selector {label} command has unexpected inner arguments")
        if expected_selector is None:
            if "--gpus" in inner or scenario["environment"].get("MOLD_GPUS") != "":
                fail("empty selector did not exercise the explicit empty environment")
        else:
            try:
                gpu_flag = inner.index("--gpus")
            except ValueError:
                fail(f"selector {label} command omits --gpus")
            if inner[gpu_flag + 1] != expected_selector:
                fail(f"selector {label} command uses the wrong selector")
        if label == "reordered-stable":
            expected_visibility = f"{physical_order[1]},{physical_order[0]}"
            if scenario["environment"].get("CUDA_VISIBLE_DEVICES") != expected_visibility:
                fail("reordered selector did not reverse CUDA visibility")
    labels = {scenario.get("label") for scenario in scenarios}
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
        for scenario in scenarios
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
    if sandbox_inner(contract["argv"]) != [
        "/bin/bash",
        "-lc",
        contract.get("command"),
    ]:
        fail("ambiguous selector source contract command is not exact")
    if contract.get("exit_code") != 0 or contract.get("hardware_claimed") is not False:
        fail("ambiguous selector source contract is invalid")
    if (
        canonical_path(
            "ambiguous selector source contract cwd",
            contract.get("cwd"),
            require_exists=True,
            require_directory=True,
        )
        != source_root
    ):
        fail("ambiguous selector source contract used the wrong source root")
    validate_probe_environment(
        "ambiguous selector source contract",
        contract.get("environment"),
        inherited_home=report["isolation"]["inherited_home"],
        runtime=sandbox_runtime(contract["argv"]),
        evidence_root=evidence_root,
        expected_runtime=evidence_root / "selector-contract-runtime",
    )

    if values["models-tree-before"] != values["models-tree-after"]:
        fail("models tree changed during qualification")
    live_before = validate_live_service_snapshot(
        "live-service-before", values["live-service-before"]
    )
    live_after = validate_live_service_snapshot(
        "live-service-after", values["live-service-after"]
    )
    if live_before != live_after:
        fail("reserved live-service PID/listener identity changed")
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
        if sandbox_inner(command["argv"]) != server_base_argv:
            fail(f"candidate command {label} has unexpected inner arguments")
        validate_isolation_environment(
            label,
            command.get("environment"),
            report=report,
            evidence_root=evidence_root,
            runtime=sandbox_runtime(command["argv"]),
            expected_runtime=evidence_root / "primary-runtime",
        )
        environment = command["environment"]
        if (
            environment.get("MOLD_HOME") != report["isolation"]["mold_home"]
            or environment.get("MOLD_DB_PATH") != report["isolation"]["db_path"]
            or environment.get("MOLD_OUTPUT_DIR")
            != report["isolation"]["output_dir"]
        ):
            fail(f"candidate command {label} does not use the exact isolated paths")


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
    binary = canonical_path(
        "candidate path",
        candidate.get("path"),
        require_exists=True,
        require_regular=True,
    )
    if sha256(binary) != candidate.get("sha256"):
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
    isolation = report["isolation"]
    canonical_path("inherited HOME", isolation["inherited_home"])
    models_dir = canonical_path(
        "isolation models_dir",
        isolation["models_dir"],
        require_exists=True,
        require_directory=True,
    )
    runtime_evidence_root = canonical_path(
        "evidence root",
        str(report_path) + ".d",
        require_exists=True,
        require_directory=True,
    )
    for key in ("mold_home", "output_dir", "db_path"):
        path = canonical_path(
            f"isolation {key}",
            isolation[key],
        )
        try:
            path.relative_to(runtime_evidence_root)
        except ValueError:
            fail(f"isolation {key} escapes the evidence scope")
    if models_dir != canonical_path(
        "request models_dir",
        isolation["models_dir"],
        require_exists=True,
        require_directory=True,
    ):
        fail("isolation models_dir is unstable")

    qualified = report.get("hardware_qualified")
    all_passed = all(check["status"] == "passed" for check in checks.values())
    if qualified is not all_passed:
        fail("hardware_qualified must equal the conjunction of all required checks")
    if qualified and server_pid is None:
        fail("passing qualification requires the exact candidate server PID")
    if qualified:
        canonical_path(
            "isolation mold_home",
            isolation["mold_home"],
            require_exists=True,
            require_directory=True,
        )
        canonical_path(
            "isolation output_dir",
            isolation["output_dir"],
            require_exists=True,
            require_directory=True,
        )
        canonical_path(
            "isolation db_path",
            isolation["db_path"],
            require_exists=True,
            require_regular=True,
        )
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
