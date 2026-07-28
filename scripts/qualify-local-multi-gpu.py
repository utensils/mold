#!/usr/bin/env python3
"""Qualify one candidate Mold CUDA binary on a real local multi-GPU host.

The runner never touches an existing server: it refuses an occupied alternate
port and launches only on 127.0.0.1 with an isolated MOLD_HOME, MOLD_DB_PATH,
and MOLD_OUTPUT_DIR. Installed model weights may be read from --models-dir, but
their filesystem metadata is compared before and after the run.

An inventory whose UUIDs have no shared prefix cannot exercise an ambiguous
prefix in hardware. In that case this runner executes the repository's
deterministic selector contract and does not claim ambiguous-prefix hardware
evidence. Reordered CUDA visibility and UUID selection still use the real GPUs.
"""

from __future__ import annotations

import argparse
import binascii
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib

SCHEMA_VERSION = "mold.local.multi-gpu.qualification.v2"
EVIDENCE_SCHEMA_VERSION = "mold.local.multi-gpu.evidence.v2"
QUALIFICATION_PROFILE = "local-2x-rtx3090-sm86"
RTX3090_NAME = "NVIDIA GeForce RTX 3090"
RTX3090_MEMORY_MIB = 24576
RTX3090_COMPUTE_CAPABILITY = "8.6"
RESERVED_SERVICE_PORT = 7680
CHECK_NAMES = (
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
)
ENV_ALLOWLIST = {
    "HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_MODULE_LOADING",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "RUST_BACKTRACE",
    "RUSTUP_HOME",
    "CARGO_HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "LANG",
    "LC_ALL",
    "TZ",
}


class Deadline:
    def __init__(self, seconds: float):
        if seconds <= 0:
            raise ValueError("timeout must be positive")
        self.expires = time.monotonic() + seconds

    def remaining(self, maximum: float | None = None) -> float:
        value = self.expires - time.monotonic()
        if value <= 0:
            raise TimeoutError("qualification deadline expired")
        return value if maximum is None else min(value, maximum)

    def remaining_or_zero(self, maximum: float | None = None) -> float:
        value = max(0.0, self.expires - time.monotonic())
        return value if maximum is None else min(value, maximum)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: pathlib.Path, deadline: Deadline | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if deadline is not None:
                deadline.remaining()
            digest.update(chunk)
    return digest.hexdigest()


def validate_qualification_port(port: int) -> None:
    if not (1024 <= port <= 65535):
        raise ValueError("qualification port must be in 1024..65535")
    if port == RESERVED_SERVICE_PORT:
        raise ValueError("port 7680 is reserved for the normal Mold service")


def validate_hardware_profile(
    devices: list[dict[str, object]], expected_uuids: list[str]
) -> None:
    if len(devices) != 2 or len(expected_uuids) != 2:
        raise ValueError(f"{QUALIFICATION_PROFILE} requires exactly two GPUs")
    observed = [str(device.get("uuid")) for device in devices]
    if set(observed) != set(expected_uuids) or len(observed) != len(expected_uuids):
        raise ValueError(
            f"host inventory {observed!r} is not exact expected inventory "
            f"{expected_uuids!r}"
        )
    for device in devices:
        if device.get("name") != RTX3090_NAME:
            raise ValueError(
                f"{QUALIFICATION_PROFILE} requires {RTX3090_NAME}, got "
                f"{device.get('name')!r}"
            )
        if device.get("compute_capability") != RTX3090_COMPUTE_CAPABILITY:
            raise ValueError(
                f"{QUALIFICATION_PROFILE} requires compute capability "
                f"{RTX3090_COMPUTE_CAPABILITY}"
            )
        if device.get("memory_total_mib") != RTX3090_MEMORY_MIB:
            raise ValueError(
                f"{QUALIFICATION_PROFILE} requires exactly "
                f"{RTX3090_MEMORY_MIB} MiB per GPU"
            )


def sandbox_environment(
    runtime_dir: pathlib.Path,
    *,
    inherited: dict[str, str] | None = None,
    overrides: dict[str, str | None] | None = None,
) -> dict[str, str]:
    source = os.environ if inherited is None else inherited
    env = {key: value for key, value in source.items() if key in ENV_ALLOWLIST}
    env.setdefault("PATH", os.defpath)
    env["TMPDIR"] = str(runtime_dir / "tmp")
    env["XDG_CACHE_HOME"] = str(runtime_dir / "cache" / "xdg")
    env["XDG_CONFIG_HOME"] = str(runtime_dir / "config")
    env["XDG_DATA_HOME"] = str(runtime_dir / "data")
    if overrides:
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return env


def sandbox_command(runtime_dir: pathlib.Path, inner_command: list[str]) -> list[str]:
    command = [
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
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--bind",
        str(runtime_dir),
        str(runtime_dir),
        "--bind",
        str(runtime_dir / "tmp"),
        "/tmp",
    ]
    device_nodes = [
        path
        for path in pathlib.Path("/dev").glob("nvidia*")
        if path.is_char_device()
    ]
    for subtree in (pathlib.Path("/dev/nvidia-caps"), pathlib.Path("/dev/dri")):
        if subtree.is_dir():
            command.extend(["--dir", str(subtree)])
            device_nodes.extend(
                path for path in subtree.rglob("*") if path.is_char_device()
            )
    for node in device_nodes:
        if node.is_char_device():
            command.extend(["--dev-bind-try", str(node), str(node)])
    user_runtime = pathlib.Path("/run/user") / str(os.getuid())
    if user_runtime.is_dir():
        command.extend(["--tmpfs", str(user_runtime)])
    command.extend(["--", *inner_command])
    return command


def run_process_group(
    command: list[str],
    *,
    env: dict[str, str],
    deadline: Deadline,
    cwd: pathlib.Path | None = None,
    text: bool = True,
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        start_new_session=True,
    )
    try:
        remaining = deadline.remaining()
        stdout, stderr = process.communicate(timeout=max(0.001, remaining - 2.0))
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=deadline.remaining_or_zero(1))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.communicate(timeout=deadline.remaining_or_zero())
            except subprocess.TimeoutExpired:
                process.stdout.close()
                process.stderr.close()
        raise TimeoutError(f"command exceeded qualification deadline: {command!r}")
    return subprocess.CompletedProcess(
        command, process.returncode, stdout=stdout, stderr=stderr
    )


def validate_png_output(
    path: pathlib.Path, expected_width: int, expected_height: int
) -> dict[str, object]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("output is not a PNG")
    offset = 8
    ihdr = None
    idat = bytearray()
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("PNG has a truncated chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("PNG chunk extends beyond file")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        if binascii.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError(f"PNG CRC mismatch in {kind!r}")
        if kind == b"IHDR":
            if ihdr is not None or length != 13:
                raise ValueError("PNG has invalid IHDR")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            if length != 0 or end != len(data):
                raise ValueError("PNG has invalid IEND or trailing bytes")
            saw_iend = True
            break
        offset = end
    if ihdr is None or not idat or not saw_iend:
        raise ValueError("PNG is missing IHDR, IDAT, or IEND")
    width, height, depth, color_type, compression, filtering, interlace = ihdr
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"PNG dimensions are {width}x{height}, expected "
            f"{expected_width}x{expected_height}"
        )
    if depth != 8 or compression != 0 or filtering != 0 or interlace != 0:
        raise ValueError("qualification PNG must be non-interlaced 8-bit")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"PNG has unsupported color type {color_type}")
    try:
        decoded = zlib.decompress(bytes(idat))
    except zlib.error as error:
        raise ValueError(f"PNG IDAT cannot be decompressed: {error}") from error
    if len(decoded) != height * (1 + width * channels):
        raise ValueError("PNG decoded byte count does not match dimensions")
    for row in range(height):
        if decoded[row * (1 + width * channels)] > 4:
            raise ValueError("PNG has an invalid row filter")
    return {"decoded": True, "width": width, "height": height}


def has_typed_cancelled_sse(value: str) -> bool:
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


def prove_replanned_work(
    before: dict[str, str], after: dict[str, str], disabled_id: str
) -> list[str]:
    if not before:
        raise ValueError("queue replan did not capture queued work before disable")
    if set(after) != set(before):
        raise ValueError("same queued work IDs were not observed after disable")
    if any(device_id == disabled_id for device_id in after.values()):
        raise ValueError("queued work remained assigned to the draining device")
    moved = sorted(
        work_id for work_id, device_id in before.items() if after[work_id] != device_id
    )
    if set(moved) != set(before):
        raise ValueError("same queued work did not move away from the draining device")
    return moved


def parse_csv_lines(text: str, columns: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("No running"):
            continue
        row = [part.strip() for part in raw.split(",")]
        if len(row) != columns:
            raise ValueError(f"expected {columns} CSV columns, got {row!r}")
        rows.append(row)
    return rows


def common_hex_prefix(stable_ids: list[str]) -> str | None:
    values = [value.removeprefix("cuda:").lower() for value in stable_ids]
    if len(values) < 2:
        return None
    prefix = os.path.commonprefix(values)
    return prefix or None


def models_tree_manifest(
    root: pathlib.Path, deadline: Deadline | None = None
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    paths: list[pathlib.Path] = []
    for directory, names, files in os.walk(root, followlinks=False):
        if deadline is not None:
            deadline.remaining()
        base = pathlib.Path(directory)
        for name in sorted([*names, *files]):
            if deadline is not None:
                deadline.remaining()
            paths.append(base / name)
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        if deadline is not None:
            deadline.remaining()
        stat = path.lstat()
        row: dict[str, object] = {
            "path": str(path.relative_to(root)),
            "mode": stat.st_mode,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "kind": (
                "symlink"
                if path.is_symlink()
                else "directory"
                if path.is_dir()
                else "file"
            ),
        }
        if path.is_symlink():
            row["target"] = os.readlink(path)
        result.append(row)
    return result


def collect_model_artifacts(
    models_dir: pathlib.Path,
    paths: list[pathlib.Path],
    deadline: Deadline | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    root = models_dir.resolve()
    supplied_files: set[pathlib.Path] = set()
    roots: set[pathlib.Path] = set()
    for supplied in paths:
        path = supplied.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"model artifact is outside --models-dir: {path}"
            ) from error
        if path in supplied_files:
            raise ValueError(f"duplicate model artifact: {path}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"model artifact is not a regular file: {path}")
        supplied_files.add(path)
        roots.add(path.parent)
    artifacts: list[dict[str, object]] = []
    seen: set[pathlib.Path] = set()
    for artifact_root in sorted(roots, key=str):
        candidates: list[pathlib.Path] = []
        for directory, names, files in os.walk(artifact_root, followlinks=False):
            if deadline is not None:
                deadline.remaining()
            base = pathlib.Path(directory)
            for name in sorted([*names, *files]):
                if deadline is not None:
                    deadline.remaining()
                candidates.append(base / name)
        for path in sorted(candidates, key=str):
            if deadline is not None:
                deadline.remaining()
            if path in seen or not path.is_file() or path.is_symlink():
                continue
            seen.add(path)
            artifacts.append(
                {
                    "path": str(path),
                    "sha256": sha256(path, deadline),
                    "size": path.stat().st_size,
                }
            )
    if not artifacts:
        raise ValueError("--model-artifact must identify every selected model file")
    return artifacts, [str(path) for path in sorted(roots, key=str)]


def git_source_commit(source_root: pathlib.Path, deadline: Deadline) -> str:
    result = run_process_group(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        env=os.environ.copy(),
        deadline=deadline,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40 or any(
        char not in "0123456789abcdef" for char in commit
    ):
        raise RuntimeError(f"cannot resolve exact source commit: {result.stderr}")
    return commit


def run_candidate_probe(
    *,
    binary: pathlib.Path,
    runtime_dir: pathlib.Path,
    argv: list[str],
    deadline: Deadline,
    overrides: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for relative in (
        "home",
        "tmp",
        "cache/xdg",
        "config",
        "data",
    ):
        (runtime_dir / relative).mkdir(parents=True, exist_ok=True)
    env = sandbox_environment(runtime_dir, overrides=overrides)
    return run_process_group(
        sandbox_command(runtime_dir, [str(binary), *argv]),
        env=env,
        deadline=deadline,
    )


class Evidence:
    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.items: list[dict[str, str]] = []
        self._labels: set[str] = set()

    def _add(
        self,
        label: str,
        path: pathlib.Path,
        kind: str,
        deadline: Deadline | None = None,
    ) -> pathlib.Path:
        if label in self._labels:
            raise ValueError(f"duplicate evidence label: {label}")
        digest = sha256(path, deadline)
        self._labels.add(label)
        self.items.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": digest,
                "kind": kind,
            }
        )
        return path

    def text(self, label: str, value: str) -> pathlib.Path:
        path = self.directory / f"{label}.txt"
        path.write_text(value, encoding="utf-8")
        return self._add(label, path, "text")

    def json(self, label: str, value: object) -> pathlib.Path:
        path = self.directory / f"{label}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self._add(label, path, "json")

    def jsonl(self, label: str, path: pathlib.Path) -> pathlib.Path:
        return self._add(label, path, "jsonl")

    def existing(
        self,
        label: str,
        path: pathlib.Path,
        kind: str,
        deadline: Deadline | None = None,
    ) -> pathlib.Path:
        return self._add(label, path, kind, deadline)

    def contains(self, label: str) -> bool:
        return label in self._labels


class Api:
    def __init__(self, port: int, api_key: str):
        self.base = f"http://127.0.0.1:{port}"
        self.api_key = api_key
        self.deadline: Deadline | None = None

    def request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        timeout: float = 30,
    ) -> tuple[int, dict[str, str], bytes]:
        if self.deadline is not None:
            timeout = self.deadline.remaining(timeout)
        data = None if body is None else json.dumps(body).encode()
        headers = {"x-api-key": self.api_key}
        if data is not None:
            headers["content-type"] = "application/json"
        request = urllib.request.Request(
            self.base + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, dict(response.headers.items()), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers.items()), error.read()

    def json(
        self,
        method: str,
        path: str,
        body: object | None = None,
        timeout: float = 30,
    ) -> tuple[int, object]:
        status, _, payload = self.request(method, path, body, timeout)
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{method} {path} returned {status} with non-JSON body: "
                f"{payload[:500]!r}"
            ) from error
        return status, decoded


class CandidateServer:
    def __init__(
        self,
        *,
        binary: pathlib.Path,
        models_dir: pathlib.Path,
        runtime_dir: pathlib.Path,
        port: int,
        api_key: str,
        label: str,
        evidence: Evidence,
        gpus: str | None = None,
        dispatch_mode: str = "v2",
        cuda_visible_devices: str | None = None,
        empty_gpu_env: bool = False,
    ):
        self.binary = binary
        self.models_dir = models_dir
        self.runtime_dir = runtime_dir
        self.port = port
        self.api_key = api_key
        self.label = label
        self.evidence = evidence
        self.gpus = gpus
        self.dispatch_mode = dispatch_mode
        self.cuda_visible_devices = cuda_visible_devices
        self.empty_gpu_env = empty_gpu_env
        self.process: subprocess.Popen[bytes] | None = None
        self.server_pid: int | None = None
        self.process_identity: dict[str, object] | None = None
        self.log_handle = None
        self.log_path = evidence.directory / f"{label}-server.log"
        self.home = runtime_dir / "home"
        self.db = runtime_dir / "mold.db"
        self.output = runtime_dir / "output"
        self.api = Api(port, api_key)

    def inner_command(self) -> list[str]:
        result = [
            str(self.binary),
            "serve",
            "--bind",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--models-dir",
            str(self.models_dir),
            "--queue-size",
            "64",
            "--log-format",
            "json",
        ]
        if self.gpus is not None:
            result.extend(["--gpus", self.gpus])
        return result

    def command(self) -> list[str]:
        return sandbox_command(self.runtime_dir, self.inner_command())

    def environment(self) -> dict[str, str]:
        overrides: dict[str, str | None] = {
                "MOLD_HOME": str(self.home),
                "MOLD_DB_PATH": str(self.db),
                "MOLD_OUTPUT_DIR": str(self.output),
                "MOLD_API_KEY": self.api_key,
                "MOLD_DISPATCH_MODE": self.dispatch_mode,
                "MOLD_MDNS": "0",
                "MOLD_LOG": "info",
                "TMPDIR": str(self.runtime_dir / "tmp"),
                "XDG_CACHE_HOME": str(self.runtime_dir / "cache" / "xdg"),
                "XDG_CONFIG_HOME": str(self.runtime_dir / "config"),
                "XDG_DATA_HOME": str(self.runtime_dir / "data"),
                "CUDA_CACHE_PATH": str(self.runtime_dir / "cache" / "cuda"),
                "HF_HOME": str(self.runtime_dir / "cache" / "huggingface"),
                "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
            }
        if self.cuda_visible_devices is not None:
            overrides["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        if self.empty_gpu_env:
            overrides["MOLD_GPUS"] = ""
        elif self.gpus is None:
            overrides["MOLD_GPUS"] = None
        return sandbox_environment(self.runtime_dir, overrides=overrides)

    def recorded_environment(self) -> dict[str, str | None]:
        env = self.environment()
        return {
            key: env.get(key)
            for key in (
                "HOME",
                "TMPDIR",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "CUDA_CACHE_PATH",
                "HF_HOME",
                "MOLD_HOME",
                "MOLD_DB_PATH",
                "MOLD_OUTPUT_DIR",
                "MOLD_HOST",
                "MOLD_DISPATCH_MODE",
                "MOLD_MDNS",
                "MOLD_GPUS",
                "CUDA_VISIBLE_DEVICES",
            )
        }

    def prepare_runtime(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.home.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "tmp").mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "cache" / "cuda").mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "cache" / "huggingface").mkdir(
            parents=True, exist_ok=True
        )
        (self.runtime_dir / "cache" / "xdg").mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "config").mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "data").mkdir(parents=True, exist_ok=True)

    def start(self, deadline: Deadline) -> None:
        self.prepare_runtime()
        self.api.deadline = deadline
        self.log_handle = self.log_path.open("ab", buffering=0)
        command_path = self.evidence.directory / f"{self.label}-command.json"
        command_path.write_text(
            json.dumps(
                {
                    "argv": self.command(),
                    "environment": self.recorded_environment(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.evidence.existing(f"{self.label}-command", command_path, "json")
        self.process = subprocess.Popen(
            self.command(),
            env=self.environment(),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ready_deadline = time.monotonic() + deadline.remaining(45)
        last_error = ""
        while time.monotonic() < ready_deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.label} server exited {self.process.returncode}; "
                    f"see {self.log_path}"
                )
            try:
                status, _ = self.api.json(
                    "GET", "/api/status", timeout=deadline.remaining(2)
                )
                if status == 200:
                    self.server_pid = self._resolve_candidate_pid()
                    self.process_identity = candidate_process_identity(
                        self.server_pid, self.binary, deadline
                    )
                    return
                last_error = f"HTTP {status}"
            except (OSError, RuntimeError) as error:
                last_error = str(error)
            time.sleep(0.2)
        raise TimeoutError(f"{self.label} server did not become ready: {last_error}")

    def stop(self, deadline: Deadline) -> str | None:
        cleanup_error = None
        try:
            if self.process is not None and self.process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=deadline.remaining_or_zero(5))
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                    try:
                        self.process.wait(timeout=deadline.remaining_or_zero(2))
                    except subprocess.TimeoutExpired:
                        cleanup_error = (
                            f"{self.label} process group did not exit before the "
                            "campaign deadline"
                        )
        except OSError as error:
            cleanup_error = f"{self.label} cleanup failed: {error}"
        finally:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            if self.log_path.exists() and self.label + "-server-log" not in {
                item["label"] for item in self.evidence.items
            }:
                try:
                    self.evidence.existing(
                        f"{self.label}-server-log",
                        self.log_path,
                        "text",
                        deadline,
                    )
                except TimeoutError as error:
                    cleanup_error = (
                        f"{cleanup_error}; {error}"
                        if cleanup_error
                        else str(error)
                    )
        return cleanup_error

    @property
    def pid(self) -> int:
        if self.server_pid is None:
            raise RuntimeError("exact candidate PID has not been resolved")
        return self.server_pid

    def _resolve_candidate_pid(self) -> int:
        if self.process is None:
            raise RuntimeError("server has not started")
        pending = [self.process.pid]
        visited: set[int] = set()
        matches: list[int] = []
        expected = self.binary.resolve()
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            try:
                executable = pathlib.Path(f"/proc/{pid}/exe").resolve()
                if executable == expected:
                    matches.append(pid)
                children = pathlib.Path(
                    f"/proc/{pid}/task/{pid}/children"
                ).read_text(encoding="utf-8")
                pending.extend(int(value) for value in children.split())
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        if len(matches) != 1:
            raise RuntimeError(
                "candidate mount namespace did not expose exactly one Mold "
                f"process descendant: launcher={self.process.pid}, matches={matches}"
            )
        return matches[0]


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def candidate_process_identity(
    pid: int, binary: pathlib.Path, deadline: Deadline | None = None
) -> dict[str, object]:
    proc = pathlib.Path(f"/proc/{pid}")
    stat_fields = (proc / "stat").read_text(encoding="utf-8").split()
    executable = (proc / "exe").resolve()
    executable_stat = executable.stat()
    status = (proc / "status").read_text(encoding="utf-8")
    nspid_line = next(
        (line for line in status.splitlines() if line.startswith("NSpid:")),
        None,
    )
    if executable != binary.resolve() or nspid_line is None:
        raise RuntimeError("candidate process identity is incomplete")
    return {
        "pid": pid,
        "start_ticks": int(stat_fields[21]),
        "executable": str(executable),
        "executable_device": executable_stat.st_dev,
        "executable_inode": executable_stat.st_ino,
        "binary_sha256": sha256(binary, deadline),
        "pid_namespace": os.readlink(proc / "ns" / "pid"),
        "nspid": [int(value) for value in nspid_line.split()[1:]],
    }


def _tcp_rows(
    path: pathlib.Path, deadline: Deadline | None = None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return rows
    for line in lines:
        if deadline is not None:
            deadline.remaining()
        fields = line.split()
        if len(fields) < 10:
            continue
        local_port = int(fields[1].split(":")[1], 16)
        remote_port = int(fields[2].split(":")[1], 16)
        rows.append(
            {
                "family": path.name,
                "local_port": local_port,
                "remote_port": remote_port,
                "state": fields[3],
                "inode": fields[9],
            }
        )
    return rows


def _process_socket_inodes(
    pid: int, deadline: Deadline | None = None
) -> set[str]:
    result: set[str] = set()
    try:
        descriptors = pathlib.Path(f"/proc/{pid}/fd").iterdir()
        for descriptor in descriptors:
            if deadline is not None:
                deadline.remaining()
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                result.add(target[8:-1])
    except OSError:
        pass
    return result


def reserved_service_snapshot(
    deadline: Deadline, port: int = RESERVED_SERVICE_PORT
) -> dict[str, object]:
    listener_rows = [
        row
        for table in (pathlib.Path("/proc/net/tcp"), pathlib.Path("/proc/net/tcp6"))
        for row in _tcp_rows(table, deadline)
        if row["local_port"] == port and row["state"] == "0A"
    ]
    listener_inodes = {str(row["inode"]) for row in listener_rows}
    owners: list[dict[str, object]] = []
    processes: list[pathlib.Path] = []
    with os.scandir("/proc") as entries:
        for entry in entries:
            deadline.remaining()
            if entry.name.isdigit():
                processes.append(pathlib.Path(entry.path))
    for proc in sorted(processes, key=lambda path: int(path.name)):
        deadline.remaining()
        pid = int(proc.name)
        owned = sorted(listener_inodes & _process_socket_inodes(pid, deadline))
        if not owned:
            continue
        try:
            stat_fields = (proc / "stat").read_text(encoding="utf-8").split()
            executable = str((proc / "exe").resolve())
            start_ticks = int(stat_fields[21])
        except (OSError, IndexError, ValueError):
            continue
        owners.append(
            {
                "pid": pid,
                "start_ticks": start_ticks,
                "executable": executable,
                "socket_inodes": owned,
            }
        )
    return {
        "port": port,
        "listeners": sorted(
            listener_rows,
            key=lambda row: (
                str(row["family"]),
                int(row["local_port"]),
                str(row["inode"]),
            ),
        ),
        "owners": owners,
    }


def candidate_reserved_connections(
    pid: int, deadline: Deadline, port: int = RESERVED_SERVICE_PORT
) -> list[dict[str, object]]:
    owned = _process_socket_inodes(pid, deadline)
    return [
        row
        for table in (
            pathlib.Path(f"/proc/{pid}/net/tcp"),
            pathlib.Path(f"/proc/{pid}/net/tcp6"),
        )
        for row in _tcp_rows(table, deadline)
        if str(row["inode"]) in owned
        and row["state"] != "0A"
        and (row["local_port"] == port or row["remote_port"] == port)
    ]


def nvidia_inventory(deadline: Deadline) -> tuple[list[dict[str, object]], str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = run_process_group(command, env=os.environ.copy(), deadline=deadline)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory failed: {result.stderr}")
    devices = []
    for index, gpu_uuid, name, memory, capability, driver in parse_csv_lines(
        result.stdout, 6
    ):
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
    return devices, result.stdout


def compute_apps(deadline: Deadline) -> tuple[list[dict[str, object]], str]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]
    result = run_process_group(command, env=os.environ.copy(), deadline=deadline)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute-app query failed: {result.stderr}")
    rows = [
        {"pid": int(pid), "gpu_uuid": gpu_uuid}
        for pid, gpu_uuid in parse_csv_lines(result.stdout, 2)
    ]
    return rows, result.stdout


def wait_for(
    description: str,
    callback,
    *,
    timeout: float,
    interval: float = 0.2,
):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = callback()
        if last:
            return last
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting for {description}; last={last!r}")


def check(status: str, summary: str, labels: list[str]) -> dict[str, object]:
    return {"status": status, "summary": summary, "evidence_labels": labels}


def default_request(model: str) -> dict[str, object]:
    return {
        "prompt": "Mold real local multi-GPU acceptance",
        "model": model,
        "width": 512,
        "height": 512,
        "steps": 8,
        "guidance": 1.0,
        "seed": 9042026,
        "batch_size": 1,
        "output_format": "png",
    }


def api_snapshot(api: Api) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, path in (
        ("status", "/api/status"),
        ("devices", "/api/devices"),
        ("resources", "/api/resources"),
        ("queue", "/api/queue"),
        ("capabilities", "/api/capabilities"),
        ("models", "/api/models"),
    ):
        status, body = api.json("GET", path)
        if status != 200:
            raise RuntimeError(f"GET {path} returned {status}: {body}")
        result[name] = body
    return result


def validate_initial_projection(
    snapshot: dict[str, object], expected_uuids: list[str]
) -> tuple[list[dict[str, object]], list[str]]:
    devices_payload = snapshot["devices"]
    if not isinstance(devices_payload, dict):
        raise RuntimeError("/api/devices did not return an object")
    devices = devices_payload.get("devices")
    if not isinstance(devices, list):
        raise RuntimeError("/api/devices omitted devices")
    observed = [device.get("nvml_uuid") for device in devices]
    if set(observed) != set(expected_uuids) or len(observed) != len(expected_uuids):
        raise RuntimeError(f"/api/devices UUID mismatch: {observed!r}")
    if not all(
        device.get("desired_enabled") is True
        and device.get("admin_state") == "enabled"
        and device.get("schedulable") is True
        for device in devices
    ):
        raise RuntimeError("not every expected device started enabled and schedulable")
    status_gpus = snapshot["status"].get("gpus", [])
    resources_gpus = snapshot["resources"].get("gpus", [])
    if len(status_gpus) != len(expected_uuids) or len(resources_gpus) != len(
        expected_uuids
    ):
        raise RuntimeError("legacy status or resource telemetry omitted a GPU")
    expected_projection = {
        (device.get("ordinal"), device.get("name")) for device in devices
    }
    status_projection = {
        (device.get("ordinal"), device.get("name")) for device in status_gpus
    }
    resource_projection = {
        (device.get("ordinal"), device.get("name")) for device in resources_gpus
    }
    if (
        status_projection != expected_projection
        or resource_projection != expected_projection
    ):
        raise RuntimeError(
            "legacy status/resource ordinal-name projections differ from /api/devices"
        )
    capabilities = snapshot["capabilities"]
    if not (
        capabilities.get("devices", {}).get("available")
        and capabilities.get("devices", {}).get("lifecycle")
        and capabilities.get("devices", {}).get("planned_lanes")
        and capabilities.get("dispatch", {}).get("v2_authoritative")
        and capabilities.get("dispatch", {}).get("active_mode") == "v2"
    ):
        raise RuntimeError("candidate did not advertise authoritative V2 device APIs")
    queue = snapshot["queue"]
    if not isinstance(queue, dict) or not isinstance(queue.get("plan"), dict):
        raise RuntimeError("queue listing omitted the V2 plan")
    return devices, [device["id"] for device in devices]


def run_client_projection(
    server: CandidateServer,
    binary: pathlib.Path,
    port: int,
    api_key: str,
    expected_ids: list[str],
    deadline: Deadline,
) -> tuple[dict[str, object], str, list[str], dict[str, str | None]]:
    env = server.environment()
    env["MOLD_HOST"] = f"http://127.0.0.1:{port}"
    env["MOLD_API_KEY"] = api_key
    command = sandbox_command(
        server.runtime_dir, [str(binary), "gpu", "list", "--json"]
    )
    result = run_process_group(
        command,
        env=env,
        deadline=deadline,
    )
    if result.returncode != 0:
        raise RuntimeError(f"candidate gpu list failed: {result.stderr}")
    payload = json.loads(result.stdout)
    ids = [device["id"] for device in payload.get("devices", [])]
    if ids != expected_ids:
        raise RuntimeError(f"CLI projection IDs differ: {ids!r} != {expected_ids!r}")
    recorded = {
        key: env.get(key)
        for key in (
            "HOME",
            "TMPDIR",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "CUDA_CACHE_PATH",
            "HF_HOME",
            "MOLD_HOME",
            "MOLD_DB_PATH",
            "MOLD_OUTPUT_DIR",
            "MOLD_HOST",
        )
    }
    return payload, result.stderr, command, recorded


def run_parallel_lifecycle(
    server: CandidateServer,
    request: dict[str, object],
    jobs: int,
    evidence: Evidence,
    deadline: Deadline,
    eligible_idle_seconds: float,
) -> dict[str, object]:
    sample_path = evidence.directory / "parallel-runtime-samples.jsonl"
    lock = threading.Lock()
    prompt_to_index = {
        f"{request['prompt']} variation {index + 1}": index + 1
        for index in range(jobs)
    }
    work_id_by_index: dict[int, str] = {}

    def generate(index: int) -> dict[str, object]:
        child = dict(request)
        child["prompt"] = f"{request['prompt']} variation {index + 1}"
        child["seed"] = int(request.get("seed", 0)) + index
        child["batch_size"] = 1
        status, headers, payload = server.api.request(
            "POST", "/api/generate", child, timeout=deadline.remaining()
        )
        output = evidence.directory / f"parallel-output-{index + 1}.png"
        output.write_bytes(payload)
        decoded = validate_png_output(
            output, int(child["width"]), int(child["height"])
        )
        normalized_headers = {key.lower(): value for key, value in headers.items()}
        if normalized_headers.get("content-type", "").split(";", 1)[0] != "image/png":
            raise RuntimeError(
                f"parallel output {index + 1} did not return image/png"
            )
        expected_seed = int(child["seed"])
        try:
            response_seed = int(normalized_headers["x-mold-seed-used"])
            gpu_ordinal = int(normalized_headers["x-mold-gpu"])
        except (KeyError, ValueError) as error:
            raise RuntimeError(
                f"parallel output {index + 1} omitted exact seed/GPU headers"
            ) from error
        if response_seed != expected_seed:
            raise RuntimeError(
                f"parallel output {index + 1} seed mismatch: "
                f"{response_seed} != {expected_seed}"
            )
        with lock:
            evidence.existing(f"parallel-output-{index + 1}", output, "png")
        return {
            "index": index + 1,
            "prompt": child["prompt"],
            "seed": expected_seed,
            "status": status,
            "headers": normalized_headers,
            "content_type": "image/png",
            "gpu_ordinal": gpu_ordinal,
            "decoded": decoded,
            "path": str(output.resolve()),
            "size": len(payload),
            "sha256": sha256(output),
        }

    observed_compute_uuids: set[str] = set()
    observed_active_uuids: set[str] = set()
    disabled_id = None
    disabled_uuid = None
    initial_plan_version = None
    replanned = False
    pre_disable_assignments: dict[str, str] = {}
    post_disable_assignments: dict[str, str] = {}
    replanned_work_ids: list[str] = []
    disable_response = None
    unexcused_idle_started: float | None = None
    maximum_unexcused_idle_seconds = 0.0
    typed_idle_exceptions: list[dict[str, object]] = []
    decisive_samples: list[dict[str, object]] = []
    ordinal_to_uuid: dict[int, str] = {}
    execution_bindings: dict[str, str] = {}

    with sample_path.open("w", encoding="utf-8") as samples, concurrent.futures.ThreadPoolExecutor(
        max_workers=jobs
    ) as pool:
        futures = [pool.submit(generate, index) for index in range(jobs)]
        loop_deadline = time.monotonic() + deadline.remaining()
        while time.monotonic() < loop_deadline:
            _, devices_payload = server.api.json("GET", "/api/devices")
            _, queue_payload = server.api.json("GET", "/api/queue")
            apps, raw_apps = compute_apps(deadline)
            devices = devices_payload["devices"]
            ordinal_to_uuid.update(
                {
                    int(device["ordinal"]): str(device["nvml_uuid"])
                    for device in devices
                    if device.get("ordinal") is not None and device.get("nvml_uuid")
                }
            )
            plan_items = (queue_payload.get("plan") or {}).get("work_items", [])
            for entry in queue_payload.get("entries", []):
                metadata = entry.get("metadata") or {}
                index = prompt_to_index.get(metadata.get("prompt"))
                if index is None:
                    continue
                matching = [
                    item
                    for item in plan_items
                    if item.get("parent_id") == entry.get("id")
                ]
                if len(matching) == 1 and matching[0].get("work_id"):
                    work_id_by_index[index] = str(matching[0]["work_id"])
            active = [
                device
                for device in devices
                if device.get("active_work_id")
                and device.get("activity") not in {"idle", "stopping"}
            ]
            for device in active:
                if device.get("nvml_uuid"):
                    observed_active_uuids.add(device["nvml_uuid"])
            for row in apps:
                if row["pid"] == server.pid:
                    observed_compute_uuids.add(str(row["gpu_uuid"]))
            active_uuids = {
                str(device["nvml_uuid"])
                for device in active
                if device.get("nvml_uuid")
            }
            exact_pid_compute = {
                str(row["gpu_uuid"]) for row in apps if row["pid"] == server.pid
            }
            for device in active:
                work_id = device.get("active_work_id")
                gpu_uuid = device.get("nvml_uuid")
                if work_id and gpu_uuid in exact_pid_compute:
                    existing = execution_bindings.get(str(work_id))
                    if existing is not None and existing != gpu_uuid:
                        raise RuntimeError(
                            f"work ID {work_id} changed physical GPU UUID"
                        )
                    execution_bindings[str(work_id)] = str(gpu_uuid)
            if (
                len({device.get("active_work_id") for device in active}) >= 2
                and active_uuids
                and active_uuids <= exact_pid_compute
            ):
                decisive_samples.append(
                    {
                        "server_pid": server.pid,
                        "active": [
                            {
                                "device_id": device["id"],
                                "gpu_uuid": device["nvml_uuid"],
                                "work_id": device["active_work_id"],
                            }
                            for device in active
                        ],
                        "compute_apps": [
                            row for row in apps if row["pid"] == server.pid
                        ],
                    }
                )
            sample = {
                "at": utc_now(),
                "server_pid": server.pid,
                "candidate_identity": server.process_identity,
                "devices": devices,
                "queue": queue_payload,
                "compute_apps": apps,
                "compute_apps_raw": raw_apps,
                "reserved_service_connections": candidate_reserved_connections(
                    server.pid, deadline
                ),
            }
            samples.write(json.dumps(sample, sort_keys=True) + "\n")
            samples.flush()

            if disabled_id is None:
                queued = [
                    entry
                    for entry in queue_payload.get("entries", [])
                    if entry.get("state") == "queued"
                ]
                idle_schedulable = [
                    device
                    for device in devices
                    if device.get("schedulable")
                    and device.get("activity") == "idle"
                ]
                planned_waits = [
                    item
                    for item in (queue_payload.get("plan") or {}).get(
                        "work_items", []
                    )
                    if item.get("activity_phase")
                    in {"queued", "planned", "blocked"}
                    and (
                        item.get("blocked_reason")
                        or item.get("warm_wait_deadline_unix_ms")
                    )
                ]
                if queued and idle_schedulable:
                    if planned_waits:
                        typed_idle_exceptions.append(
                            {
                                "at": sample["at"],
                                "idle_device_ids": [
                                    device["id"] for device in idle_schedulable
                                ],
                                "planned_waits": planned_waits,
                            }
                        )
                        unexcused_idle_started = None
                    else:
                        now = time.monotonic()
                        if unexcused_idle_started is None:
                            unexcused_idle_started = now
                        duration = now - unexcused_idle_started
                        maximum_unexcused_idle_seconds = max(
                            maximum_unexcused_idle_seconds, duration
                        )
                        if duration > eligible_idle_seconds:
                            raise RuntimeError(
                                "schedulable GPU remained idle with queued compatible "
                                f"work and no typed plan exception for {duration:.2f}s"
                            )
                else:
                    unexcused_idle_started = None

            if disabled_id is None and len(active) >= 2:
                distinct_work = {device["active_work_id"] for device in active}
                if len(distinct_work) >= 2:
                    chosen = next(
                        (
                            device
                            for device in active
                            if any(
                                item.get("planned_device_id") == device["id"]
                                and item.get("activity_phase")
                                in {"queued", "planned", "blocked"}
                                for item in plan_items
                            )
                        ),
                        None,
                    )
                    if chosen is not None:
                        disabled_id = chosen["id"]
                        disabled_uuid = chosen.get("nvml_uuid")
                        pre_disable_assignments = {
                            str(item["work_id"]): str(item["planned_device_id"])
                            for item in plan_items
                            if item.get("planned_device_id") == disabled_id
                            and item.get("activity_phase")
                            in {"queued", "planned", "blocked"}
                        }
                        initial_plan_version = queue_payload.get("plan", {}).get(
                            "plan_version"
                        )
                        status, body = server.api.json(
                            "PATCH",
                            f"/api/devices/{urllib.parse.quote(disabled_id, safe='')}",
                            {"enabled": False},
                        )
                        disable_response = {"status": status, "body": body}
                        if status != 202 or body.get("admin_state") != "draining":
                            raise RuntimeError(
                                "busy disable did not return 202 draining: "
                                f"{disable_response!r}"
                            )

            if disabled_id is not None:
                plan = queue_payload.get("plan") or {}
                version = plan.get("plan_version")
                if (
                    isinstance(version, int)
                    and isinstance(initial_plan_version, int)
                    and version > initial_plan_version
                ):
                    current = {
                        str(item["work_id"]): str(item["planned_device_id"])
                        for item in plan.get("work_items", [])
                        if item.get("work_id") in pre_disable_assignments
                        and item.get("planned_device_id")
                    }
                    try:
                        moved = prove_replanned_work(
                            pre_disable_assignments, current, disabled_id
                        )
                    except ValueError:
                        pass
                    else:
                        post_disable_assignments = current
                        replanned_work_ids = moved
                        replanned = True

            if all(future.done() for future in futures):
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("parallel generation workload did not finish")
        results = [future.result() for future in futures]

    evidence.jsonl("parallel-runtime-samples", sample_path)
    if disabled_id is None:
        raise RuntimeError("never observed two distinct active GPU work IDs")
    if not replanned:
        raise RuntimeError(
            "queue plan never moved the same queued work IDs away from the "
            "draining device"
        )
    if not decisive_samples:
        raise RuntimeError(
            "no same-sample exact-PID/active-work/GPU-UUID execution evidence"
        )

    def disabled_state():
        _, body = server.api.json("GET", "/api/devices")
        device = next(item for item in body["devices"] if item["id"] == disabled_id)
        return device if device.get("admin_state") == "disabled" else None

    drained = wait_for(
        "busy device to finish and disable",
        disabled_state,
        timeout=deadline.remaining(30),
    )
    status, enabled = server.api.json(
        "PATCH",
        f"/api/devices/{urllib.parse.quote(disabled_id, safe='')}",
        {"enabled": True},
    )
    if status not in {200, 202}:
        raise RuntimeError(f"re-enable returned {status}: {enabled}")

    def enabled_state():
        _, body = server.api.json("GET", "/api/devices")
        device = next(item for item in body["devices"] if item["id"] == disabled_id)
        return device if device.get("admin_state") == "enabled" else None

    reenabled = wait_for(
        "device to re-enable", enabled_state, timeout=deadline.remaining(30)
    )
    if any(result["status"] != 200 or result["size"] == 0 for result in results):
        raise RuntimeError(f"one or more parallel generations failed: {results!r}")
    for result in results:
        index = int(result["index"])
        if index not in work_id_by_index:
            raise RuntimeError(f"parallel output {index} was not bound to an exact work ID")
        result["work_id"] = work_id_by_index[index]
        ordinal = int(result["gpu_ordinal"])
        if ordinal not in ordinal_to_uuid:
            raise RuntimeError(
                f"parallel output {index} GPU ordinal {ordinal} has no UUID mapping"
            )
        result["gpu_uuid"] = ordinal_to_uuid[ordinal]
        if execution_bindings.get(result["work_id"]) != result["gpu_uuid"]:
            raise RuntimeError(
                f"parallel output {index} lacks exact work-ID/PID/GPU-UUID binding"
            )
    response_gpus = {int(result["gpu_ordinal"]) for result in results}
    response_uuids = {str(result["gpu_uuid"]) for result in results}
    if len(response_gpus) < 2 or response_uuids != observed_active_uuids:
        raise RuntimeError(f"generation headers did not prove two GPUs: {response_gpus}")
    evidence.json(
        "parallel-results",
        {
            "server_pid": server.pid,
            "candidate_identity": server.process_identity,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "results": results,
            "observed_active_uuids": sorted(observed_active_uuids),
            "observed_compute_uuids": sorted(observed_compute_uuids),
            "decisive_samples": decisive_samples,
            "execution_bindings": execution_bindings,
            "disabled_id": disabled_id,
            "disabled_uuid": disabled_uuid,
            "disable_response": disable_response,
            "drained": drained,
            "reenabled": reenabled,
            "queue_replanned": replanned,
            "pre_disable_assignments": pre_disable_assignments,
            "post_disable_assignments": post_disable_assignments,
            "replanned_work_ids": replanned_work_ids,
            "maximum_unexcused_idle_seconds": maximum_unexcused_idle_seconds,
            "eligible_idle_limit_seconds": eligible_idle_seconds,
            "typed_idle_exceptions": typed_idle_exceptions,
        },
    )
    return {
        "observed_active_uuids": observed_active_uuids,
        "observed_compute_uuids": observed_compute_uuids,
        "disabled_id": disabled_id,
        "disabled_uuid": disabled_uuid,
        "response_gpus": response_gpus,
    }


def run_queued_cancellation(
    server: CandidateServer,
    request: dict[str, object],
    evidence: Evidence,
    deadline: Deadline,
) -> None:
    output_before = models_tree_manifest(server.output)
    paused = False
    thread: threading.Thread | None = None
    result: dict[str, object] = {}
    resume_status = None
    resume: object = None
    terminal_observed_at = None
    resume_at = None
    try:
        status, pause = server.api.json("POST", "/api/queue/pause")
        if status != 200 or not pause.get("paused"):
            raise RuntimeError(f"queue pause failed: {status} {pause}")
        paused = True
        _, before = server.api.json("GET", "/api/queue")
        before_ids = {entry["id"] for entry in before.get("entries", [])}
        stream_request = dict(request)
        stream_request["prompt"] = f"{request['prompt']} queued cancellation"

        def stream():
            result["response"] = server.api.request(
                "POST",
                "/api/generate/stream",
                stream_request,
                timeout=deadline.remaining(),
            )

        thread = threading.Thread(target=stream, daemon=True)
        thread.start()

        def queued_context():
            _, listing = server.api.json("GET", "/api/queue")
            candidates = [
                entry["id"]
                for entry in listing.get("entries", [])
                if entry["id"] not in before_ids
            ]
            if len(candidates) != 1:
                return None
            job_id = candidates[0]
            entry = next(
                row for row in listing.get("entries", []) if row.get("id") == job_id
            )
            if entry.get("state") != "queued":
                return None
            plan_items = (listing.get("plan") or {}).get("work_items", [])
            work = [
                item
                for item in plan_items
                if item.get("parent_id") == job_id and item.get("work_id")
                and item.get("activity_phase") in {"queued", "planned", "blocked"}
            ]
            if len(work) != 1:
                return None
            _, devices = server.api.json("GET", "/api/devices")
            return {
                "job_id": job_id,
                "work_id": str(work[0]["work_id"]),
                "queue": listing,
                "devices": devices,
            }

        queued = wait_for(
            "paused SSE job to enter queue with an exact work ID",
            queued_context,
            timeout=deadline.remaining(20),
        )
        job_id = str(queued["job_id"])
        work_id = str(queued["work_id"])
        if any(
            device.get("active_work_id") == work_id
            for device in queued["devices"]["devices"]
        ):
            raise RuntimeError("paused queued work became active before cancellation")
        cancel_status, _, cancel_payload = server.api.request(
            "DELETE", f"/api/queue/{urllib.parse.quote(job_id, safe='')}"
        )
        cancel_body = cancel_payload.decode(errors="replace")
        if cancel_status != 204:
            raise RuntimeError(
                f"queued cancellation failed: {cancel_status} {cancel_body}"
            )
        assert thread is not None
        thread.join(timeout=deadline.remaining(20))
        if thread.is_alive():
            raise RuntimeError("cancelled SSE request did not terminate")
        response_status, _, response_body = result["response"]
        stream_text = response_body.decode(errors="replace")
        typed_cancelled = response_status == 200 and has_typed_cancelled_sse(stream_text)
        if not typed_cancelled:
            raise RuntimeError("queued cancellation lacked a typed terminal SSE event")
        terminal_observed_at = utc_now()
        _, after_cancel = server.api.json("GET", "/api/queue")
        _, devices_after_cancel = server.api.json("GET", "/api/devices")
    finally:
        if paused:
            resume_status, resume = server.api.json("POST", "/api/queue/resume")
            resume_at = utc_now()

    if resume_status != 200 or not isinstance(resume, dict) or resume.get("paused"):
        raise RuntimeError(f"queue resume failed: {resume_status} {resume}")
    _, after = server.api.json("GET", "/api/queue")
    if any(entry["id"] == job_id for entry in after.get("entries", [])):
        raise RuntimeError("cancelled queued job remained in queue")
    if any(
        item.get("work_id") == work_id
        for item in (after.get("plan") or {}).get("work_items", [])
    ):
        raise RuntimeError("cancelled queued work remained in the scheduler plan")
    _, devices_after = server.api.json("GET", "/api/devices")
    never_active = not any(
        device.get("active_work_id") == work_id
        for device in devices_after["devices"]
    )
    output_after = models_tree_manifest(server.output)
    output_unchanged = output_after == output_before
    if not typed_cancelled or not never_active or not output_unchanged:
        raise RuntimeError(
            "queued cancellation lacked typed/no-inference/output proof"
        )
    evidence.json(
        "queued-cancellation",
        {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "server_pid": server.pid,
            "job_id": job_id,
            "work_id": work_id,
            "cancel_status": cancel_status,
            "cancel_body": cancel_body,
            "resume_status": resume_status,
            "stream_http_status": response_status,
            "stream_tail": stream_text[-2000:],
            "typed_cancelled": typed_cancelled,
            "never_active": never_active,
            "output_tree_unchanged": output_unchanged,
            "queue_was_paused": True,
            "request": stream_request,
            "terminal_observed_at": terminal_observed_at,
            "resume_at": resume_at,
            "queue_before_cancel": queued["queue"],
            "devices_before_cancel": queued["devices"],
            "queue_after_cancel_before_resume": after_cancel,
            "devices_after_cancel_before_resume": devices_after_cancel,
            "queue_after": after,
            "devices_after": devices_after,
            "output_tree_before": output_before,
            "output_tree_after": output_after,
        },
    )


def set_all_devices(
    server: CandidateServer, enabled: bool, deadline: Deadline
) -> None:
    _, payload = server.api.json("GET", "/api/devices")
    for device in payload["devices"]:
        status, body = server.api.json(
            "PATCH",
            f"/api/devices/{urllib.parse.quote(device['id'], safe='')}",
            {"enabled": enabled},
        )
        if status not in {200, 202}:
            raise RuntimeError(f"device mutation failed: {status} {body}")

    target = "enabled" if enabled else "disabled"

    def all_target():
        _, current = server.api.json("GET", "/api/devices")
        return current if all(
            device.get("admin_state") == target for device in current["devices"]
        ) else None

    wait_for(
        f"all devices to become {target}",
        all_target,
        timeout=deadline.remaining(30),
    )


def run_maintenance(
    server: CandidateServer,
    request: dict[str, object],
    evidence: Evidence,
    deadline: Deadline,
    expected_mapping: dict[str, str],
) -> None:
    set_all_devices(server, False, deadline)
    status, _, body = server.api.request(
        "POST", "/api/generate", request, timeout=deadline.remaining(10)
    )
    text = body.decode(errors="replace")
    if status not in {409, 503} or "maintenance" not in text.lower():
        raise RuntimeError(
            f"all-disabled generation was not rejected as maintenance: {status} {text}"
        )
    _, devices = server.api.json("GET", "/api/devices")
    maintenance_rows = devices.get("devices", [])
    maintenance_mapping = {
        str(device.get("id")): str(device.get("nvml_uuid"))
        for device in maintenance_rows
    }
    if (
        len(maintenance_rows) != 2
        or maintenance_mapping != expected_mapping
        or any(
            device.get("admin_state") != "disabled"
            for device in maintenance_rows
        )
    ):
        raise RuntimeError("maintenance snapshot is not the exact disabled inventory")
    evidence.json(
        "all-disabled-maintenance",
        {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "server_pid": server.pid,
            "status": status,
            "body": text,
            "devices": devices,
        },
    )
    set_all_devices(server, True, deadline)


def start_selector_server(
    *,
    args,
    evidence: Evidence,
    api_key: str,
    label: str,
    gpus: str | None,
    expected_uuids: set[str],
    qualification_mapping: dict[str, str],
    deadline: Deadline,
    cuda_visible_devices: str | None = None,
    empty_gpu_env: bool = False,
) -> dict[str, object]:
    runtime = evidence.directory / "selector-runtimes" / label
    server = CandidateServer(
        binary=args.binary,
        models_dir=args.models_dir,
        runtime_dir=runtime,
        port=args.port,
        api_key=api_key,
        label=f"selector-{label}",
        evidence=evidence,
        gpus=gpus,
        cuda_visible_devices=cuda_visible_devices,
        empty_gpu_env=empty_gpu_env,
    )
    try:
        server.start(deadline)
        _, payload = server.api.json("GET", "/api/devices")
        selected = {
            device["nvml_uuid"]
            for device in payload["devices"]
            if device.get("admin_state") != "startup_excluded"
        }
        observed = {
            str(device["id"]): str(device["nvml_uuid"])
            for device in payload.get("devices", [])
        }
        if len(payload.get("devices", [])) != 2 or observed != qualification_mapping:
            raise RuntimeError(
                f"selector {label} omitted or changed qualification inventory: "
                f"{observed!r}"
            )
        if selected != expected_uuids:
            raise RuntimeError(
                f"selector {label} selected {sorted(selected)!r}, "
                f"expected {sorted(expected_uuids)!r}"
            )
        return {
            "label": label,
            "selector": gpus,
            "expected_uuids": sorted(expected_uuids),
            "devices": payload,
            "pid": server.pid,
            "process_identity": server.process_identity,
            "argv": server.command(),
            "environment": server.recorded_environment(),
        }
    finally:
        cleanup_error = server.stop(deadline)
        if cleanup_error:
            raise RuntimeError(cleanup_error)


def run_selector_matrix(
    args,
    evidence: Evidence,
    api_key: str,
    expected_uuids: list[str],
    stable_ids_by_uuid: dict[str, str],
    deadline: Deadline,
) -> None:
    all_set = set(expected_uuids)
    qualification_mapping = {
        stable_ids_by_uuid[gpu_uuid]: gpu_uuid for gpu_uuid in expected_uuids
    }
    first_uuid, second_uuid = expected_uuids[:2]
    results = [
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="empty",
            gpus=None,
            expected_uuids=all_set,
            qualification_mapping=qualification_mapping,
            deadline=deadline,
            empty_gpu_env=True,
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="all",
            gpus="all",
            expected_uuids=all_set,
            qualification_mapping=qualification_mapping,
            deadline=deadline,
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="ordinal-one",
            gpus="1",
            expected_uuids={second_uuid},
            qualification_mapping=qualification_mapping,
            deadline=deadline,
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="stable-id",
            gpus=stable_ids_by_uuid[first_uuid],
            expected_uuids={first_uuid},
            qualification_mapping=qualification_mapping,
            deadline=deadline,
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="nvidia-uuid",
            gpus=second_uuid,
            expected_uuids={second_uuid},
            qualification_mapping=qualification_mapping,
            deadline=deadline,
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="none",
            gpus="none",
            expected_uuids=set(),
            qualification_mapping=qualification_mapping,
            deadline=deadline,
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="reordered-stable",
            gpus=stable_ids_by_uuid[first_uuid],
            expected_uuids={first_uuid},
            qualification_mapping=qualification_mapping,
            deadline=deadline,
            cuda_visible_devices=f"{second_uuid},{first_uuid}",
        ),
    ]

    missing = CandidateServer(
        binary=args.binary,
        models_dir=args.models_dir,
        runtime_dir=evidence.directory / "selector-runtimes" / "missing",
        port=args.port,
        api_key=api_key,
        label="selector-missing",
        evidence=evidence,
        gpus="GPU-ffffffff",
    )
    missing.prepare_runtime()
    missing_command = missing.command()
    process = run_process_group(
        missing_command, env=missing.environment(), deadline=deadline, text=False
    )
    missing.log_path.write_bytes(process.stdout + process.stderr)
    evidence.existing("selector-missing-server-log", missing.log_path, "text")
    if process.returncode == 0:
        raise RuntimeError("unmatched NVIDIA UUID selector unexpectedly started")
    if "did not match" not in missing.log_path.read_text(errors="replace"):
        raise RuntimeError("unmatched selector did not report a typed match failure")
    results.append(
        {
            "label": "missing",
            "selector": "GPU-ffffffff",
            "expected_uuids": [],
            "exit_code": process.returncode,
            "expected_failure": "did not match",
            "argv": missing_command,
            "environment": missing.recorded_environment(),
        }
    )

    contract_runtime = evidence.directory / "selector-contract-runtime"
    CandidateServer(
        binary=args.binary,
        models_dir=args.models_dir,
        runtime_dir=contract_runtime,
        port=args.port,
        api_key=api_key,
        label="selector-contract",
        evidence=evidence,
    ).prepare_runtime()
    contract_env = sandbox_environment(
        contract_runtime,
        overrides={"CARGO_TARGET_DIR": str(contract_runtime / "cargo-target")},
    )
    contract_argv = sandbox_command(
        contract_runtime,
        ["/bin/bash", "-lc", args.selector_contract_command],
    )
    command = run_process_group(
        contract_argv,
        cwd=args.source_root,
        env=contract_env,
        deadline=deadline,
    )
    evidence.json(
        "ambiguous-selector-source-contract",
        {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "command": args.selector_contract_command,
            "cwd": str(args.source_root),
            "exit_code": command.returncode,
            "stdout": command.stdout,
            "stderr": command.stderr,
            "argv": contract_argv,
            "environment": {
                key: contract_env.get(key)
                for key in (
                    "HOME",
                    "TMPDIR",
                    "XDG_CACHE_HOME",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                    "MOLD_HOST",
                )
            },
            "hardware_prefix": common_hex_prefix(list(stable_ids_by_uuid.values())),
            "hardware_claimed": False,
        },
    )
    if command.returncode != 0:
        raise RuntimeError("deterministic ambiguous selector contract failed")
    evidence.json(
        "selector-matrix",
        {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "scenarios": results,
        },
    )


def self_test() -> int:
    assert parse_csv_lines("1, GPU-a\n", 2) == [["1", "GPU-a"]]
    assert common_hex_prefix(["cuda:aa00", "cuda:aaff"]) == "aa"
    assert common_hex_prefix(["cuda:4400", "cuda:ba00"]) is None
    with contextlib.ExitStack() as stack:
        import tempfile

        root = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory()))
        (root / "a").write_text("x", encoding="utf-8")
        before = models_tree_manifest(root)
        assert before == models_tree_manifest(root)
        (root / "a").write_text("xx", encoding="utf-8")
        assert before != models_tree_manifest(root)
    print("local multi-GPU qualification self-test passed")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--binary", type=pathlib.Path)
    parser.add_argument("--models-dir", type=pathlib.Path)
    parser.add_argument(
        "--model-artifact",
        type=pathlib.Path,
        action="append",
        default=[],
        help="exact selected model/component file; repeat for every required artifact",
    )
    parser.add_argument("--request", type=pathlib.Path)
    parser.add_argument(
        "--model",
        help="model used only when --request is omitted",
    )
    parser.add_argument(
        "--expected-gpu-uuid",
        action="append",
        default=[],
        help="exact NVIDIA UUID; repeat once per expected GPU",
    )
    parser.add_argument("--port", type=int, default=17681)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument(
        "--eligible-idle-seconds",
        type=float,
        default=6.0,
        help="fail after this much eligible idle time without a typed queue-plan exception",
    )
    parser.add_argument(
        "--source-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--selector-contract-command",
        default=(
            "cargo test -p mold-ai-inference "
            "startup_selection_rejects_ambiguous_or_missing_uuid_prefixes"
        ),
        help="deterministic source test used only for the unconstructable ambiguous-prefix case",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    missing = [
        name
        for name in ("binary", "models_dir", "report")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "required arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if not args.request and not args.model:
        parser.error("one of --request or --model is required")
    if len(args.expected_gpu_uuid) != 2:
        parser.error("--expected-gpu-uuid must be repeated for exactly two GPUs")
    if len(args.expected_gpu_uuid) != len(set(args.expected_gpu_uuid)):
        parser.error("--expected-gpu-uuid values must be unique")
    if args.jobs < 4:
        parser.error("--jobs must be at least 4 to prove queued replanning")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.eligible_idle_seconds <= 0:
        parser.error("--eligible-idle-seconds must be positive")
    try:
        validate_qualification_port(args.port)
    except ValueError as error:
        parser.error(str(error))
    args.binary = args.binary.resolve()
    args.models_dir = args.models_dir.resolve()
    args.report = args.report.resolve()
    args.source_root = args.source_root.resolve()
    args.model_artifact = [path.resolve() for path in args.model_artifact]
    if not args.model_artifact:
        parser.error("--model-artifact must be repeated for every selected model file")
    if args.request:
        args.request = args.request.resolve()
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    for command in ("nvidia-smi", "cargo", "bwrap"):
        if shutil.which(command) is None:
            print(f"required command is unavailable: {command}", file=sys.stderr)
            return 69
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        print(f"candidate binary is not executable: {args.binary}", file=sys.stderr)
        return 66
    if not args.models_dir.is_dir():
        print(f"models directory does not exist: {args.models_dir}", file=sys.stderr)
        return 66
    if not port_is_free(args.port):
        print(
            f"alternate port {args.port} already has a listener; refusing to touch it",
            file=sys.stderr,
        )
        return 73
    evidence_dir = pathlib.Path(str(args.report) + ".d")
    if args.report.exists() or evidence_dir.exists():
        print(
            f"report or evidence path already exists; refusing to overwrite: "
            f"{args.report}, {evidence_dir}",
            file=sys.stderr,
        )
        return 73

    started = utc_now()
    deadline = Deadline(args.timeout_seconds)
    evidence = Evidence(evidence_dir)
    runtime_dir = evidence_dir / "primary-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    api_key = uuid.uuid4().hex + uuid.uuid4().hex
    checks = {
        name: check("failed", "qualification did not reach this gate", [])
        for name in CHECK_NAMES
    }
    server: CandidateServer | None = None
    primary_pid = None
    error_message = None
    expected = args.expected_gpu_uuid
    binary_sha = sha256(args.binary, deadline)
    version = "unavailable"
    source_commit = "0" * 40
    request: dict[str, object] = default_request(args.model or "unavailable")
    request_path = evidence_dir / "normalized-request.json"
    devices: list[dict[str, object]] = []
    model_artifacts: list[dict[str, object]] = []
    artifact_roots: list[str] = []
    models_before: list[dict[str, object]] | None = None
    live_service_before: dict[str, object] | None = None
    try:
        live_service_before = reserved_service_snapshot(deadline)
        evidence.json("live-service-before", live_service_before)
        models_before = models_tree_manifest(args.models_dir, deadline)
        evidence.json("models-tree-before", models_before)
        request_value = (
            json.loads(args.request.read_text(encoding="utf-8"))
            if args.request
            else default_request(args.model)
        )
        if not isinstance(request_value, dict):
            raise ValueError("qualification request must be a JSON object")
        request = request_value
        request["batch_size"] = 1
        if request.get("output_format", "png") != "png":
            raise ValueError("local multi-GPU qualification requires output_format=png")
        if not isinstance(request.get("width"), int) or not isinstance(
            request.get("height"), int
        ):
            raise ValueError("qualification request requires integer width/height")
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence.existing("normalized-request", request_path, "json")
        model_artifacts, artifact_roots = collect_model_artifacts(
            args.models_dir, args.model_artifact, deadline
        )
        evidence.json(
            "model-artifacts",
            {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "model": request.get("model"),
                "artifacts": model_artifacts,
                "artifact_roots": artifact_roots,
            },
        )
        source_commit = git_source_commit(args.source_root, deadline)
        evidence.json(
            "source-provenance",
            {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "commit": source_commit,
                "source_root": str(args.source_root),
            },
        )
        version_result = run_candidate_probe(
            binary=args.binary,
            runtime_dir=evidence_dir / "candidate-probe-runtime",
            argv=["version"],
            deadline=deadline,
        )
        if version_result.returncode != 0:
            raise RuntimeError(
                f"sandboxed candidate version failed: {version_result.stderr}"
            )
        version = version_result.stdout.strip()
        if not version:
            raise RuntimeError("sandboxed candidate version returned no version")
        evidence.json(
            "candidate-version",
            {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "binary": str(args.binary),
                "binary_sha256": binary_sha,
                "version": version,
                "sandboxed": True,
                "argv": version_result.args,
                "environment": {
                    key: sandbox_environment(
                        evidence_dir / "candidate-probe-runtime"
                    ).get(key)
                    for key in (
                        "HOME",
                        "TMPDIR",
                        "XDG_CACHE_HOME",
                        "XDG_CONFIG_HOME",
                        "XDG_DATA_HOME",
                        "MOLD_HOST",
                    )
                },
                "build_identity": {
                    "source_commit": source_commit,
                    "binary_sha256": binary_sha,
                    "version": version,
                },
            },
        )

        devices, raw_inventory = nvidia_inventory(deadline)
        evidence.text("nvidia-inventory", raw_inventory)
        validate_hardware_profile(devices, expected)

        server = CandidateServer(
            binary=args.binary,
            models_dir=args.models_dir,
            runtime_dir=runtime_dir,
            port=args.port,
            api_key=api_key,
            label="primary",
            evidence=evidence,
        )
        server.start(deadline)
        primary_pid = server.pid
        initial = api_snapshot(server.api)
        if initial["status"].get("git_sha") != source_commit:
            raise RuntimeError(
                "candidate embedded git SHA does not match the exact source commit"
            )
        evidence.json("initial-api-projection", initial)
        api_devices, stable_ids = validate_initial_projection(initial, expected)
        installed = [
            model
            for model in initial["models"]
            if model.get("name") == request.get("model") and model.get("downloaded")
        ]
        if not installed:
            raise RuntimeError(
                f"requested model is not installed in the read-only inventory: "
                f"{request.get('model')!r}"
            )
        stable_by_uuid = {
            device["nvml_uuid"]: device["id"] for device in api_devices
        }
        initial_mapping = {
            str(device["id"]): str(device["nvml_uuid"]) for device in api_devices
        }
        checks["both_devices_discovered"] = check(
            "passed",
            "exact expected UUIDs are enabled in devices, status, resources, and queue plan",
            ["nvidia-inventory", "initial-api-projection"],
        )

        cli_payload, cli_stderr, cli_command, cli_environment = run_client_projection(
            server, args.binary, args.port, api_key, stable_ids, deadline
        )
        evidence.json(
            "client-projection",
            {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "server_pid": server.pid,
                "gpu_list": cli_payload,
                "stderr": cli_stderr,
                "argv": cli_command,
                "environment": cli_environment,
            },
        )
        checks["client_projection"] = check(
            "passed",
            "candidate CLI stable-ID projection exactly matches /api/devices",
            ["client-projection", "initial-api-projection"],
        )

        lifecycle = run_parallel_lifecycle(
            server,
            request,
            args.jobs,
            evidence,
            deadline,
            args.eligible_idle_seconds,
        )
        expected_set = set(expected)
        if lifecycle["observed_active_uuids"] != expected_set:
            raise RuntimeError(
                "device API did not observe active work on every expected UUID: "
                f"{sorted(lifecycle['observed_active_uuids'])!r}"
            )
        if lifecycle["observed_compute_uuids"] != expected_set:
            raise RuntimeError(
                "nvidia-smi did not bind the exact server PID to every expected UUID: "
                f"{sorted(lifecycle['observed_compute_uuids'])!r}"
            )
        checks["both_devices_executed"] = check(
            "passed",
            "distinct active work plus exact candidate PID observations prove both GPUs executed",
            ["parallel-runtime-samples", "parallel-results"],
        )
        checks["busy_disable_drained"] = check(
            "passed",
            "busy disable returned draining, allowed current work to finish, then re-enabled",
            ["parallel-runtime-samples", "parallel-results"],
        )
        checks["queue_replanned_after_disable"] = check(
            "passed",
            "plan version advanced and no queued work remained assigned to draining GPU",
            ["parallel-runtime-samples", "parallel-results"],
        )

        run_queued_cancellation(server, request, evidence, deadline)
        checks["queued_cancellation"] = check(
            "passed",
            "paused queued SSE work was cancelled by stable work ID without execution",
            ["queued-cancellation"],
        )

        run_maintenance(server, request, evidence, deadline, initial_mapping)
        checks["all_disabled_maintenance"] = check(
            "passed",
            "all devices disabled cleanly and generation failed with maintenance error",
            ["all-disabled-maintenance"],
        )

        # Persist one disabled preference across an exact-process restart.
        first = stable_ids[0]
        disabled_uuid = initial_mapping[first]
        old_pid = server.pid
        old_process_identity = server.process_identity
        status, body = server.api.json(
            "PATCH",
            f"/api/devices/{urllib.parse.quote(first, safe='')}",
            {"enabled": False},
        )
        if status not in {200, 202}:
            raise RuntimeError(f"pre-restart disable failed: {status} {body}")
        wait_for(
            "device disabled before restart",
            lambda: (
                lambda payload: payload
                if next(
                    device for device in payload["devices"] if device["id"] == first
                )["admin_state"]
                == "disabled"
                else None
            )(server.api.json("GET", "/api/devices")[1]),
            timeout=deadline.remaining(30),
        )
        cleanup_error = server.stop(deadline)
        if cleanup_error:
            raise RuntimeError(cleanup_error)
        server = CandidateServer(
            binary=args.binary,
            models_dir=args.models_dir,
            runtime_dir=runtime_dir,
            port=args.port,
            api_key=api_key,
            label="restart",
            evidence=evidence,
        )
        server.start(deadline)
        new_pid = server.pid
        new_process_identity = server.process_identity
        if new_pid == old_pid:
            raise RuntimeError("candidate restart reused the same process PID")
        _, restarted = server.api.json("GET", "/api/devices")
        restarted_mapping = {
            str(device["id"]): str(device["nvml_uuid"])
            for device in restarted["devices"]
        }
        if restarted_mapping != initial_mapping:
            raise RuntimeError("stable device-ID/UUID mapping changed across restart")
        persisted = next(device for device in restarted["devices"] if device["id"] == first)
        if (
            persisted.get("desired_enabled") is not False
            or persisted.get("nvml_uuid") != disabled_uuid
        ):
            raise RuntimeError("disabled preference did not survive restart")
        status, _ = server.api.json(
            "PATCH",
            f"/api/devices/{urllib.parse.quote(first, safe='')}",
            {"enabled": True},
        )
        if status not in {200, 202}:
            raise RuntimeError("failed to restore persisted-disabled device")
        restored = wait_for(
            "persisted device to re-enable",
            lambda: (
                lambda payload: payload
                if all(
                    device["admin_state"] == "enabled"
                    for device in payload["devices"]
                )
                else None
            )(server.api.json("GET", "/api/devices")[1]),
            timeout=deadline.remaining(30),
        )
        restored_mapping = {
            str(device["id"]): str(device["nvml_uuid"])
            for device in restored["devices"]
        }
        if (
            len(restored["devices"]) != 2
            or restored_mapping != initial_mapping
            or any(
                device.get("desired_enabled") is not True
                or device.get("admin_state") != "enabled"
                for device in restored["devices"]
            )
        ):
            raise RuntimeError("restored inventory is not exactly enabled")
        evidence.json(
            "restart-persistence",
            {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "old_pid": old_pid,
                "new_pid": new_pid,
                "disabled_id": first,
                "disabled_uuid": disabled_uuid,
                "before_mapping": initial_mapping,
                "after_mapping": restarted_mapping,
                "old_process_identity": old_process_identity,
                "new_process_identity": new_process_identity,
                "persisted_devices": restarted,
                "restored_devices": restored,
            },
        )
        checks["restart_persistence"] = check(
            "passed",
            "machine-wide desired enablement survived restart and was restored",
            ["restart-persistence", "restart-server-log"],
        )

        cleanup_error = server.stop(deadline)
        if cleanup_error:
            raise RuntimeError(cleanup_error)
        server = CandidateServer(
            binary=args.binary,
            models_dir=args.models_dir,
            runtime_dir=runtime_dir,
            port=args.port,
            api_key=api_key,
            label="legacy",
            evidence=evidence,
            dispatch_mode="legacy",
        )
        server.start(deadline)
        legacy = api_snapshot(server.api)
        capabilities = legacy["capabilities"]
        legacy_mapping = {
            str(device["id"]): str(device["nvml_uuid"])
            for device in legacy["devices"].get("devices", [])
        }
        if (
            capabilities.get("dispatch", {}).get("active_mode") != "legacy"
            or capabilities.get("dispatch", {}).get("v2_authoritative")
            or capabilities.get("devices", {}).get("lifecycle")
            or len(legacy["devices"].get("devices", [])) != len(expected)
            or legacy_mapping != initial_mapping
        ):
            raise RuntimeError("legacy rollback projection is not fail-closed")
        target = legacy["devices"]["devices"][0]["id"]
        patch_status, patch_body = server.api.json(
            "PATCH",
            f"/api/devices/{urllib.parse.quote(target, safe='')}",
            {"enabled": False},
        )
        if (
            patch_status != 409
            or patch_body.get("code") != "DEVICE_LIFECYCLE_MODE_CONFLICT"
            or "authoritative scheduler V2" not in patch_body.get("error", "")
        ):
            raise RuntimeError(
                f"legacy lifecycle mutation was not fenced: {patch_status} {patch_body}"
            )
        evidence.json(
            "legacy-rollback",
            {
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "server_pid": server.pid,
                "process_identity": server.process_identity,
                "device_mapping": legacy_mapping,
                "snapshot": legacy,
                "patch_target": target,
                "patch_status": patch_status,
                "patch_body": patch_body,
            },
        )
        checks["legacy_rollback"] = check(
            "passed",
            "legacy mode kept both devices visible but rejected live lifecycle mutation",
            ["legacy-rollback", "legacy-server-log"],
        )
        cleanup_error = server.stop(deadline)
        if cleanup_error:
            raise RuntimeError(cleanup_error)
        server = None

        physical_order = [
            str(device["uuid"])
            for device in sorted(devices, key=lambda device: int(device["index"]))
        ]
        run_selector_matrix(
            args, evidence, api_key, physical_order, stable_by_uuid, deadline
        )
        checks["selector_matrix"] = check(
            "passed",
            "real selector/reorder matrix passed; ambiguity is separately "
            "deterministic and not hardware-claimed",
            ["selector-matrix", "ambiguous-selector-source-contract"],
        )

    except Exception as error:  # report durable failure evidence before returning
        error_message = f"{type(error).__name__}: {error}"
    finally:
        if (
            primary_pid is None
            and server is not None
            and server.server_pid is not None
        ):
            primary_pid = server.pid
        if server is not None:
            cleanup_error = server.stop(deadline)
            if cleanup_error:
                error_message = (
                    f"{error_message}; RuntimeError: {cleanup_error}"
                    if error_message
                    else f"RuntimeError: {cleanup_error}"
                )
        if models_before is not None:
            try:
                models_after = models_tree_manifest(args.models_dir, deadline)
                if not evidence.contains("models-tree-after"):
                    evidence.json("models-tree-after", models_after)
                if models_after == models_before:
                    checks["models_tree_unchanged"] = check(
                        "passed",
                        "installed model tree path/type/mode/size/mtime manifest is unchanged",
                        ["models-tree-before", "models-tree-after"],
                    )
                else:
                    mutation_error = (
                        "read-only installed models tree changed during qualification"
                    )
                    error_message = (
                        f"{error_message}; RuntimeError: {mutation_error}"
                        if error_message
                        else f"RuntimeError: {mutation_error}"
                    )
            except Exception as manifest_error:
                error_message = (
                    f"{error_message}; model post-check failed: {manifest_error}"
                    if error_message
                    else f"model post-check failed: {manifest_error}"
                )
        if live_service_before is not None:
            try:
                live_service_after = reserved_service_snapshot(deadline)
                if not evidence.contains("live-service-after"):
                    evidence.json("live-service-after", live_service_after)
                if live_service_after == live_service_before:
                    checks["live_service_unchanged"] = check(
                        "passed",
                        "reserved live-service listener PID/start-time identity is unchanged",
                        ["live-service-before", "live-service-after"],
                    )
                else:
                    live_error = (
                        "reserved live-service listener identity changed during qualification"
                    )
                    error_message = (
                        f"{error_message}; RuntimeError: {live_error}"
                        if error_message
                        else f"RuntimeError: {live_error}"
                    )
            except Exception as live_error:
                error_message = (
                    f"{error_message}; live-service post-check failed: {live_error}"
                    if error_message
                    else f"live-service post-check failed: {live_error}"
                )
        if error_message and not evidence.contains("qualification-error"):
            evidence.text("qualification-error", error_message + "\n")

    if not evidence.contains("normalized-request"):
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence.existing("normalized-request", request_path, "json")

    finished = utc_now()
    all_passed = all(item["status"] == "passed" for item in checks.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "qualification_profile": QUALIFICATION_PROFILE,
        "source_commit": source_commit,
        "started_at": started,
        "finished_at": finished,
        "hardware_qualified": all_passed,
        "candidate": {
            "path": str(args.binary),
            "sha256": binary_sha,
            "version": version,
            "server_pid": primary_pid,
        },
        "host": {
            "hostname": socket.gethostname(),
            "expected_gpu_uuids": expected,
            "devices": devices,
        },
        "isolation": {
            "bind": "127.0.0.1",
            "port": args.port,
            "mold_home": str(runtime_dir / "home"),
            "db_path": str(runtime_dir / "mold.db"),
            "output_dir": str(runtime_dir / "output"),
            "models_dir": str(args.models_dir),
            "inherited_home": os.environ.get("HOME", ""),
            "preexisting_listener_absent": True,
        },
        "request": {
            "path": str(request_path),
            "sha256": sha256(request_path),
            "model": str(request.get("model", "unavailable")),
            "job_count": args.jobs,
            "artifacts": model_artifacts,
            "artifact_roots": artifact_roots,
        },
        "checks": checks,
        "evidence": evidence.items,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validator = pathlib.Path(__file__).with_name(
        "validate-local-multi-gpu-report.py"
    )
    command = [str(validator), str(args.report)]
    if not all_passed:
        command.append("--allow-failure")
    validation = run_process_group(
        command,
        env=os.environ.copy(),
        deadline=Deadline(60),
    )
    if validation.returncode != 0:
        print(validation.stderr, file=sys.stderr)
        return 1
    print(validation.stdout.strip())
    if error_message:
        print(f"qualification failed: {error_message}", file=sys.stderr)
    print(f"report: {args.report}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
