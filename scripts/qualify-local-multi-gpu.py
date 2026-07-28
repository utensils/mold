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
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

SCHEMA_VERSION = "mold.local.multi-gpu.qualification.v1"
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
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def models_tree_manifest(root: pathlib.Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
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


class Evidence:
    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.items: list[dict[str, str]] = []
        self._labels: set[str] = set()

    def _add(self, label: str, path: pathlib.Path) -> pathlib.Path:
        if label in self._labels:
            raise ValueError(f"duplicate evidence label: {label}")
        self._labels.add(label)
        self.items.append(
            {"label": label, "path": str(path.resolve()), "sha256": sha256(path)}
        )
        return path

    def text(self, label: str, value: str) -> pathlib.Path:
        path = self.directory / f"{label}.txt"
        path.write_text(value, encoding="utf-8")
        return self._add(label, path)

    def json(self, label: str, value: object) -> pathlib.Path:
        path = self.directory / f"{label}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self._add(label, path)

    def existing(self, label: str, path: pathlib.Path) -> pathlib.Path:
        return self._add(label, path)


class Api:
    def __init__(self, port: int, api_key: str):
        self.base = f"http://127.0.0.1:{port}"
        self.api_key = api_key

    def request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        timeout: float = 30,
    ) -> tuple[int, dict[str, str], bytes]:
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
        # The whole host is read-only inside the candidate mount namespace.
        # Only this run's private runtime tree and temporary directory are
        # overlaid writable. NVIDIA device nodes remain available.
        return [
            "bwrap",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            "--bind",
            str(self.runtime_dir),
            str(self.runtime_dir),
            "--bind",
            str(self.runtime_dir / "tmp"),
            "/tmp",
            "--",
            *self.inner_command(),
        ]

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "MOLD_HOME": str(self.home),
                "MOLD_DB_PATH": str(self.db),
                "MOLD_OUTPUT_DIR": str(self.output),
                "MOLD_API_KEY": self.api_key,
                "MOLD_DISPATCH_MODE": self.dispatch_mode,
                "MOLD_MDNS": "0",
                "MOLD_LOG": "info",
                "HOME": str(self.home),
                "TMPDIR": str(self.runtime_dir / "tmp"),
                "XDG_CACHE_HOME": str(self.runtime_dir / "cache" / "xdg"),
                "XDG_CONFIG_HOME": str(self.runtime_dir / "config"),
                "XDG_DATA_HOME": str(self.runtime_dir / "data"),
                "CUDA_CACHE_PATH": str(self.runtime_dir / "cache" / "cuda"),
                "HF_HOME": str(self.runtime_dir / "cache" / "huggingface"),
            }
        )
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.empty_gpu_env:
            env["MOLD_GPUS"] = ""
        elif self.gpus is None:
            env.pop("MOLD_GPUS", None)
        return env

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

    def start(self, timeout: float = 45) -> None:
        self.prepare_runtime()
        self.log_handle = self.log_path.open("ab", buffering=0)
        command_path = self.evidence.directory / f"{self.label}-command.json"
        command_path.write_text(
            json.dumps(
                {
                    "argv": self.command(),
                    "environment": {
                        key: self.environment().get(key)
                        for key in (
                            "MOLD_HOME",
                            "MOLD_DB_PATH",
                            "MOLD_OUTPUT_DIR",
                            "MOLD_DISPATCH_MODE",
                            "MOLD_MDNS",
                            "MOLD_GPUS",
                            "CUDA_VISIBLE_DEVICES",
                        )
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.evidence.existing(f"{self.label}-command", command_path)
        self.process = subprocess.Popen(
            self.command(),
            env=self.environment(),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.label} server exited {self.process.returncode}; "
                    f"see {self.log_path}"
                )
            try:
                status, _ = self.api.json("GET", "/api/status", timeout=2)
                if status == 200:
                    self.server_pid = self._resolve_candidate_pid()
                    return
                last_error = f"HTTP {status}"
            except (OSError, RuntimeError) as error:
                last_error = str(error)
            time.sleep(0.2)
        raise TimeoutError(f"{self.label} server did not become ready: {last_error}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        if self.log_path.exists() and self.label + "-server-log" not in {
            item["label"] for item in self.evidence.items
        }:
            self.evidence.existing(f"{self.label}-server-log", self.log_path)

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


def nvidia_inventory() -> tuple[list[dict[str, object]], str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
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


def compute_apps() -> tuple[list[dict[str, object]], str]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
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
) -> tuple[dict[str, object], str]:
    env = server.environment()
    env["MOLD_HOST"] = f"http://127.0.0.1:{port}"
    env["MOLD_API_KEY"] = api_key
    result = subprocess.run(
        [str(binary), "gpu", "list", "--json"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    ids = [device["id"] for device in payload.get("devices", [])]
    if ids != expected_ids:
        raise RuntimeError(f"CLI projection IDs differ: {ids!r} != {expected_ids!r}")
    return payload, result.stderr


def run_parallel_lifecycle(
    server: CandidateServer,
    request: dict[str, object],
    jobs: int,
    evidence: Evidence,
    timeout: float,
    eligible_idle_seconds: float,
) -> dict[str, object]:
    sample_path = evidence.directory / "parallel-runtime-samples.jsonl"
    lock = threading.Lock()

    def generate(index: int) -> dict[str, object]:
        child = dict(request)
        child["prompt"] = f"{request['prompt']} variation {index + 1}"
        child["seed"] = int(request.get("seed", 0)) + index
        child["batch_size"] = 1
        status, headers, payload = server.api.request(
            "POST", "/api/generate", child, timeout=timeout
        )
        output = evidence.directory / f"parallel-output-{index + 1}.bin"
        output.write_bytes(payload)
        with lock:
            evidence.existing(f"parallel-output-{index + 1}", output)
        return {
            "index": index + 1,
            "status": status,
            "headers": {key.lower(): value for key, value in headers.items()},
            "size": len(payload),
            "sha256": sha256(output),
        }

    observed_compute_uuids: set[str] = set()
    observed_active_uuids: set[str] = set()
    disabled_id = None
    disabled_uuid = None
    initial_plan_version = None
    replanned = False
    disable_response = None
    unexcused_idle_started: float | None = None
    maximum_unexcused_idle_seconds = 0.0
    typed_idle_exceptions: list[dict[str, object]] = []

    with sample_path.open("w", encoding="utf-8") as samples, concurrent.futures.ThreadPoolExecutor(
        max_workers=jobs
    ) as pool:
        futures = [pool.submit(generate, index) for index in range(jobs)]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, devices_payload = server.api.json("GET", "/api/devices")
            _, queue_payload = server.api.json("GET", "/api/queue")
            apps, raw_apps = compute_apps()
            devices = devices_payload["devices"]
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
            sample = {
                "at": utc_now(),
                "server_pid": server.pid,
                "devices": devices,
                "queue": queue_payload,
                "compute_apps": apps,
                "compute_apps_raw": raw_apps,
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
                    chosen = active[0]
                    disabled_id = chosen["id"]
                    disabled_uuid = chosen.get("nvml_uuid")
                    initial_plan_version = queue_payload.get("plan", {}).get("plan_version")
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
                queued_on_disabled = [
                    item
                    for item in plan.get("work_items", [])
                    if item.get("planned_device_id") == disabled_id
                    and item.get("activity_phase") in {"queued", "planned", "blocked"}
                ]
                version = plan.get("plan_version")
                if (
                    isinstance(version, int)
                    and isinstance(initial_plan_version, int)
                    and version > initial_plan_version
                    and not queued_on_disabled
                ):
                    replanned = True

            if all(future.done() for future in futures):
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("parallel generation workload did not finish")
        results = [future.result() for future in futures]

    evidence.existing("parallel-runtime-samples", sample_path)
    if disabled_id is None:
        raise RuntimeError("never observed two distinct active GPU work IDs")
    if not replanned:
        raise RuntimeError("queue plan never advanced away from the draining device")

    def disabled_state():
        _, body = server.api.json("GET", "/api/devices")
        device = next(item for item in body["devices"] if item["id"] == disabled_id)
        return device if device.get("admin_state") == "disabled" else None

    drained = wait_for("busy device to finish and disable", disabled_state, timeout=30)
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

    reenabled = wait_for("device to re-enable", enabled_state, timeout=30)
    results_path = evidence.json(
        "parallel-results",
        {
            "server_pid": server.pid,
            "results": results,
            "observed_active_uuids": sorted(observed_active_uuids),
            "observed_compute_uuids": sorted(observed_compute_uuids),
            "disabled_id": disabled_id,
            "disabled_uuid": disabled_uuid,
            "disable_response": disable_response,
            "drained": drained,
            "reenabled": reenabled,
            "queue_replanned": replanned,
            "maximum_unexcused_idle_seconds": maximum_unexcused_idle_seconds,
            "eligible_idle_limit_seconds": eligible_idle_seconds,
            "typed_idle_exceptions": typed_idle_exceptions,
        },
    )
    del results_path
    if any(result["status"] != 200 or result["size"] == 0 for result in results):
        raise RuntimeError(f"one or more parallel generations failed: {results!r}")
    response_gpus = {
        int(result["headers"]["x-mold-gpu"])
        for result in results
        if "x-mold-gpu" in result["headers"]
    }
    if len(response_gpus) < 2:
        raise RuntimeError(f"generation headers did not prove two GPUs: {response_gpus}")
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
    timeout: float,
) -> None:
    status, pause = server.api.json("POST", "/api/queue/pause")
    if status != 200 or not pause.get("paused"):
        raise RuntimeError(f"queue pause failed: {status} {pause}")
    _, before = server.api.json("GET", "/api/queue")
    before_ids = {entry["id"] for entry in before.get("entries", [])}
    result: dict[str, object] = {}

    def stream():
        stream_request = dict(request)
        stream_request["prompt"] = f"{request['prompt']} queued cancellation"
        result["response"] = server.api.request(
            "POST", "/api/generate/stream", stream_request, timeout=timeout
        )

    thread = threading.Thread(target=stream, daemon=True)
    thread.start()

    def queued_id():
        _, listing = server.api.json("GET", "/api/queue")
        candidates = [
            entry["id"]
            for entry in listing.get("entries", [])
            if entry["id"] not in before_ids
        ]
        return candidates[0] if len(candidates) == 1 else None

    job_id = wait_for("paused SSE job to enter queue", queued_id, timeout=20)
    cancel_status, _, cancel_payload = server.api.request(
        "DELETE", f"/api/queue/{urllib.parse.quote(job_id, safe='')}"
    )
    cancel_body = cancel_payload.decode(errors="replace")
    resume_status, resume = server.api.json("POST", "/api/queue/resume")
    thread.join(timeout=20)
    if thread.is_alive():
        raise RuntimeError("cancelled SSE request did not terminate")
    if cancel_status != 204 or resume_status != 200 or resume.get("paused"):
        raise RuntimeError(
            f"queued cancel/resume failed: {cancel_status} {cancel_body} "
            f"{resume_status} {resume}"
        )
    _, after = server.api.json("GET", "/api/queue")
    if any(entry["id"] == job_id for entry in after.get("entries", [])):
        raise RuntimeError("cancelled queued job remained in queue")
    response_status, _, response_body = result["response"]
    evidence.json(
        "queued-cancellation",
        {
            "job_id": job_id,
            "cancel_status": cancel_status,
            "cancel_body": cancel_body,
            "resume_status": resume_status,
            "stream_http_status": response_status,
            "stream_tail": response_body[-2000:].decode(errors="replace"),
            "queue_after": after,
        },
    )


def set_all_devices(server: CandidateServer, enabled: bool, timeout: float = 30) -> None:
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

    wait_for(f"all devices to become {target}", all_target, timeout=timeout)


def run_maintenance(server: CandidateServer, request: dict[str, object], evidence: Evidence) -> None:
    set_all_devices(server, False)
    status, _, body = server.api.request("POST", "/api/generate", request, timeout=10)
    text = body.decode(errors="replace")
    if status not in {409, 503} or "maintenance" not in text.lower():
        raise RuntimeError(
            f"all-disabled generation was not rejected as maintenance: {status} {text}"
        )
    _, devices = server.api.json("GET", "/api/devices")
    evidence.json(
        "all-disabled-maintenance",
        {"status": status, "body": text, "devices": devices},
    )
    set_all_devices(server, True)


def start_selector_server(
    *,
    args,
    evidence: Evidence,
    api_key: str,
    label: str,
    gpus: str | None,
    expected_uuids: set[str],
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
        server.start()
        _, payload = server.api.json("GET", "/api/devices")
        selected = {
            device["nvml_uuid"]
            for device in payload["devices"]
            if device.get("admin_state") != "startup_excluded"
        }
        if selected != expected_uuids:
            raise RuntimeError(
                f"selector {label} selected {sorted(selected)!r}, "
                f"expected {sorted(expected_uuids)!r}"
            )
        return {"label": label, "devices": payload, "pid": server.pid}
    finally:
        server.stop()


def run_selector_matrix(
    args,
    evidence: Evidence,
    api_key: str,
    expected_uuids: list[str],
    stable_ids_by_uuid: dict[str, str],
) -> None:
    all_set = set(expected_uuids)
    first_uuid, second_uuid = expected_uuids[:2]
    results = [
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="empty",
            gpus=None,
            expected_uuids=all_set,
            empty_gpu_env=True,
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="all",
            gpus="all",
            expected_uuids=all_set,
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="ordinal-one",
            gpus="1",
            expected_uuids={second_uuid},
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="stable-id",
            gpus=stable_ids_by_uuid[first_uuid],
            expected_uuids={first_uuid},
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="nvidia-uuid",
            gpus=second_uuid,
            expected_uuids={second_uuid},
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="none",
            gpus="none",
            expected_uuids=set(),
        ),
        start_selector_server(
            args=args,
            evidence=evidence,
            api_key=api_key,
            label="reordered-stable",
            gpus=stable_ids_by_uuid[first_uuid],
            expected_uuids={first_uuid},
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
    with missing.log_path.open("wb") as log:
        process = subprocess.run(
            missing.command(),
            env=missing.environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    evidence.existing("selector-missing-server-log", missing.log_path)
    if process.returncode == 0:
        raise RuntimeError("unmatched NVIDIA UUID selector unexpectedly started")
    if "did not match" not in missing.log_path.read_text(errors="replace"):
        raise RuntimeError("unmatched selector did not report a typed match failure")
    results.append(
        {
            "label": "missing",
            "exit_code": process.returncode,
            "expected_failure": "did not match",
        }
    )

    command = subprocess.run(
        args.selector_contract_command,
        cwd=args.source_root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    contract_path = evidence.text(
        "ambiguous-selector-source-contract",
        json.dumps(
            {
                "command": args.selector_contract_command,
                "cwd": str(args.source_root),
                "exit_code": command.returncode,
                "stdout": command.stdout,
                "stderr": command.stderr,
                "hardware_prefix": common_hex_prefix(list(stable_ids_by_uuid.values())),
                "hardware_claimed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    del contract_path
    if command.returncode != 0:
        raise RuntimeError("deterministic ambiguous selector contract failed")
    evidence.json("selector-matrix", {"scenarios": results})


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
        parser.error("required arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    if not args.request and not args.model:
        parser.error("one of --request or --model is required")
    if len(args.expected_gpu_uuid) < 2:
        parser.error("--expected-gpu-uuid must be repeated for at least two GPUs")
    if len(args.expected_gpu_uuid) != len(set(args.expected_gpu_uuid)):
        parser.error("--expected-gpu-uuid values must be unique")
    if args.jobs < 2:
        parser.error("--jobs must be at least 2")
    if args.eligible_idle_seconds <= 0:
        parser.error("--eligible-idle-seconds must be positive")
    if not (1024 <= args.port <= 65535):
        parser.error("--port must be in 1024..65535")
    args.binary = args.binary.resolve()
    args.models_dir = args.models_dir.resolve()
    args.report = args.report.resolve()
    args.source_root = args.source_root.resolve()
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

    binary_sha = sha256(args.binary)
    version = subprocess.run(
        [str(args.binary), "version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    request = (
        json.loads(args.request.read_text(encoding="utf-8"))
        if args.request
        else default_request(args.model)
    )
    request["batch_size"] = 1
    request_path = evidence_dir / "normalized-request.json"
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence.existing("normalized-request", request_path)

    devices, raw_inventory = nvidia_inventory()
    evidence.text("nvidia-inventory", raw_inventory)
    expected = args.expected_gpu_uuid
    observed = [str(device["uuid"]) for device in devices]
    if set(observed) != set(expected) or len(observed) != len(expected):
        print(
            f"host inventory {observed!r} is not exact expected inventory {expected!r}",
            file=sys.stderr,
        )
        return 65
    models_before = models_tree_manifest(args.models_dir)
    evidence.json("models-tree-before", models_before)

    try:
        server = CandidateServer(
            binary=args.binary,
            models_dir=args.models_dir,
            runtime_dir=runtime_dir,
            port=args.port,
            api_key=api_key,
            label="primary",
            evidence=evidence,
        )
        server.start()
        primary_pid = server.pid
        initial = api_snapshot(server.api)
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
        checks["both_devices_discovered"] = check(
            "passed",
            "exact expected UUIDs are enabled in devices, status, resources, and queue plan",
            ["nvidia-inventory", "initial-api-projection"],
        )

        cli_payload, cli_stderr = run_client_projection(
            server, args.binary, args.port, api_key, stable_ids
        )
        evidence.json(
            "client-projection",
            {"gpu_list": cli_payload, "stderr": cli_stderr},
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
            args.timeout_seconds,
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

        run_queued_cancellation(server, request, evidence, args.timeout_seconds)
        checks["queued_cancellation"] = check(
            "passed",
            "paused queued SSE work was cancelled by stable work ID without execution",
            ["queued-cancellation"],
        )

        run_maintenance(server, request, evidence)
        checks["all_disabled_maintenance"] = check(
            "passed",
            "all devices disabled cleanly and generation failed with maintenance error",
            ["all-disabled-maintenance"],
        )

        # Persist one disabled preference across an exact-process restart.
        first = stable_ids[0]
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
            timeout=30,
        )
        server.stop()
        server = CandidateServer(
            binary=args.binary,
            models_dir=args.models_dir,
            runtime_dir=runtime_dir,
            port=args.port,
            api_key=api_key,
            label="restart",
            evidence=evidence,
        )
        server.start()
        _, restarted = server.api.json("GET", "/api/devices")
        persisted = next(device for device in restarted["devices"] if device["id"] == first)
        if persisted.get("desired_enabled") is not False:
            raise RuntimeError("disabled preference did not survive restart")
        evidence.json("restart-persistence", restarted)
        status, _ = server.api.json(
            "PATCH",
            f"/api/devices/{urllib.parse.quote(first, safe='')}",
            {"enabled": True},
        )
        if status not in {200, 202}:
            raise RuntimeError("failed to restore persisted-disabled device")
        wait_for(
            "persisted device to re-enable",
            lambda: (
                lambda payload: payload
                if all(
                    device["admin_state"] == "enabled"
                    for device in payload["devices"]
                )
                else None
            )(server.api.json("GET", "/api/devices")[1]),
            timeout=30,
        )
        checks["restart_persistence"] = check(
            "passed",
            "machine-wide desired enablement survived restart and was restored",
            ["restart-persistence", "restart-server-log"],
        )

        server.stop()
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
        server.start()
        legacy = api_snapshot(server.api)
        capabilities = legacy["capabilities"]
        if (
            capabilities.get("dispatch", {}).get("active_mode") != "legacy"
            or capabilities.get("dispatch", {}).get("v2_authoritative")
            or capabilities.get("devices", {}).get("lifecycle")
            or len(legacy["devices"].get("devices", [])) != len(expected)
        ):
            raise RuntimeError("legacy rollback projection is not fail-closed")
        target = legacy["devices"]["devices"][0]["id"]
        patch_status, patch_body = server.api.json(
            "PATCH",
            f"/api/devices/{urllib.parse.quote(target, safe='')}",
            {"enabled": False},
        )
        if patch_status != 409:
            raise RuntimeError(
                f"legacy lifecycle mutation was not fenced: {patch_status} {patch_body}"
            )
        evidence.json(
            "legacy-rollback",
            {"snapshot": legacy, "patch_status": patch_status, "patch_body": patch_body},
        )
        checks["legacy_rollback"] = check(
            "passed",
            "legacy mode kept both devices visible but rejected live lifecycle mutation",
            ["legacy-rollback", "legacy-server-log"],
        )
        server.stop()
        server = None

        physical_order = [
            str(device["uuid"])
            for device in sorted(devices, key=lambda device: int(device["index"]))
        ]
        run_selector_matrix(
            args, evidence, api_key, physical_order, stable_by_uuid
        )
        checks["selector_matrix"] = check(
            "passed",
            "real selector/reorder matrix passed; ambiguity is separately deterministic and not hardware-claimed",
            ["selector-matrix", "ambiguous-selector-source-contract"],
        )

        models_after = models_tree_manifest(args.models_dir)
        evidence.json("models-tree-after", models_after)
        if models_after != models_before:
            raise RuntimeError("read-only installed models tree changed during qualification")
        checks["models_tree_unchanged"] = check(
            "passed",
            "installed model tree path/type/mode/size/mtime manifest is unchanged",
            ["models-tree-before", "models-tree-after"],
        )
    except Exception as error:  # report durable failure evidence before returning
        error_message = f"{type(error).__name__}: {error}"
        evidence.text("qualification-error", error_message + "\n")
    finally:
        if (
            primary_pid is None
            and server is not None
            and server.server_pid is not None
        ):
            primary_pid = server.pid
        if server is not None:
            server.stop()

    finished = utc_now()
    all_passed = all(item["status"] == "passed" for item in checks.values())
    report = {
        "schema_version": SCHEMA_VERSION,
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
            "preexisting_listener_absent": True,
        },
        "request": {
            "path": str(request_path),
            "sha256": sha256(request_path),
            "model": str(request["model"]),
            "job_count": args.jobs,
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
    validation = subprocess.run(command, capture_output=True, text=True)
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
