#!/usr/bin/env python3
"""Capture a CUDA experiment without deleting inputs, weights or failed outputs.

This records execution evidence, not a parity verdict. Commands receive a fresh
MOLD_OUTPUT_DIR and MOLD_DB_PATH beneath the evidence root, with the existing
MOLD_HOME and model store retained. Use {output_dir} in an argv element to name
an output file. Never use this launcher for a server sharing production's queue
owner; it is for forced-local mold runs and upstream oracle processes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_roots(home, models, evidence_root):
    home = Path(home).resolve(strict=True)
    models = Path(models).resolve(strict=True)
    evidence_root = Path(evidence_root).resolve()
    if not home.is_dir() or not models.is_dir():
        raise ValueError("MOLD_HOME and MOLD_MODELS_DIR must be existing directories")
    if not evidence_root.is_relative_to(home / "output"):
        raise ValueError("evidence must be under MOLD_HOME/output")
    if evidence_root.is_relative_to(models) or models.is_relative_to(evidence_root):
        raise ValueError("evidence and model storage must not overlap")
    return home, models, evidence_root


def validate_outputs(outputs):
    if not outputs:
        raise ValueError("at least one expected output is required")
    for name in outputs:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError(f"output must be a relative artifact path: {name}")
        if path.parts[0] in {"run.json", "stdout.log", "stderr.log", "gpu.jsonl", "runtime.db"}:
            raise ValueError(f"reserved capture path: {name}")


def sample_nvidia(gpu_uuid):
    data = subprocess.check_output(
        ["nvidia-smi", "-i", gpu_uuid,
         "--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    ).strip().split(", ")
    if len(data) != 6 or data[0] != gpu_uuid:
        raise ValueError("nvidia-smi did not identify the requested stable GPU UUID")
    return {"uuid": data[0], "name": data[1], "driver": data[2],
            "total_mib": int(data[3]), "used_mib": int(data[4]),
            "utilization_percent": int(data[5])}


def git_state():
    def read(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    return {"commit": read("rev-parse", "HEAD"),
            "dirty": bool(read("status", "--porcelain"))}


def capture(*, home, models, evidence_root, gpu_uuid, command, inputs,
            model_files, expected_outputs, sample_gpu=sample_nvidia,
            timeout=14400, upstream=None):
    home, models, evidence_root = validate_roots(home, models, evidence_root)
    validate_outputs(expected_outputs)
    if not command or not gpu_uuid.startswith("GPU-"):
        raise ValueError("command and stable GPU UUID are required")
    input_facts = [identity(path) for path in inputs]
    model_facts = [identity(path) for path in model_files]
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(command[0])
    executable_facts = identity(executable)
    initial_gpu = sample_gpu(gpu_uuid)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = evidence_root / f"capture-{stamp}-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    argv = [str(arg).replace("{output_dir}", str(directory)) for arg in command]
    env = dict(os.environ)
    env.update(MOLD_HOME=str(home), MOLD_MODELS_DIR=str(models),
               MOLD_OUTPUT_DIR=str(directory), MOLD_DB_PATH=str(directory / "runtime.db"),
               CUDA_VISIBLE_DEVICES=gpu_uuid)
    report = {
        "schema": "mold.hunyuan3d.capture.v1", "status": "running",
        "started_utc": stamp, "source": git_state(), "upstream": upstream,
        "command": argv, "executable": executable_facts,
        "gpu_uuid": gpu_uuid, "gpu_initial": initial_gpu,
        "inputs": input_facts, "model_files": model_facts,
        "environment": {key: env[key] for key in
                        ("MOLD_HOME", "MOLD_MODELS_DIR", "MOLD_OUTPUT_DIR",
                         "MOLD_DB_PATH", "CUDA_VISIBLE_DEVICES")},
        "expected_outputs": expected_outputs,
        "gpu_board_used_mib_max": initial_gpu["used_mib"],
        "measurement_note": "Board memory is sampled, includes other processes, and is not an allocation peak.",
    }

    def save():
        staging = directory / "run.json.tmp"
        staging.write_text(json.dumps(report, indent=2) + "\n")
        staging.replace(directory / "run.json")

    save()
    started = time.monotonic()
    code = 1
    process = None
    try:
        with (directory / "stdout.log").open("wb") as stdout, \
                (directory / "stderr.log").open("wb") as stderr, \
                (directory / "gpu.jsonl").open("w") as samples:
            process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if timeout and elapsed > timeout:
                    report["error"] = f"capture exceeded {timeout} seconds"
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    code = 124
                    break
                try:
                    sample = sample_gpu(gpu_uuid)
                    report["gpu_board_used_mib_max"] = max(
                        report["gpu_board_used_mib_max"], sample["used_mib"])
                    samples.write(json.dumps({"elapsed_seconds": elapsed, **sample}) + "\n")
                    samples.flush()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    report.setdefault("telemetry_errors", []).append(str(error))
                time.sleep(0.1)
            else:
                code = process.returncode
    except (OSError, KeyboardInterrupt) as error:
        report["error"] = str(error) or type(error).__name__
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["process_exit_code"] = code
        report["missing_outputs"] = [name for name in expected_outputs
                                     if not (directory / name).is_file()
                                     or (directory / name).stat().st_size == 0]
        if code == 0 and (report["missing_outputs"] or report.get("telemetry_errors")):
            code = 1
        report["exit_code"] = code
        report["status"] = "captured" if code == 0 else "failed"
        report["outputs"] = [identity(directory / name) for name in expected_outputs
                             if name not in report["missing_outputs"]]
        save()
    return directory, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--input", action="append", default=[], type=Path)
    parser.add_argument("--model-file", action="append", required=True, type=Path)
    parser.add_argument("--expect", action="append", required=True)
    parser.add_argument("--upstream", help="oracle repository and pinned commit")
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    home = Path(os.environ.get("MOLD_HOME", "/storage/mold"))
    models = Path(os.environ.get("MOLD_MODELS_DIR", str(home / "models")))
    directory, code = capture(
        home=home, models=models,
        evidence_root=home / "output/verification/hunyuan3d/campaign-1511-1496",
        gpu_uuid=args.gpu_uuid, command=command, inputs=args.input,
        model_files=args.model_file, expected_outputs=args.expect,
        timeout=args.timeout, upstream=args.upstream,
    )
    print(directory)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
