#!/usr/bin/env python3
"""Relational validator for CUDA qualification evidence.

JSON Schema describes the wire shape; this validator enforces cross-field
relationships that must never be inferred from a caller-set boolean.
"""

import hashlib
import json
import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"invalid CUDA qualification report: {message}", file=sys.stderr)
    raise SystemExit(1)


if len(sys.argv) != 2:
    fail("usage: validate-cuda-qualification-report.py <report.json>")

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

if report.get("schema_version") != "mold.cuda.sm86.qualification.v4":
    fail("unexpected schema_version")
if not re.fullmatch(r"[0-9a-f]{40}", report.get("source_sha", "")):
    fail("invalid source_sha")
if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", report.get("release_tag", "")):
    fail("invalid release_tag")

artifacts = report.get("artifacts", {})
sm86 = artifacts.get("sm86", {})
for target, artifact in (("sm_86", sm86),):
    if artifact.get("cuda_target") != target:
        fail(f"{target} artifact target mismatch")
    if not artifact.get("trusted_checksum_verified"):
        fail(f"{target} trusted checksum was not verified")
    if not artifact.get("elf_target_verified"):
        fail(f"{target} ELF/CUDA target was not verified")
    if not artifact.get("ptx_target_verified"):
        fail(f"{target} PTX target was not verified")
    if not artifact.get("source_identity_verified"):
        fail(f"{target} source identity was not verified")
    if artifact.get("expected_sha256") != artifact.get("actual_sha256"):
        fail(f"{target} checksum mismatch")

devices = report.get("host", {}).get("devices", [])
device_uuids = {device.get("uuid") for device in devices}
if not devices:
    fail("no qualified devices")
if any(
    "RTX 3090" not in device.get("name", "").upper()
    or device.get("compute_capability") != "8.6"
    for device in devices
):
    fail("qualification device is not RTX 3090 compute capability 8.6")

required_tests = {
    "sm86_attention_image_smoke": "image",
    "sm86_ptx_image_smoke": "image",
    "sm86_video_smoke": "video",
    "sm86_chained_video_smoke": "video",
}
tests = report.get("tests", {})
for name, media_kind in required_tests.items():
    result = tests.get(name)
    if not isinstance(result, dict):
        fail(f"missing {name}")
    if result.get("status") == "passed":
        if result.get("exit_code") != 0:
            fail(f"{name} passed with nonzero exit")
        if result.get("selected_gpu_uuid") not in device_uuids:
            fail(f"{name} lacks a selected qualified GPU UUID")
        if not result.get("cuda_work_observed"):
            fail(f"{name} lacks CUDA work evidence")
        if not result.get("media_decoded"):
            fail(f"{name} output was not decoded")
        if result.get("width") != 256 or result.get("height") != 256:
            fail(f"{name} dimensions were not 256x256")
        if not re.fullmatch(r"[0-9a-f]{64}", result.get("output_sha256", "")):
            fail(f"{name} lacks output checksum")
        if not re.fullmatch(r"[0-9a-f]{64}", result.get("log_sha256", "")):
            fail(f"{name} lacks log checksum")
        if media_kind == "video" and result.get("frame_count", 0) < 1:
            fail(f"{name} has no decoded frames")

ptx_result = tests.get("sm86_ptx_image_smoke", {})
if ptx_result.get("status") == "passed":
    if not ptx_result.get("embedded_ptx_module_loaded"):
        fail("sm86 PTX regression passed without loading exact embedded PTX")
    expected_probe_sha = ptx_result.get("embedded_ptx_probe_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_probe_sha):
        fail("sm86 PTX regression lacks exact probe evidence checksum")
    probe_path = Path(ptx_result.get("embedded_ptx_probe_path", ""))
    if not probe_path.is_file():
        fail("sm86 PTX regression probe evidence is missing")
    actual_probe_sha = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    if actual_probe_sha != expected_probe_sha:
        fail("sm86 PTX regression probe evidence checksum mismatch")
    try:
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        fail("sm86 PTX regression probe evidence is not valid JSON")
    if (
        probe.get("expected_target") != "sm_86"
        or not probe.get("loaded")
        or probe.get("artifact_sha256") != sm86.get("actual_sha256")
        or not any(attempt.get("loaded") for attempt in probe.get("attempts", []))
    ):
        fail("sm86 PTX regression probe does not bind a loaded module to the artifact")

if report.get("hardware_qualified"):
    if not report.get("provenance", {}).get("official_release_manifest_verified"):
        fail("qualified report lacks official release provenance")
    if set(tests) != set(required_tests):
        fail("qualified report has an incomplete or unexpected test set")
    if any(result.get("status") != "passed" for result in tests.values()):
        fail("qualified report contains a non-passing test")

print("CUDA qualification report relationships: ok")
