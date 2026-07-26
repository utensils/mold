#!/usr/bin/env python3
"""Relational validator for CUDA qualification evidence.

JSON Schema describes the wire shape; this validator enforces cross-field
relationships that must never be inferred from a caller-set boolean.
"""

import json
import re
import sys


def fail(message: str) -> None:
    print(f"invalid CUDA qualification report: {message}", file=sys.stderr)
    raise SystemExit(1)


if len(sys.argv) != 2:
    fail("usage: validate-cuda-qualification-report.py <report.json>")

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

if report.get("schema_version") != "mold.cuda.sm86.qualification.v3":
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

if tests.get("sm86_ptx_image_smoke", {}).get("status") == "passed" and not tests[
    "sm86_ptx_image_smoke"
].get("cuda_force_ptx_jit"):
    fail("sm86 PTX regression passed without CUDA_FORCE_PTX_JIT=1")

if report.get("hardware_qualified"):
    if not report.get("provenance", {}).get("official_release_manifest_verified"):
        fail("qualified report lacks official release provenance")
    if set(tests) != set(required_tests):
        fail("qualified report has an incomplete or unexpected test set")
    if any(result.get("status") != "passed" for result in tests.values()):
        fail("qualified report contains a non-passing test")

print("CUDA qualification report relationships: ok")
