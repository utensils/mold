#!/usr/bin/env python3
"""Audit an exported H3 Metal phase-budget snapshot without launching anything.

This offline arithmetic check cannot authenticate an export, verify live host
headroom or prove allocator/watchdog instrumentation. Even a fitting snapshot
always reports launch_ready=false. No subprocess, device or model APIs are used.
"""
import argparse
import json
from pathlib import Path
import re
import sys

GIB = 1 << 30
U64_MAX = (1 << 64) - 1
BASELINE_BYTES = 24 * GIB
HOST_FLOOR_BYTES = 12 * GIB
PHASES = (
    "reference_decode", "reference_preprocess", "reference_visual_encode",
    "reference_audio_encode", "vae_load", "qwen_encode", "qwen_transfer",
    "condition_encode", "noise_allocation", "transformer_load", "denoise",
    "visual_decode", "audio_decode", "waveform_transfer", "mux",
)
IDENTITIES = {
    "source_commit": 40, "candle_commit": 40, "executable_sha256": 64,
    "request_sha256": 64, "plan_sha256": 64, "budget_sha256": 64,
}


def fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label}: expected exactly {sorted(expected)}")
    return value


def byte_count(value, label):
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise ValueError(f"{label}: expected a u64 byte count")
    return value


def loads(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def audit(data):
    fields(data, ("schema", "identities", "phase_bytes", "owner_projection",
                  "snapshot", "native_allocation_ceiling_bytes"), "capture")
    if data["schema"] != "mold.h3-metal-budget-snapshot.v1":
        raise ValueError("unsupported budget snapshot schema")
    identities = fields(data["identities"], IDENTITIES, "identities")
    for key, length in IDENTITIES.items():
        value = identities[key]
        if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
            raise ValueError(f"{key}: expected lowercase hexadecimal identity")
    expected = [f"{phase}_phase_{space}_bytes" for phase in PHASES for space in ("host", "device")]
    budget = fields(data["phase_bytes"], expected, "phase_bytes")
    phases = []
    for phase in PHASES:
        host = byte_count(budget[f"{phase}_phase_host_bytes"], phase + " host")
        device = byte_count(budget[f"{phase}_phase_device_bytes"], phase + " device")
        combined = host + device
        if combined > U64_MAX:
            raise ValueError(f"{phase}: unified phase sum overflow")
        phases.append({"phase": phase, "host_bytes": host, "device_bytes": device,
                       "unified_bytes": combined})
    peak = max(row["unified_bytes"] for row in phases)
    if peak == 0:
        raise ValueError("zero budget is not a prepared H3 request")
    projection = fields(data["owner_projection"], ("device_bytes", "additional_host_bytes"), "owner projection")
    for key, value in projection.items():
        byte_count(value, "owner projection " + key)
    if projection != {"device_bytes": peak, "additional_host_bytes": 0}:
        raise ValueError("owner projection must equal the combined phase peak with zero additional host bytes")
    snapshot = fields(data["snapshot"], ("available_bytes", "device_headroom_bytes"), "snapshot")
    available = byte_count(snapshot["available_bytes"], "available_bytes")
    headroom = byte_count(snapshot["device_headroom_bytes"], "device_headroom_bytes")
    ceiling = byte_count(data["native_allocation_ceiling_bytes"], "native_allocation_ceiling_bytes")
    if ceiling == 0:
        raise ValueError("native allocation ceiling must be positive")
    refusals = []
    if available < BASELINE_BYTES:
        refusals.append("baseline")
    if peak + HOST_FLOOR_BYTES > available:
        refusals.append("host_floor")
    if peak > headroom or ceiling > headroom:
        refusals.append("device_headroom")
    if max(row["device_bytes"] for row in phases) > ceiling:
        refusals.append("native_ceiling")
    if ceiling + max(row["host_bytes"] for row in phases) + HOST_FLOOR_BYTES > available:
        refusals.append("native_ceiling_host_floor")
    return {
        "schema": "mold.h3-metal-budget-audit.v1",
        "decision": "budget_refused" if refusals else "budget_fits_snapshot",
        "launch_ready": False,
        "identities": dict(identities),
        "unified_peak_bytes": peak,
        "binding_phases": [row["phase"] for row in phases if row["unified_bytes"] == peak],
        "phases": phases,
        "snapshot": dict(snapshot),
        "native_allocation_ceiling_bytes": ceiling,
        "baseline_bytes": BASELINE_BYTES,
        "host_floor_bytes": HOST_FLOOR_BYTES,
        "refusals": refusals,
        "remaining_launch_requirements": [
            "authenticate the export against the actual prepared request and executable",
            "verify per-allocation ceiling and external watchdog with failure/cleanup evidence",
            "capture native per-phase peaks on the qualification build",
            "verify configured external Mold library and artifact/output paths on the execution host",
            "obtain the exclusive GPU slot and revalidate live headroom immediately before launch",
        ],
        "qualification": "offline budget arithmetic only; no device or model execution",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="exported phase budget JSON (read only)")
    args = parser.parse_args()
    try:
        report = audit(loads(args.snapshot.read_text()))
    except (OSError, ValueError) as error:
        print(f"H3 Metal preflight refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 1 if report["refusals"] else 0


if __name__ == "__main__":
    sys.exit(main())
