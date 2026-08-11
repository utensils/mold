#!/usr/bin/env python3
"""Validate an authorization-bound MiniMax H3 GPU conformance campaign.

This runner consumes already captured external evidence. Adapter commands in
the evidence are provenance only and are never executed here. The public Mold
product remains fail-closed and no evidence is copied into the checkout or CI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import stat
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
REQUIRED_ENVIRONMENT = (
    "MOLD_H3_FIXTURE_ROOT",
    "MOLD_H3_AUTHORIZATION_RECORD",
    "MOLD_H3_ORACLE_BUNDLE",
    "MOLD_H3_MOLD_BUNDLE",
    "MOLD_H3_SOURCE_SHA",
)
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MAX_BUNDLE_JSON_BYTES = 2 * 1024 * 1024
MAX_LAYER_JSON_BYTES = 64 * 1024 * 1024
EXACT_AUTHORITY_TIERS = frozenset({"exact-full-bf16", "exact-full-fp32"})
CANONICAL_BF16_DTYPE = "bfloat16"
CANONICAL_F16_DTYPE = "float16"
CANONICAL_F32_DTYPE = "float32"
CANONICAL_INTEGER_DTYPE = "int64"
CANONICAL_METRIC_DTYPE = "float64"
CANONICAL_MIXED_CAPTURE_DTYPE = "official-bf16-fp32-mixed"
VISUAL_VAE_ENVIRONMENT_DTYPE = "official-f32-encode-fp16-decode"
CANONICAL_TENSOR_HASH_ENCODING = "canonical-typed-le-v1"
CANONICAL_E2E_INPUT_SCHEMA = "mold.minimax-h3.e2e-input.v1"
CANONICAL_E2E_INPUT_KIND = "canonical-e2e-json-sha256-v1"
IDENTITY_INPUT_KIND = "identity-sha256-v1"
COMPONENT_AUTHORITY_SET_SCHEMA = "mold.minimax-h3.component-authority-set.v1"
MAX_EXACT_ABSOLUTE_TOLERANCE = 1.0 / 64.0
MAX_EXACT_RELATIVE_TOLERANCE = 1.0 / 64.0
REVIEWED_ABSOLUTE_TOLERANCE_CAPS = {
    ("token-refiner", "sampled-values"): 8.0,
    ("transformer-block", "partial-mm-rope"): 1.0 / 16.0,
    ("transformer-block", "video-head"): 1.25,
    ("transformer-block", "audio-head"): 0.5,
}
AUDIO_VAE_CHECKPOINT_SHA256_BY_ROLE = {
    "oracle": "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb",
    "mold": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
}
SIGNED_INT64_MIN = -(2**63)
SIGNED_INT64_MAX = 2**63 - 1
UNSIGNED_INT64_MAX = 2**64 - 1
E2E_LAYER_TASKS = {
    "end-to-end-t2va": "t2va",
    "end-to-end-fl2va": "fl2va",
    "end-to-end-ref2va": "ref2va",
}
EXPECTED_EXCLUDED_ACCELERATION_ALIASES = {
    "torch-compile": frozenset({"torch-compile", "torch-inductor", "inductor"}),
    "fp8": frozenset({"fp8", "float8", "e4m3", "e4m3fn", "e5m2"}),
    "cache-dit": frozenset({"cache-dit"}),
    "sageattention": frozenset({"sageattention", "sage-attn"}),
    "stochastic-sampling": frozenset(
        {"stochastic-sampling", "stochastic", "ancestral"}
    ),
    "approximate-attention": frozenset(
        {"approximate-attention", "approximate", "approx-attn", "approx"}
    ),
    "tf32": frozenset({"tf32", "tensorfloat32"}),
}
MEASUREMENT_DTYPES = {
    "tokenizer-processor": {
        "token-ids": CANONICAL_INTEGER_DTYPE,
        "special-token-presentation": CANONICAL_INTEGER_DTYPE,
        "processor-shapes": CANONICAL_INTEGER_DTYPE,
        "processor-grids": CANONICAL_INTEGER_DTYPE,
        "sampled-values": CANONICAL_BF16_DTYPE,
    },
    "qwen-layer-50": {
        "shape": CANONICAL_INTEGER_DTYPE,
        "dtype": CANONICAL_INTEGER_DTYPE,
        "statistics": CANONICAL_METRIC_DTYPE,
        "sampled-values": CANONICAL_BF16_DTYPE,
        "tolerance": CANONICAL_METRIC_DTYPE,
    },
    "visual-vae": {
        "pixel-normalization": CANONICAL_F32_DTYPE,
        "posterior-moments": CANONICAL_F32_DTYPE,
        "posterior-seed-42": CANONICAL_F32_DTYPE,
        "posterior-sample": CANONICAL_F32_DTYPE,
        "posterior-fp16-roundtrip": CANONICAL_F16_DTYPE,
        "latent-normalization": CANONICAL_F32_DTYPE,
        "decoded-frames": CANONICAL_F32_DTYPE,
        "tile-seams": CANONICAL_F32_DTYPE,
    },
    "audio-vae": {
        "stereo-packing": CANONICAL_INTEGER_DTYPE,
        "latent-rate": CANONICAL_METRIC_DTYPE,
        "waveform-statistics": CANONICAL_METRIC_DTYPE,
        "phase-polarity": CANONICAL_METRIC_DTYPE,
        "decoded-pcm": CANONICAL_F32_DTYPE,
        "sampled-values": CANONICAL_F32_DTYPE,
    },
    "token-refiner": {
        "shape": CANONICAL_INTEGER_DTYPE,
        "statistics": CANONICAL_METRIC_DTYPE,
        "sampled-values": CANONICAL_BF16_DTYPE,
        "tolerance": CANONICAL_METRIC_DTYPE,
    },
    "transformer-block": {
        "qk-rms": CANONICAL_METRIC_DTYPE,
        "partial-mm-rope": CANONICAL_BF16_DTYPE,
        "adaln": CANONICAL_BF16_DTYPE,
        "video-head": CANONICAL_F32_DTYPE,
        "audio-head": CANONICAL_F32_DTYPE,
        "tolerance": CANONICAL_METRIC_DTYPE,
    },
    "packed-layout": {
        "row-order": CANONICAL_INTEGER_DTYPE,
        "modality-tags": CANONICAL_INTEGER_DTYPE,
        "rotary-coordinates": CANONICAL_INTEGER_DTYPE,
        "timestep-indices": CANONICAL_INTEGER_DTYPE,
    },
    "noise-allocation": {
        "draw-order": CANONICAL_INTEGER_DTYPE,
        "tensor-shapes": CANONICAL_INTEGER_DTYPE,
        "sampled-values": CANONICAL_BF16_DTYPE,
        "hash": CANONICAL_INTEGER_DTYPE,
    },
    "end-to-end-t2va": {
        "latent-statistics": CANONICAL_METRIC_DTYPE,
        "av-timing": CANONICAL_METRIC_DTYPE,
        "video-metrics": CANONICAL_METRIC_DTYPE,
        "audio-metrics": CANONICAL_METRIC_DTYPE,
    },
    "end-to-end-fl2va": {
        "condition-latents": CANONICAL_BF16_DTYPE,
        "latent-statistics": CANONICAL_METRIC_DTYPE,
        "av-timing": CANONICAL_METRIC_DTYPE,
        "video-metrics": CANONICAL_METRIC_DTYPE,
        "audio-metrics": CANONICAL_METRIC_DTYPE,
    },
    "end-to-end-ref2va": {
        "packed-order": CANONICAL_INTEGER_DTYPE,
        "latent-statistics": CANONICAL_METRIC_DTYPE,
        "av-timing": CANONICAL_METRIC_DTYPE,
        "video-metrics": CANONICAL_METRIC_DTYPE,
        "audio-metrics": CANONICAL_METRIC_DTYPE,
    },
}
PROVENANCE_HASH_KEYS = {
    "adapter-implementation-sha256",
    "checkpoint-sha256",
    "oracle-runtime-sha256",
    "runtime-sha256",
    "tokenizer-hash",
    "processor-hash",
    "artifact-set-sha256",
}
REQUIRED_STRUCTURED_HASH_PROVENANCE = frozenset(
    {
        "adapter-implementation-sha256",
        "checkpoint-sha256",
        "runtime-sha256",
    }
)
RECORD_ONLY_FLOAT_MEASUREMENTS = {
    "visual-vae": frozenset(
        {
            "pixel-normalization",
            "posterior-moments",
            "posterior-seed-42",
            "posterior-sample",
            "posterior-fp16-roundtrip",
            "latent-normalization",
            "decoded-frames",
            "tile-seams",
        }
    ),
    "audio-vae": frozenset(
        {
            "waveform-statistics",
            "phase-polarity",
            "decoded-pcm",
            "sampled-values",
        }
    ),
    "token-refiner": frozenset({"statistics", "sampled-values"}),
    "transformer-block": frozenset(
        {"qk-rms", "partial-mm-rope", "adaln", "video-head", "audio-head"}
    ),
}
COMPLETE_SAMPLE_MEASUREMENTS = {
    "qwen-layer-50": frozenset({"sampled-values"}),
    "visual-vae": frozenset({"posterior-seed-42"}),
    "token-refiner": RECORD_ONLY_FLOAT_MEASUREMENTS["token-refiner"],
    "transformer-block": RECORD_ONLY_FLOAT_MEASUREMENTS["transformer-block"],
}
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
REVIEWED_AUTHORIZATION_EVIDENCE_SHA256 = (
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d"
)


class GpuConformanceFailure(Exception):
    """A fail-closed private GPU campaign contract violation."""


def fail(message: str) -> None:
    raise GpuConformanceFailure(message)


def load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("minimax_h3_conformance", TOOL_PATH)
    if spec is None or spec.loader is None:
        fail("cannot load the pinned MiniMax H3 conformance tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_environment(environment: Mapping[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in REQUIRED_ENVIRONMENT:
        value = environment.get(name, "").strip()
        if not value:
            fail(f"{name} is required")
        values[name] = value
    if COMMIT_SHA_PATTERN.fullmatch(values["MOLD_H3_SOURCE_SHA"]) is None:
        fail("MOLD_H3_SOURCE_SHA must be one lowercase 40-character commit SHA")
    return values


def resolve_external_file(label: str, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        fail(f"{label} is unavailable")
    if not resolved.is_file():
        fail(f"{label} is not a file")
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved
    fail(f"{label} must live outside the Mold repository")


def resolve_bundle_path(
    fixture_root: pathlib.Path, label: str, value: str
) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        fail(f"{label} is unavailable")
    if not resolved.is_file():
        fail(f"{label} is not a file")
    try:
        resolved.relative_to(fixture_root)
    except ValueError:
        fail(f"{label} must be inside MOLD_H3_FIXTURE_ROOT")
    return resolved


def probe_cuda() -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        fail("CUDA probe failed")
    if result.returncode != 0 or not result.stdout.strip():
        fail("CUDA probe failed")


def probe_source_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        fail("checkout source SHA probe failed")
    source_sha = result.stdout.strip()
    if result.returncode != 0 or COMMIT_SHA_PATTERN.fullmatch(source_sha) is None:
        fail("checkout source SHA probe failed")
    return source_sha


def is_cuda_environment(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith("cuda:")


def normalized_acceleration(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def names_excluded_acceleration(
    value: str, excluded_accelerations: frozenset[str]
) -> bool:
    normalized = normalized_acceleration(value)
    return any(
        normalized_acceleration(alias) in normalized
        for identifier in excluded_accelerations
        for alias in EXPECTED_EXCLUDED_ACCELERATION_ALIASES.get(
            identifier, frozenset({identifier})
        )
        if normalized_acceleration(alias)
    )


def reject_excluded_attention_backend(
    value: Any, excluded_accelerations: frozenset[str], context: str
) -> None:
    if not isinstance(value, str):
        fail(f"{context} attention backend is invalid")
    if names_excluded_acceleration(value, excluded_accelerations):
        fail(f"{context} uses a manifest-excluded acceleration")


def validate_acceleration_policy(
    environment: dict[str, Any],
    excluded_accelerations: frozenset[str],
    context: str,
) -> None:
    policy = environment.get("acceleration_policy")
    if not isinstance(policy, dict):
        fail(f"{context} lacks structured acceleration evidence")
    enabled = policy.get("enabled")
    disabled = policy.get("disabled")
    if not isinstance(enabled, list) or not isinstance(disabled, list):
        fail(f"{context} structured acceleration evidence is invalid")
    if (
        any(
            not isinstance(value, str) or not value or value != value.strip().lower()
            for value in [*enabled, *disabled]
        )
        or len(enabled) != len(set(enabled))
        or len(disabled) != len(set(disabled))
    ):
        fail(f"{context} structured acceleration evidence is not canonical")
    if frozenset(disabled) != excluded_accelerations:
        fail(f"{context} does not disable every manifest-excluded acceleration")
    if any(
        names_excluded_acceleration(value, excluded_accelerations) for value in enabled
    ):
        fail(f"{context} enables a manifest-excluded acceleration")
    reject_excluded_attention_backend(
        environment.get("attention_backend"), excluded_accelerations, context
    )


def call_redacted(tool: Any, context: str, action: Callable[[], Any]) -> Any:
    try:
        return action()
    except (tool.ConformanceFailure, OSError, OverflowError, ValueError):
        fail(f"{context} failed; inspect evidence only on the protected runner")


def read_json_once(
    path: pathlib.Path, label: str, max_bytes: int
) -> tuple[bytes, Any]:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > max_bytes
        ):
            fail(f"{label} exceeds its bounded regular-file contract")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            encoded = handle.read(metadata.st_size + 1)
        if len(encoded) != metadata.st_size:
            fail(f"{label} changed during its bounded read")
        value = json.loads(encoded)
    except (OSError, ValueError):
        fail(f"{label} is unavailable or invalid")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return encoded, value


def manifest_layer_tiers(manifest: dict[str, Any]) -> dict[str, str]:
    tiers = {
        layer["id"]: layer["authority_tier"] for layer in manifest["fixture_layers"]
    }
    if len(tiers) != len(manifest["fixture_layers"]):
        fail("checked manifest contains duplicate layer identities")
    return tiers


def manifest_component_authorities(manifest: dict[str, Any]) -> dict[str, str]:
    authorities = {
        component["id"]: component["sha256"]
        for component in manifest["component_indexes"]
    }
    if len(authorities) != len(manifest["component_indexes"]):
        fail("checked manifest contains duplicate component authority identities")
    return authorities


def manifest_excluded_accelerations(manifest: dict[str, Any]) -> frozenset[str]:
    values = manifest["numerical_authority"]["excluded_accelerations"]
    excluded = frozenset(value.lower() for value in values)
    raw_aliases = manifest["numerical_authority"]["excluded_acceleration_aliases"]
    aliases = {
        identifier: frozenset(values) for identifier, values in raw_aliases.items()
    }
    if (
        len(excluded) != len(values)
        or set(raw_aliases) != excluded
        or any(len(values) != len(set(values)) for values in raw_aliases.values())
        or aliases != EXPECTED_EXCLUDED_ACCELERATION_ALIASES
    ):
        fail("checked manifest contains invalid acceleration exclusions")
    return excluded


def canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        fail("canonical JSON evidence is invalid")
    return hashlib.sha256(encoded).hexdigest()


def component_authority_set_sha256(
    component_ids: list[str], component_authorities: dict[str, str]
) -> str:
    return canonical_json_sha256(
        {
            "schema_version": COMPONENT_AUTHORITY_SET_SCHEMA,
            "components": [
                {"id": identifier, "sha256": component_authorities[identifier]}
                for identifier in sorted(component_ids)
            ],
        }
    )


def exact_layer_contracts(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    component_authorities = manifest_component_authorities(manifest)
    excluded_accelerations = manifest_excluded_accelerations(manifest)
    source_revisions = {
        source["id"]: source["revision"] for source in manifest.get("sources", [])
    }
    oracle_runtime = manifest.get("numerical_authority", {}).get("oracle_runtime")
    if (
        not isinstance(oracle_runtime, dict)
        or set(oracle_runtime)
        != {"python", "torch", "numpy", "cuda", "transformers_revision"}
        or any(
            not isinstance(value, str) or not value for value in oracle_runtime.values()
        )
        or oracle_runtime["transformers_revision"]
        != source_revisions.get("transformers")
    ):
        fail("checked manifest contains invalid oracle runtime authority")
    contracts: dict[str, dict[str, Any]] = {}
    for layer in manifest["fixture_layers"]:
        if layer["authority_tier"] not in EXACT_AUTHORITY_TIERS:
            continue
        identifier = layer["id"]
        if identifier in contracts:
            fail("checked manifest contains duplicate exact layer identities")
        oracle_adapter = layer.get("oracle_adapter")
        if oracle_adapter is not None:
            if (
                not isinstance(oracle_adapter, dict)
                or set(oracle_adapter)
                != {
                    "id",
                    "command_prefix",
                    "implementation_path",
                    "implementation_sha256",
                }
                or not isinstance(oracle_adapter["id"], str)
                or not oracle_adapter["id"]
                or not isinstance(oracle_adapter["command_prefix"], str)
                or not oracle_adapter["command_prefix"]
                or re.fullmatch(
                    r"scripts/[a-zA-Z0-9._/-]+\.py",
                    oracle_adapter["implementation_path"],
                )
                is None
                or ".."
                in pathlib.PurePosixPath(oracle_adapter["implementation_path"]).parts
                or not oracle_adapter["command_prefix"].startswith(
                    f"python3 {oracle_adapter['implementation_path']} "
                )
                or re.fullmatch(
                    r"[0-9a-f]{64}", oracle_adapter["implementation_sha256"]
                )
                is None
            ):
                fail(
                    f"checked manifest layer {identifier} has invalid adapter authority"
                )
        required_components = layer["required_component_indexes"]
        if len(required_components) != len(set(required_components)) or not set(
            required_components
        ).issubset(component_authorities):
            fail(f"checked manifest layer {identifier} has invalid component authority")
        role_invariants = layer["role_invariant_provenance"]
        if len(role_invariants) != len(set(role_invariants)) or not set(
            role_invariants
        ).issubset(layer["required_provenance"]):
            fail(f"checked manifest layer {identifier} has invalid provenance parity")
        pinned_provenance = layer["pinned_provenance"]
        pinned_keys = [record["key"] for record in pinned_provenance]
        if (
            len(pinned_keys) != len(set(pinned_keys))
            or not set(pinned_keys).issubset(layer["required_provenance"])
            or not set(pinned_keys).issubset(role_invariants)
            or any(
                len(record["component_indexes"])
                != len(set(record["component_indexes"]))
                or not set(record["component_indexes"]).issubset(required_components)
                for record in pinned_provenance
            )
        ):
            fail(f"checked manifest layer {identifier} has invalid pinned provenance")
        expected_task = E2E_LAYER_TASKS.get(identifier)
        input_contract = layer["input_contract"]
        if expected_task is None:
            if input_contract != {"kind": IDENTITY_INPUT_KIND}:
                fail(f"checked manifest layer {identifier} has invalid input authority")
        elif input_contract != {
            "kind": CANONICAL_E2E_INPUT_KIND,
            "task": expected_task,
            "algorithm": "minimax-h3-flow-euler-v1",
            "guidance": "0",
            "video_shift": "12",
            "audio_shift": "3",
            "dimension_multiple": 32,
            "frame_step": 17,
            "frame_offset": 5,
            "min_frames": 124,
            "max_frames": 362,
            "max_pixels": 1_069_056,
            "min_aspect_ratio": "0.25",
            "max_aspect_ratio": "4",
            "fps": 24,
            "video_scheduler_component": "official-video-scheduler-config",
            "audio_scheduler_component": "official-audio-scheduler-config",
        } or not {
            input_contract["video_scheduler_component"],
            input_contract["audio_scheduler_component"],
        }.issubset(required_components):
            fail(f"checked manifest layer {identifier} has invalid input authority")
        contract = dict(layer)
        contract["_required_component_authorities"] = [
            {"id": component_id, "sha256": component_authorities[component_id]}
            for component_id in required_components
        ]
        contract["_pinned_provenance_values"] = {
            record["key"]: component_authority_set_sha256(
                record["component_indexes"], component_authorities
            )
            for record in pinned_provenance
        }
        if identifier == "tokenizer-processor":
            contract["_pinned_provenance_values"].update(
                {
                    "transformers-revision": source_revisions["transformers"],
                    "oracle-runtime-sha256": canonical_json_sha256(oracle_runtime),
                }
            )
        if layer["authority_tier"] == "exact-full-bf16":
            contract["_oracle_runtime"] = dict(oracle_runtime)
        contract["_excluded_accelerations"] = excluded_accelerations
        contracts[identifier] = contract

    if set(contracts) != set(MEASUREMENT_DTYPES):
        fail("protected measurement policy does not cover the exact manifest layers")
    for identifier, contract in contracts.items():
        required = contract["required_measurements"]
        if len(required) != len(set(required)):
            fail(f"checked manifest layer {identifier} repeats required measurements")
        if set(required) != set(MEASUREMENT_DTYPES[identifier]):
            fail(
                f"protected measurement policy drifts from manifest layer {identifier}"
            )
        provenance = contract["required_provenance"]
        if len(provenance) != len(set(provenance)):
            fail(f"checked manifest layer {identifier} repeats required provenance")
    return contracts


def load_bundle_once(
    tool: Any,
    path: pathlib.Path,
    authorization_sha: str,
    label: str,
) -> dict[str, Any]:
    _, bundle = read_json_once(path, label, MAX_BUNDLE_JSON_BYTES)
    call_redacted(
        tool,
        f"{label} schema validation",
        lambda: tool.validate_schema(bundle, tool.BUNDLE_SCHEMA_PATH),
    )
    if bundle["schema_version"] != tool.BUNDLE_SCHEMA_VERSION:
        fail(f"{label} schema version is unsupported")
    if bundle["manifest_sha256"] != tool.sha256_file(tool.MANIFEST_PATH):
        fail(f"{label} targets a different conformance manifest")
    if bundle["authorization_document_sha256"] != authorization_sha:
        fail(f"{label} is not bound to the reviewed authorization evidence")
    if not bundle["fixtures"]:
        fail(f"{label} contains no evidence")
    return bundle


def expected_oracle_runtime(
    layer_contracts: Mapping[str, dict[str, Any]],
) -> dict[str, str]:
    if not layer_contracts:
        fail("protected layer contracts lack oracle runtime authority")
    runtimes = {
        json.dumps(
            contract.get("_oracle_runtime"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for contract in layer_contracts.values()
    }
    if len(runtimes) != 1:
        fail("protected layer contracts disagree on oracle runtime authority")
    runtime = next(iter(layer_contracts.values())).get("_oracle_runtime")
    if not isinstance(runtime, dict):
        fail("protected layer contracts lack oracle runtime authority")
    return dict(runtime)


def validate_bundle_oracle_adapters(
    bundle: dict[str, Any],
    layer_contracts: Mapping[str, dict[str, Any]],
    device: str,
) -> dict[str, dict[str, str]]:
    fixture_layers = {fixture["layer"] for fixture in bundle["fixtures"]}
    expected = {
        layer: contract["oracle_adapter"]
        for layer, contract in layer_contracts.items()
        if layer in fixture_layers and contract.get("oracle_adapter") is not None
    }
    records = bundle["capture_environment"].get("oracle_adapters")
    if not expected:
        if records is not None:
            fail("oracle bundle declares adapter records for unbound fixture layers")
        return {}
    if not isinstance(records, list) or not records:
        fail("oracle bundle is missing manifest-required adapter records")
    observed: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "layer",
            "id",
            "command",
            "implementation_sha256",
        }:
            fail("oracle bundle contains an invalid adapter record")
        layer = record.get("layer")
        if not isinstance(layer, str) or not layer:
            fail("oracle bundle contains an invalid adapter layer")
        if layer in observed:
            fail(f"oracle bundle repeats adapter record for layer {layer}")
        order.append(layer)
        observed[layer] = record
    if order != sorted(order):
        fail("oracle bundle adapter records are not in canonical layer order")
    if set(observed) != set(expected):
        fail("oracle bundle adapter layers differ from its manifest-bound fixtures")
    for layer, authority in expected.items():
        expected_record = {
            "layer": layer,
            "id": authority["id"],
            "command": f"{authority['command_prefix']} --device {device}",
            "implementation_sha256": authority["implementation_sha256"],
        }
        if observed[layer] != expected_record:
            fail(f"oracle bundle adapter record for layer {layer} is not exact")
    return observed


def validate_capture_environment(
    tool: Any,
    bundle: dict[str, Any],
    role: str,
    source_sha: str,
    excluded_accelerations: frozenset[str],
    layer_contracts: Mapping[str, dict[str, Any]] | None = None,
) -> None:
    if layer_contracts is None:
        layer_contracts = exact_layer_contracts(tool.validate_manifest())
    capture = bundle["capture_environment"]
    if not is_cuda_environment(capture["device"]):
        fail(f"{role} bundle does not declare a CUDA capture environment")
    fixture_profiles = set()
    for fixture in bundle["fixtures"]:
        contract = layer_contracts.get(fixture["layer"])
        if contract is None:
            fail(
                f"{role} bundle layer {fixture['layer']} lacks an exact "
                "manifest authority tier"
            )
        fixture_profiles.add(
            layer_environment_dtype(fixture["layer"], contract["authority_tier"])
        )
    expected_capture_dtype = (
        next(iter(fixture_profiles))
        if len(fixture_profiles) == 1
        else CANONICAL_MIXED_CAPTURE_DTYPE
    )
    if capture["dtype"] != expected_capture_dtype:
        fail(
            f"{role} bundle capture environment dtype must be "
            f"{expected_capture_dtype}"
        )
    validate_acceleration_policy(capture, excluded_accelerations, f"{role} bundle")
    if role == "oracle":
        expected_framework = "diffusers"
        expected_revision = tool.EXPECTED_REVISIONS["diffusers"]
        fixture_layers = {fixture["layer"] for fixture in bundle["fixtures"]}
        runtime_contracts = {
            layer: contract
            for layer, contract in layer_contracts.items()
            if layer in fixture_layers and "_oracle_runtime" in contract
        }
        if runtime_contracts:
            if capture.get("oracle_runtime") != expected_oracle_runtime(
                runtime_contracts
            ):
                fail("oracle bundle runtime identity is not exact")
        elif "oracle_runtime" in capture:
            fail("oracle bundle claims an unreviewed runtime identity")
        validate_bundle_oracle_adapters(bundle, layer_contracts, capture["device"])
    else:
        expected_framework = "mold"
        expected_revision = source_sha
        if "oracle_runtime" in capture or "oracle_adapters" in capture:
            fail("Mold bundle must not claim oracle runtime or adapter identity")
    if capture["framework"] != expected_framework:
        fail(f"{role} bundle framework is not {expected_framework}")
    if capture["framework_revision"] != expected_revision:
        fail(f"{role.capitalize()} bundle framework revision is not exact")


def expected_tensor_summary(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": output["shape"],
        "dtype": output["dtype"],
        "min": output["statistics"]["min"],
        "max": output["statistics"]["max"],
        "mean": output["statistics"]["mean"],
        "std": output["statistics"]["std"],
        "sampled_values": [sample["value"] for sample in output["samples"]],
    }


def expected_tolerance_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "absolute": policy["absolute"],
        "relative": policy["relative"],
        "metric": policy["metric"],
    }


def keyed_evidence_records(
    records: Any,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or not records:
        fail(f"{label} must be a non-empty record list")
    keyed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("key"), str):
            fail(f"{label} contains an invalid record")
        key = record["key"]
        if key in keyed:
            fail(f"{label} contains duplicate key {key!r}")
        keyed[key] = record
    return keyed


def keyed_component_records(records: Any, label: str) -> dict[str, str]:
    if not isinstance(records, list) or not records:
        fail(f"{label} must be a non-empty component authority list")
    keyed: dict[str, str] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not isinstance(record.get("sha256"), str)
        ):
            fail(f"{label} contains an invalid component authority")
        identifier = record["id"]
        if identifier in keyed:
            fail(f"{label} contains duplicate component id {identifier!r}")
        keyed[identifier] = record["sha256"]
    return keyed


def structured_provenance_value(document: dict[str, Any], key: str) -> str | None:
    if key == "source-revision":
        return document["producer"]["revision"]
    if key == "transformers-revision":
        runtime = document["environment"].get("oracle_runtime")
        if isinstance(runtime, dict):
            return runtime.get("transformers_revision")
        return None
    if key == "adapter-implementation-sha256":
        return document["adapter"].get("implementation_sha256")
    if key == "checkpoint-sha256":
        return document["input"].get("checkpoint_sha256")
    if key == "oracle-runtime-sha256":
        runtime = document["environment"].get("oracle_runtime")
        if isinstance(runtime, dict):
            return canonical_json_sha256(runtime)
        return None
    if key == "component-index-hash":
        components = document["input"]["component_indexes"]
        if len(components) != 1:
            fail("singular component provenance requires one component authority")
        return components[0]["sha256"]
    if key == "component-index-hashes":
        components = sorted(
            document["input"]["component_indexes"],
            key=lambda component: component["id"],
        )
        return ",".join(
            f"{component['id']}={component['sha256']}" for component in components
        )
    if key in {"device", "generator-device"}:
        return document["environment"]["device"]
    if key == "dtype":
        return document["environment"]["dtype"]
    if key == "attention-backend":
        return document["environment"]["attention_backend"]
    if key == "capture-command":
        return document["adapter"]["command"]
    if key == "runtime-sha256":
        runtime = document["environment"].get("runtime")
        if (
            not isinstance(runtime, dict)
            or not runtime
            or any(
                not isinstance(runtime_key, str)
                or not runtime_key
                or runtime_key != runtime_key.strip()
                or len(runtime_key) > 256
                or CONTROL_CHARACTER_PATTERN.search(runtime_key) is not None
                or not isinstance(runtime_value, str)
                or not runtime_value
                or runtime_value != runtime_value.strip()
                or len(runtime_value) > 4096
                or CONTROL_CHARACTER_PATTERN.search(runtime_value) is not None
                for runtime_key, runtime_value in runtime.items()
            )
        ):
            return None
        return canonical_json_sha256(runtime)
    return None


def layer_environment_dtype(layer: str, authority_tier: str) -> str:
    if layer == "visual-vae":
        return VISUAL_VAE_ENVIRONMENT_DTYPE
    if authority_tier == "exact-full-bf16":
        return CANONICAL_BF16_DTYPE
    if authority_tier == "exact-full-fp32":
        return CANONICAL_F32_DTYPE
    fail(f"unsupported exact authority tier {authority_tier!r}")


def is_json_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_signed_int64(value: Any) -> bool:
    return is_json_integer(value) and SIGNED_INT64_MIN <= value <= SIGNED_INT64_MAX


def is_unsigned_int64(value: Any) -> bool:
    return is_json_integer(value) and 0 <= value <= UNSIGNED_INT64_MAX


def is_finite_float64(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(converted) and (
        not is_json_integer(value) or int(converted) == value
    )


def is_exact_bfloat16(value: Any) -> bool:
    if not is_finite_float64(value):
        return False
    converted = float(value)
    try:
        float32_bits = struct.unpack(">I", struct.pack(">f", converted))[0]
    except (OverflowError, struct.error):
        return False
    rounding_bias = 0x7FFF + ((float32_bits >> 16) & 1)
    bfloat16_bits = (float32_bits + rounding_bias) >> 16
    if bfloat16_bits & 0x7F80 == 0x7F80:
        return False
    reconstructed = struct.unpack(">f", struct.pack(">I", bfloat16_bits << 16))[0]
    return reconstructed == converted


def is_exact_ieee_value(value: Any, encoding: str) -> bool:
    if not is_finite_float64(value):
        return False
    converted = float(value)
    try:
        reconstructed = struct.unpack(encoding, struct.pack(encoding, converted))[0]
    except (OverflowError, struct.error):
        return False
    return math.isfinite(reconstructed) and reconstructed == converted


def validate_output_numeric_domain(
    output: dict[str, Any], expected_dtype: str, context: str
) -> None:
    statistics = output["statistics"]
    samples = output["samples"]
    if not all(is_finite_float64(statistics[key]) for key in ("mean", "std")):
        fail(f"{context} summary is outside the finite float64 domain")
    if expected_dtype == CANONICAL_INTEGER_DTYPE:
        if (
            not is_signed_int64(statistics["min"])
            or not is_signed_int64(statistics["max"])
            or any(not is_signed_int64(sample["value"]) for sample in samples)
        ):
            fail(f"{context} integer min/max and samples must be signed int64")
        return
    if expected_dtype == CANONICAL_BF16_DTYPE:
        if (
            not is_exact_bfloat16(statistics["min"])
            or not is_exact_bfloat16(statistics["max"])
            or any(not is_exact_bfloat16(sample["value"]) for sample in samples)
        ):
            fail(f"{context} activation min/max and samples must be exact bfloat16")
        return
    if expected_dtype in {CANONICAL_F16_DTYPE, CANONICAL_F32_DTYPE}:
        encoding = ">e" if expected_dtype == CANONICAL_F16_DTYPE else ">f"
        if (
            not is_exact_ieee_value(statistics["min"], encoding)
            or not is_exact_ieee_value(statistics["max"], encoding)
            or any(
                not is_exact_ieee_value(sample["value"], encoding) for sample in samples
            )
        ):
            fail(f"{context} min/max and samples must be exact {expected_dtype}")
        return
    if expected_dtype == CANONICAL_METRIC_DTYPE and (
        not is_finite_float64(statistics["min"])
        or not is_finite_float64(statistics["max"])
        or any(not is_finite_float64(sample["value"]) for sample in samples)
    ):
        fail(f"{context} metric min/max and samples must be finite float64")


def validate_complete_sample_coverage(output: dict[str, Any], context: str) -> None:
    shape = output["shape"]
    total = math.prod(shape)
    samples = output["samples"]
    if len(samples) != total:
        fail(f"{context} must retain every tensor coordinate")
    for ordinal, sample in enumerate(samples):
        index = [0] * len(shape)
        remaining = ordinal
        for axis in range(len(shape) - 1, -1, -1):
            index[axis], remaining = remaining % shape[axis], remaining // shape[axis]
        if sample["index"] != index:
            fail(f"{context} sample coordinates are not complete and ordered")


def validate_e2e_input_contract(
    document: dict[str, Any], manifest_layer: dict[str, Any], role: str
) -> None:
    layer = manifest_layer["id"]
    input_contract = manifest_layer["input_contract"]
    conformance = document["input"].get("conformance")
    if input_contract["kind"] == IDENTITY_INPUT_KIND:
        if conformance is not None:
            fail(f"{role} layer {layer} has unexpected end-to-end input evidence")
        return
    if input_contract["kind"] != CANONICAL_E2E_INPUT_KIND:
        fail(f"protected manifest layer {layer} has an unsupported input contract")
    if not isinstance(conformance, dict):
        fail(f"{role} layer {layer} lacks canonical end-to-end input evidence")
    if (
        conformance.get("schema_version") != CANONICAL_E2E_INPUT_SCHEMA
        or conformance.get("task") != input_contract["task"]
    ):
        fail(f"{role} layer {layer} end-to-end input task is invalid")
    if not is_unsigned_int64(conformance.get("seed")):
        fail(f"{role} layer {layer} end-to-end input seed must be unsigned int64")
    for key in ("width", "height", "frames", "fps"):
        if not is_signed_int64(conformance.get(key)) or conformance[key] <= 0:
            fail(
                f"{role} layer {layer} end-to-end input {key} "
                "must be a positive signed int64"
            )
    width = conformance["width"]
    height = conformance["height"]
    frames = conformance["frames"]
    aspect_ratio = Fraction(width, height)
    if (
        width % input_contract["dimension_multiple"] != 0
        or height % input_contract["dimension_multiple"] != 0
        or width * height > input_contract["max_pixels"]
        or aspect_ratio < Fraction(input_contract["min_aspect_ratio"])
        or aspect_ratio > Fraction(input_contract["max_aspect_ratio"])
    ):
        fail(f"{role} layer {layer} end-to-end dimensions violate the H3 contract")
    if (
        frames < input_contract["min_frames"]
        or frames > input_contract["max_frames"]
        or (frames - input_contract["frame_offset"]) % input_contract["frame_step"] != 0
        or conformance["fps"] != input_contract["fps"]
    ):
        fail(f"{role} layer {layer} end-to-end frame contract is invalid")
    video_sampler = conformance["video_sampler"]
    audio_sampler = conformance["audio_sampler"]
    if (
        not is_signed_int64(video_sampler.get("steps"))
        or video_sampler["steps"] < 2
        or audio_sampler.get("steps") != video_sampler["steps"]
        or video_sampler.get("algorithm") != input_contract["algorithm"]
        or audio_sampler.get("algorithm") != input_contract["algorithm"]
        or video_sampler.get("guidance") != input_contract["guidance"]
        or audio_sampler.get("guidance") != input_contract["guidance"]
        or video_sampler.get("shift") != input_contract["video_shift"]
        or audio_sampler.get("shift") != input_contract["audio_shift"]
    ):
        fail(f"{role} layer {layer} end-to-end sampler contract is invalid")
    if conformance["negative_prompt_sha256"] != hashlib.sha256(b"").hexdigest():
        fail(f"{role} layer {layer} end-to-end negative prompt must be absent")
    conditioning_roles = [item["role"] for item in conformance["conditioning"]]
    task = conformance["task"]
    if task == "t2va" and conditioning_roles:
        fail(f"{role} layer {layer} T2VA input must not carry conditioning media")
    if task == "fl2va" and conditioning_roles not in (
        ["first-frame"],
        ["last-frame"],
        ["first-frame", "last-frame"],
    ):
        fail(f"{role} layer {layer} FL2VA conditioning order is invalid")
    if task == "ref2va" and (
        not conditioning_roles
        or any(not value.startswith("reference-") for value in conditioning_roles)
    ):
        fail(f"{role} layer {layer} Ref2VA conditioning order is invalid")
    if document["input"]["sha256"] != canonical_json_sha256(conformance):
        fail(f"{role} layer {layer} input hash does not match canonical evidence")


def validate_manifest_layer_evidence(
    document: dict[str, Any],
    manifest_layer: dict[str, Any],
    role: str,
    excluded_accelerations: frozenset[str] | None = None,
) -> None:
    layer = manifest_layer["id"]
    if excluded_accelerations is None:
        protected_exclusions = manifest_layer.get("_excluded_accelerations")
        if not isinstance(protected_exclusions, frozenset):
            fail(f"protected manifest layer {layer} lacks acceleration authority")
        excluded_accelerations = protected_exclusions
    if document.get("layer") != layer:
        fail(f"{role} evidence does not match manifest layer {layer}")
    if document["adapter"]["tensor_hash_encoding"] != CANONICAL_TENSOR_HASH_ENCODING:
        fail(f"{role} layer {layer} tensor hash encoding is not canonical")
    expected_runtime = manifest_layer.get("_oracle_runtime")
    if isinstance(expected_runtime, dict):
        if role == "oracle":
            if document["environment"].get("oracle_runtime") != expected_runtime:
                fail(f"oracle layer {layer} runtime identity is not exact")
        elif "oracle_runtime" in document["environment"]:
            fail(f"Mold layer {layer} must not claim oracle runtime identity")
    elif "oracle_runtime" in document["environment"]:
        fail(f"{role} layer {layer} claims an unreviewed oracle runtime identity")
    expected_execution_dtype = layer_environment_dtype(
        layer, manifest_layer["authority_tier"]
    )
    if document["environment"]["dtype"] != expected_execution_dtype:
        fail(
            f"{role} layer {layer} environment dtype must be "
            f"{expected_execution_dtype}"
        )

    adapter_authority = manifest_layer.get("oracle_adapter")
    if role == "oracle" and adapter_authority is not None:
        expected_command = (
            f"{adapter_authority['command_prefix']} "
            f"--device {document['environment']['device']}"
        )
        if (
            document["adapter"].get("id") != adapter_authority["id"]
            or document["adapter"].get("command") != expected_command
            or document["adapter"].get("implementation_sha256")
            != adapter_authority["implementation_sha256"]
        ):
            fail(f"oracle layer {layer} adapter identity is not exact")
    required_components = manifest_layer.get("_required_component_authorities")
    if not isinstance(required_components, list) or not required_components:
        fail(f"protected manifest layer {layer} lacks component authority")
    actual_components = document.get("input", {}).get("component_indexes")
    expected_component_map = {
        component["id"]: component["sha256"] for component in required_components
    }
    actual_component_map = keyed_component_records(
        actual_components, f"{role} layer {layer} component authorities"
    )
    if actual_component_map != expected_component_map:
        fail(f"{role} layer {layer} component authorities differ from the manifest")
    expected_summary = (
        required_components[0]["sha256"]
        if len(required_components) == 1
        else component_authority_set_sha256(
            [component["id"] for component in required_components],
            expected_component_map,
        )
    )
    if document["input"]["component_index_sha256"] != expected_summary:
        fail(
            f"{role} layer {layer} component authority summary differs from the manifest"
        )
    if layer == "audio-vae" and document["input"].get("checkpoint_sha256") != (
        AUDIO_VAE_CHECKPOINT_SHA256_BY_ROLE[role]
    ):
        fail(f"{role} layer audio-vae checkpoint authority is not exact")

    validate_acceleration_policy(
        document["environment"],
        excluded_accelerations,
        f"{role} layer {layer}",
    )
    validate_e2e_input_contract(document, manifest_layer, role)

    provenance = keyed_evidence_records(
        document.get("provenance"), f"{role} layer {layer} provenance"
    )
    required_provenance = set(manifest_layer["required_provenance"])
    if set(provenance) != required_provenance:
        fail(f"{role} layer {layer} provenance keys differ from the manifest")
    for key, record in provenance.items():
        value = record.get("value")
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > 4096
            or CONTROL_CHARACTER_PATTERN.search(value) is not None
        ):
            fail(f"{role} layer {layer} provenance {key!r} is not canonical")
        if key in {"source-revision", "transformers-revision"} and (
            COMMIT_SHA_PATTERN.fullmatch(value) is None
        ):
            fail(f"{role} layer {layer} provenance {key!r} is not canonical")
        if key in PROVENANCE_HASH_KEYS and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            fail(f"{role} layer {layer} provenance {key!r} is not canonical")
        if key == "seed" and re.fullmatch(r"0|[1-9][0-9]*", value) is None:
            fail(f"{role} layer {layer} provenance {key!r} is not canonical")
        pinned_value = manifest_layer["_pinned_provenance_values"].get(key)
        if pinned_value is not None and value != pinned_value:
            fail(f"{role} layer {layer} provenance {key!r} is not authority-pinned")
        structured = structured_provenance_value(document, key)
        if key in REQUIRED_STRUCTURED_HASH_PROVENANCE and structured is None:
            fail(
                f"{role} layer {layer} provenance {key!r} "
                "lacks canonical structured evidence"
            )
        if structured is not None and value != structured:
            fail(
                f"{role} layer {layer} provenance {key!r} "
                "does not match structured evidence"
            )

    outputs = keyed_evidence_records(
        document.get("outputs"), f"{role} layer {layer} outputs"
    )
    measurement_order = manifest_layer["required_measurements"]
    required_measurements = set(measurement_order)
    if set(outputs) != required_measurements:
        fail(f"{role} layer {layer} measurement keys differ from the manifest")
    if [output["key"] for output in document["outputs"]] != measurement_order:
        fail(f"{role} layer {layer} measurement order differs from the manifest")

    expected_dtypes = MEASUREMENT_DTYPES.get(layer)
    if expected_dtypes is None or set(expected_dtypes) != required_measurements:
        fail(f"protected measurement policy is missing manifest layer {layer}")

    policies: dict[str, dict[str, Any]] = {}
    if role == "oracle":
        policies = keyed_evidence_records(
            document.get("comparison"), f"{role} layer {layer} comparison policies"
        )
        if set(policies) != required_measurements:
            fail(f"{role} layer {layer} policy keys differ from the manifest")
    elif document.get("comparison") is not None:
        fail("Mold evidence must not declare oracle comparison policies")

    for key, output in outputs.items():
        expected_dtype = expected_dtypes[key]
        if output.get("dtype") != expected_dtype:
            fail(f"{role} layer {layer} output {key!r} dtype must be {expected_dtype}")
        validate_output_numeric_domain(
            output,
            expected_dtype,
            f"{role} layer {layer} output {key!r}",
        )
        if key in COMPLETE_SAMPLE_MEASUREMENTS.get(layer, frozenset()):
            validate_complete_sample_coverage(
                output, f"{role} layer {layer} output {key!r}"
            )
        if role != "oracle":
            continue
        policy = policies[key]
        if policy.get("metric") != "elementwise-atol-plus-rtol":
            fail(f"oracle layer {layer} output {key!r} policy metric is invalid")
        if expected_dtype == CANONICAL_INTEGER_DTYPE:
            if (
                policy.get("absolute") != 0
                or policy.get("relative") != 0
                or policy.get("hash_policy") != "exact"
            ):
                fail(
                    f"oracle layer {layer} output {key!r} integer policy "
                    "must use zero tolerances and exact hashes"
                )
        else:
            hash_policy = policy.get("hash_policy")
            record_only_allowed = key in RECORD_ONLY_FLOAT_MEASUREMENTS.get(
                layer, frozenset()
            )
            absolute_cap = REVIEWED_ABSOLUTE_TOLERANCE_CAPS.get(
                (layer, key), MAX_EXACT_ABSOLUTE_TOLERANCE
            )
            if (
                not is_finite_float64(policy.get("absolute"))
                or not is_finite_float64(policy.get("relative"))
                or policy["absolute"] > absolute_cap
                or policy["relative"] > MAX_EXACT_RELATIVE_TOLERANCE
                or policy["absolute"] < 0
                or policy["relative"] < 0
                or hash_policy not in {"exact", "record-only"}
                or (hash_policy == "record-only" and not record_only_allowed)
            ):
                fail(
                    f"oracle layer {layer} output {key!r} floating policy "
                    "is not the bounded protected policy"
                )


def validate_exact_outputs(
    document: dict[str, Any],
    fixture: dict[str, Any],
    manifest_layer: dict[str, Any],
    role: str,
    excluded_accelerations: frozenset[str],
) -> None:
    validate_manifest_layer_evidence(
        document, manifest_layer, role, excluded_accelerations
    )

    first_output = document["outputs"][0]
    if fixture["tensor"] != expected_tensor_summary(first_output):
        fail(f"{role} bundle tensor summary does not match layer evidence")

    if role != "oracle":
        return
    policies = {policy["key"]: policy for policy in document["comparison"]}
    first_policy = policies[first_output["key"]]
    if fixture["tolerance"] != expected_tolerance_summary(first_policy):
        fail(f"{role} bundle tolerance summary does not match oracle policy")


def load_layer_documents(
    tool: Any,
    fixture_root: pathlib.Path,
    authorization_sha: str,
    bundle: dict[str, Any],
    role: str,
    source_sha: str,
    layer_tiers: dict[str, str],
    layer_contracts: dict[str, dict[str, Any]],
    excluded_accelerations: frozenset[str],
) -> dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]:
    documents: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for position, fixture in enumerate(bundle["fixtures"], start=1):
        manifest_tier = layer_tiers.get(fixture["layer"])
        if manifest_tier not in EXACT_AUTHORITY_TIERS:
            fail(
                f"{role} bundle layer {fixture['layer']} has manifest authority tier "
                f"{manifest_tier!r}, not an exact authority tier"
            )
        if fixture["authority_tier"] != manifest_tier:
            fail(f"{role} bundle authority tier drifts from the manifest")
        try:
            evidence_path = (fixture_root / fixture["relative_path"]).resolve(
                strict=True
            )
        except OSError:
            fail(f"{role} evidence {position} is unavailable")
        try:
            evidence_path.relative_to(fixture_root)
        except ValueError:
            fail(f"{role} evidence escapes MOLD_H3_FIXTURE_ROOT")
        encoded, raw_document = read_json_once(
            evidence_path, f"{role} evidence {position}", MAX_LAYER_JSON_BYTES
        )
        if hashlib.sha256(encoded).hexdigest() != fixture["sha256"]:
            fail(f"{role} evidence {position} hash does not match its bundle")
        document = call_redacted(
            tool,
            f"{role} layer evidence validation",
            lambda: tool.validate_layer_output(
                raw_document, f"{role} evidence {position}"
            ),
        )
        producer = document["producer"]
        if producer["role"] != role:
            fail(f"{role} bundle contains a different producer role")
        if role == "oracle":
            if (
                producer["source_id"] != "diffusers"
                or producer["revision"] != tool.EXPECTED_REVISIONS["diffusers"]
            ):
                fail("oracle evidence is not from the pinned Diffusers authority")
        elif producer["source_id"] != "mold" or producer["revision"] != source_sha:
            fail("Mold evidence is not from the exact requested source SHA")
        document_manifest_tier = layer_tiers.get(document["layer"])
        if document_manifest_tier not in EXACT_AUTHORITY_TIERS:
            fail(
                f"{role} layer {document['layer']} has manifest authority tier "
                f"{document_manifest_tier!r}, not an exact authority tier"
            )
        if document["authority_tier"] != document_manifest_tier:
            fail(f"{role} layer authority tier drifts from the manifest")
        manifest_layer = layer_contracts.get(document["layer"])
        if manifest_layer is None:
            fail(f"{role} layer lacks a protected manifest contract")
        if document["authorization_document_sha256"] != authorization_sha:
            fail(f"{role} layer is not bound to the authorization evidence")
        if not is_cuda_environment(document["environment"]["device"]):
            fail(f"{role} layer does not declare a CUDA execution environment")
        expected_environment_dtype = layer_environment_dtype(
            document["layer"], document_manifest_tier
        )
        if document["environment"]["dtype"] != expected_environment_dtype:
            fail(f"{role} layer environment dtype must be {expected_environment_dtype}")
        reject_excluded_attention_backend(
            document["environment"].get("attention_backend"),
            excluded_accelerations,
            f"{role} layer {document['layer']}",
        )
        capture_environment = bundle["capture_environment"]
        if document["environment"].get(
            "acceleration_policy"
        ) != capture_environment.get("acceleration_policy"):
            fail(f"{role} layer acceleration policy differs from its bundle")
        for field in ("device", "attention_backend"):
            if document["environment"].get(field) != capture_environment.get(field):
                fail(f"{role} layer {field} differs from its bundle")
        if (
            capture_environment.get("dtype") != CANONICAL_MIXED_CAPTURE_DTYPE
            and document["environment"].get("dtype")
            != capture_environment.get("dtype")
        ):
            fail(f"{role} layer dtype differs from its bundle")
        if (
            "_oracle_runtime" in manifest_layer
            and document["environment"].get("oracle_runtime")
            != capture_environment.get("oracle_runtime")
        ):
            fail(f"{role} layer oracle runtime differs from its bundle")
        if role == "oracle" and manifest_layer.get("oracle_adapter") is not None:
            adapter_records = {
                record["layer"]: record
                for record in capture_environment["oracle_adapters"]
            }
            adapter_record = adapter_records[document["layer"]]
            if {
                "id": document["adapter"].get("id"),
                "command": document["adapter"].get("command"),
                "implementation_sha256": document["adapter"].get(
                    "implementation_sha256"
                ),
            } != {
                "id": adapter_record["id"],
                "command": adapter_record["command"],
                "implementation_sha256": adapter_record["implementation_sha256"],
            }:
                fail(f"{role} layer adapter identity differs from its bundle")
        if fixture["layer"] != document["layer"]:
            fail(f"{role} bundle layer does not match its evidence")
        if fixture["authority_tier"] != document["authority_tier"]:
            fail(f"{role} bundle authority tier does not match its evidence")
        if (
            fixture["component_index_sha256"]
            != document["input"]["component_index_sha256"]
        ):
            fail(f"{role} bundle component index does not match its evidence")
        fixture_components = keyed_component_records(
            fixture.get("component_indexes"), f"{role} bundle component authorities"
        )
        document_components = keyed_component_records(
            document["input"].get("component_indexes"),
            f"{role} layer component authorities",
        )
        if fixture_components != document_components:
            fail(f"{role} bundle component authorities do not match its evidence")
        validate_exact_outputs(
            document, fixture, manifest_layer, role, excluded_accelerations
        )
        key = (document["case_id"], document["layer"])
        if key in documents:
            fail(f"{role} bundle contains duplicate case/layer evidence")
        documents[key] = (document, fixture)
    return documents


def require_complete_exact_coverage(
    documents: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]],
    layer_tiers: dict[str, str],
    role: str,
) -> None:
    required = {
        layer
        for layer, authority_tier in layer_tiers.items()
        if authority_tier in EXACT_AUTHORITY_TIERS
    }
    observed = {layer for _, layer in documents}
    if observed != required:
        fail(
            f"{role} bundle lacks complete exact layer coverage "
            f"(required={len(required)}, observed={len(observed)})"
        )


def validate_oracle_mold_policy_parity(
    oracle_document: dict[str, Any],
    oracle_fixture: dict[str, Any],
    mold_document: dict[str, Any],
    mold_fixture: dict[str, Any],
    manifest_layer: dict[str, Any],
) -> None:
    oracle_outputs = keyed_evidence_records(
        oracle_document["outputs"], "oracle paired outputs"
    )
    mold_outputs = keyed_evidence_records(
        mold_document["outputs"], "Mold paired outputs"
    )
    if set(oracle_outputs) != set(mold_outputs):
        fail("oracle and Mold measurement sets differ")
    policies = keyed_evidence_records(
        oracle_document["comparison"], "oracle paired comparison policies"
    )
    for key in oracle_outputs:
        if oracle_outputs[key]["dtype"] != mold_outputs[key]["dtype"]:
            fail(f"oracle and Mold output {key!r} dtypes differ")
        if (
            policies[key]["hash_policy"] == "exact"
            and oracle_outputs[key]["content_sha256"]
            != mold_outputs[key]["content_sha256"]
        ):
            fail(f"oracle and Mold output {key!r} content hashes differ")
    oracle_provenance = keyed_evidence_records(
        oracle_document["provenance"], "oracle paired provenance"
    )
    mold_provenance = keyed_evidence_records(
        mold_document["provenance"], "Mold paired provenance"
    )
    if oracle_document["input"]["sha256"] != mold_document["input"]["sha256"]:
        fail("oracle and Mold input hashes differ")
    for key in manifest_layer["role_invariant_provenance"]:
        if oracle_provenance[key]["value"] != mold_provenance[key]["value"]:
            fail(f"oracle and Mold provenance {key!r} differs")
    if mold_fixture["tolerance"] != oracle_fixture["tolerance"]:
        fail("Mold bundle tolerance summary differs from oracle policy")


def run_campaign(
    environment: Mapping[str, str],
    gpu_probe: Callable[[], None] = probe_cuda,
    source_probe: Callable[[], str] = probe_source_sha,
) -> dict[str, int | str]:
    values = required_environment(environment)
    if source_probe() != values["MOLD_H3_SOURCE_SHA"]:
        fail("checkout source SHA differs from MOLD_H3_SOURCE_SHA")
    tool = load_tool()
    manifest = call_redacted(
        tool, "checked conformance contract", tool.validate_manifest
    )
    layer_tiers = manifest_layer_tiers(manifest)
    layer_contracts = exact_layer_contracts(manifest)
    excluded_accelerations = manifest_excluded_accelerations(manifest)
    fixture_root = call_redacted(
        tool,
        "external fixture root validation",
        lambda: tool.canonical_external_directory(
            "fixture root", values["MOLD_H3_FIXTURE_ROOT"]
        ),
    )
    authorization_path = resolve_external_file(
        "authorization record", values["MOLD_H3_AUTHORIZATION_RECORD"]
    )
    authorization = call_redacted(
        tool,
        "authorization validation",
        lambda: tool.validate_authorization(authorization_path),
    )
    authorization_sha = authorization["source_document_sha256"]
    if authorization_sha != REVIEWED_AUTHORIZATION_EVIDENCE_SHA256:
        fail("authorization record does not bind the reviewed authorization evidence")
    oracle_bundle_path = resolve_bundle_path(
        fixture_root, "oracle bundle", values["MOLD_H3_ORACLE_BUNDLE"]
    )
    mold_bundle_path = resolve_bundle_path(
        fixture_root, "Mold bundle", values["MOLD_H3_MOLD_BUNDLE"]
    )
    if oracle_bundle_path == mold_bundle_path:
        fail("oracle and Mold bundles must be distinct files")

    gpu_probe()

    oracle_bundle = load_bundle_once(
        tool,
        oracle_bundle_path,
        authorization_sha,
        "oracle bundle",
    )
    mold_bundle = load_bundle_once(
        tool,
        mold_bundle_path,
        authorization_sha,
        "Mold bundle",
    )
    source_sha = values["MOLD_H3_SOURCE_SHA"]
    validate_capture_environment(
        tool,
        oracle_bundle,
        "oracle",
        source_sha,
        excluded_accelerations,
        layer_contracts,
    )
    validate_capture_environment(
        tool,
        mold_bundle,
        "mold",
        source_sha,
        excluded_accelerations,
        layer_contracts,
    )
    oracle_documents = load_layer_documents(
        tool,
        fixture_root,
        authorization_sha,
        oracle_bundle,
        "oracle",
        source_sha,
        layer_tiers,
        layer_contracts,
        excluded_accelerations,
    )
    mold_documents = load_layer_documents(
        tool,
        fixture_root,
        authorization_sha,
        mold_bundle,
        "mold",
        source_sha,
        layer_tiers,
        layer_contracts,
        excluded_accelerations,
    )
    require_complete_exact_coverage(oracle_documents, layer_tiers, "oracle")
    require_complete_exact_coverage(mold_documents, layer_tiers, "mold")
    if not oracle_documents or set(oracle_documents) != set(mold_documents):
        fail(
            "oracle and Mold comparison sets differ "
            f"(oracle={len(oracle_documents)}, mold={len(mold_documents)})"
        )

    notes = 0
    for position, key in enumerate(sorted(oracle_documents), start=1):
        oracle_document, oracle_fixture = oracle_documents[key]
        mold_document, mold_fixture = mold_documents[key]
        validate_oracle_mold_policy_parity(
            oracle_document,
            oracle_fixture,
            mold_document,
            mold_fixture,
            layer_contracts[key[1]],
        )
        comparison_notes = call_redacted(
            tool,
            f"authorized comparison {position}",
            lambda oracle_document=oracle_document, mold_document=mold_document: (
                tool.compare_layer_outputs(oracle_document, mold_document)
            ),
        )
        notes += len(comparison_notes)
    return {
        "comparisons": len(oracle_documents),
        "notes": notes,
        "source_sha": source_sha,
    }


def main() -> int:
    try:
        result = run_campaign(os.environ)
        print(
            "MiniMax H3 private GPU conformance passed: "
            f"comparisons={result['comparisons']} notes={result['notes']} "
            f"source={result['source_sha']}"
        )
        return 0
    except (GpuConformanceFailure, OSError) as error:
        print(f"MiniMax H3 private GPU conformance rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
