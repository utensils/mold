#!/usr/bin/env python3
"""Contract tests for the opt-in MiniMax H3 private GPU conformance runner."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import tempfile
from collections.abc import Callable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"
TOOL_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "minimax-h3-private-conformance.yml"
)
SOURCE_SHA = "a" * 40
REVIEWED_AUTHORIZATION_EVIDENCE_SHA256 = (
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d"
)
CHECKOUT_V6_SHA = "d23441a48e516b6c34aea4fa41551a30e30af803"
CANONICAL_BF16_DTYPE = "bfloat16"
CANONICAL_F16_DTYPE = "float16"
CANONICAL_F32_DTYPE = "float32"
CANONICAL_INTEGER_DTYPE = "int64"
CANONICAL_METRIC_DTYPE = "float64"
CANONICAL_MIXED_CAPTURE_DTYPE = "official-bf16-fp32-mixed"
VISUAL_VAE_ENVIRONMENT_DTYPE = "official-f32-encode-fp16-decode"
QWEN_ORDINARY_ABSOLUTE_TOLERANCE = 48.0
QWEN_ORDINARY_RELATIVE_TOLERANCE = 1.0 / 64.0
QWEN_LARGE_MAGNITUDE_THRESHOLD = 1024.0
QWEN_LARGE_ABSOLUTE_TOLERANCE = 48.0
QWEN_LARGE_RELATIVE_TOLERANCE = 3.0 / 8.0
CANONICAL_E2E_INPUT_SCHEMA = "mold.minimax-h3.e2e-input.v1"
COMPONENT_AUTHORITY_SET_SCHEMA = "mold.minimax-h3.component-authority-set.v1"
E2E_LAYER_TASKS = {
    "end-to-end-t2va": "t2va",
    "end-to-end-fl2va": "fl2va",
    "end-to-end-ref2va": "ref2va",
}
TEST_MEASUREMENT_KINDS = {
    "tokenizer-processor": {
        "token-ids": "integer",
        "special-token-presentation": "integer",
        "processor-shapes": "integer",
        "processor-grids": "integer",
        "sampled-values": "activation",
    },
    "qwen-layer-50": {
        "shape": "integer",
        "dtype": "integer",
        "statistics": "metric",
        "sampled-values": "activation",
        "tolerance": "metric",
    },
    "visual-vae": {
        "pixel-normalization": "float32",
        "posterior-moments": "float32",
        "posterior-seed-42": "float32",
        "posterior-sample": "float32",
        "posterior-fp16-roundtrip": "float16",
        "latent-normalization": "float32",
        "decoded-frames": "float32",
        "tile-seams": "float32",
    },
    "audio-vae": {
        "stereo-packing": "integer",
        "latent-rate": "metric",
        "waveform-statistics": "metric",
        "phase-polarity": "metric",
        "decoded-pcm": "float32",
        "sampled-values": "float32",
    },
    "token-refiner": {
        "shape": "integer",
        "statistics": "metric",
        "sampled-values": "activation",
        "tolerance": "metric",
    },
    "transformer-block": {
        "qk-rms": "metric",
        "partial-mm-rope": "activation",
        "adaln": "activation",
        "video-head": "float32",
        "audio-head": "float32",
        "tolerance": "metric",
    },
    "packed-layout": {
        "row-order": "integer",
        "modality-tags": "integer",
        "rotary-coordinates": "integer",
        "timestep-indices": "integer",
    },
    "noise-allocation": {
        "draw-order": "integer",
        "tensor-shapes": "integer",
        "sampled-values": "activation",
        "hash": "integer",
    },
    "end-to-end-t2va": {
        "latent-statistics": "metric",
        "av-timing": "metric",
        "video-metrics": "metric",
        "audio-metrics": "metric",
    },
    "end-to-end-fl2va": {
        "condition-latents": "activation",
        "latent-statistics": "metric",
        "av-timing": "metric",
        "video-metrics": "metric",
        "audio-metrics": "metric",
    },
    "end-to-end-ref2va": {
        "packed-order": "integer",
        "latent-statistics": "metric",
        "av-timing": "metric",
        "video-metrics": "metric",
        "audio-metrics": "metric",
    },
}
DTYPE_BY_KIND = {
    "integer": CANONICAL_INTEGER_DTYPE,
    "activation": CANONICAL_BF16_DTYPE,
    "float16": CANONICAL_F16_DTYPE,
    "float32": CANONICAL_F32_DTYPE,
    "metric": CANONICAL_METRIC_DTYPE,
}


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
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


def refresh_component_summary(document: dict[str, object]) -> None:
    components = document["input"]["component_indexes"]
    document["input"]["component_index_sha256"] = (
        components[0]["sha256"]
        if len(components) == 1
        else component_authority_set_sha256(
            [component["id"] for component in components],
            {component["id"]: component["sha256"] for component in components},
        )
    )


def expect_failure(action: Callable[[], object], fragment: str) -> None:
    try:
        action()
    except Exception as error:  # noqa: BLE001 - adversarial fail-closed tests
        assert fragment in str(error), (fragment, str(error))
    else:
        raise AssertionError(f"expected failure containing {fragment!r}")


def authorization_fixture(
    tool,
    root: pathlib.Path,
    *,
    name: str = "reviewed",
    reviewed: bool = True,
) -> tuple[pathlib.Path, pathlib.Path, str]:
    source = root / f"{name}-authorization-source.bin"
    source.write_bytes(f"{name} contract evidence".encode())
    source_sha = REVIEWED_AUTHORIZATION_EVIDENCE_SHA256 if reviewed else sha256(source)
    record = {
        "schema_version": tool.AUTHORIZATION_SCHEMA_VERSION,
        "family": "minimax-h3",
        "decision": "approved",
        "license_revision": tool.EXPECTED_REVISIONS["minimax-official-model"],
        "license_sha256": tool.EXPECTED_LICENSE_SHA256,
        "approved_scopes": [
            "checkpoint-execution",
            "fixture-capture",
            "generated-output-retention",
        ],
        "source_document_path": str(source),
        "source_document_sha256": source_sha,
        "review_reference": "contract-test-only",
    }
    path = root / f"{name}-authorization.json"
    write_json(path, record)
    return path, source, source_sha


def exact_manifest_layers(tool) -> dict[str, dict[str, object]]:
    manifest = tool.validate_manifest()
    component_authorities = {
        component["id"]: component["sha256"]
        for component in manifest["component_indexes"]
    }
    layers = {}
    for raw_layer in manifest["fixture_layers"]:
        if raw_layer["authority_tier"] not in {
            "exact-full-bf16",
            "exact-full-fp32",
        }:
            continue
        layer = copy.deepcopy(raw_layer)
        layer["_required_component_authorities"] = [
            {"id": identifier, "sha256": component_authorities[identifier]}
            for identifier in layer["required_component_indexes"]
        ]
        layer["_pinned_provenance_values"] = {
            record["key"]: component_authority_set_sha256(
                record["component_indexes"], component_authorities
            )
            for record in layer["pinned_provenance"]
        }
        if layer["id"] == "tokenizer-processor":
            layer["_pinned_provenance_values"].update(
                {
                    "transformers-revision": tool.EXPECTED_REVISIONS["transformers"],
                    "oracle-runtime-sha256": canonical_json_sha256(
                        tool.EXPECTED_ORACLE_RUNTIME
                    ),
                }
            )
        if layer["authority_tier"] == "exact-full-bf16":
            layer["_oracle_runtime"] = copy.deepcopy(tool.EXPECTED_ORACLE_RUNTIME)
        layer["_excluded_accelerations"] = frozenset(
            manifest["numerical_authority"]["excluded_accelerations"]
        )
        layers[layer["id"]] = layer
    assert len(layers) == 11
    assert "dual-sampler" not in layers
    assert set(layers) == set(TEST_MEASUREMENT_KINDS)
    for identifier, layer in layers.items():
        assert set(layer["required_measurements"]) == set(
            TEST_MEASUREMENT_KINDS[identifier]
        )
    return layers


def output_fixture(layer: str, key: str, kind: str) -> dict[str, object]:
    integer = kind == "integer"
    low: int | float = 1 if integer else 0.25
    high: int | float = 2 if integer else 0.75
    mean: int | float = 1 if integer else 0.5
    deviation: int | float = 0 if integer else 0.25
    return {
        "key": key,
        "shape": [2],
        "dtype": DTYPE_BY_KIND[kind],
        "content_sha256": hashlib.sha256(
            f"gpu-contract:{layer}:{key}:{kind}".encode()
        ).hexdigest(),
        "statistics": {
            "min": low,
            "max": high,
            "mean": mean,
            "std": deviation,
        },
        "samples": [
            {"index": [0], "value": low},
            {"index": [1], "value": high},
        ],
    }


def comparison_fixture(
    key: str, kind: str, layer: str | None = None
) -> dict[str, object]:
    integer = kind == "integer"
    policy: dict[str, object] = {
        "key": key,
        "absolute": 0 if integer else 0.000002,
        "relative": 0 if integer else 0.000001,
        "metric": "elementwise-atol-plus-rtol",
        "hash_policy": "record-only" if kind in {"float16", "float32"} else "exact",
    }
    if layer == "qwen-layer-50" and key in {"statistics", "sampled-values"}:
        policy.update(
            {
                "absolute": QWEN_ORDINARY_ABSOLUTE_TOLERANCE,
                "relative": QWEN_ORDINARY_RELATIVE_TOLERANCE,
                "metric": "piecewise-magnitude-atol-plus-rtol",
                "magnitude_threshold": QWEN_LARGE_MAGNITUDE_THRESHOLD,
                "large_absolute": QWEN_LARGE_ABSOLUTE_TOLERANCE,
                "large_relative": QWEN_LARGE_RELATIVE_TOLERANCE,
                "hash_policy": "record-only",
            }
        )
    return policy


def provenance_fixture(
    tool, manifest_layer: dict[str, object], document: dict[str, object]
) -> list[dict[str, str]]:
    component_indexes = document["input"]["component_indexes"]
    component_index_hashes = ",".join(
        f"{component['id']}={component['sha256']}"
        for component in sorted(
            component_indexes, key=lambda component: component["id"]
        )
    )
    values = {
        "source-revision": document["producer"]["revision"],
        "diffusers-revision": tool.EXPECTED_REVISIONS["diffusers"],
        "transformers-revision": tool.EXPECTED_REVISIONS["transformers"],
        "adapter-implementation-sha256": document["adapter"].get(
            "implementation_sha256", "0" * 64
        ),
        "checkpoint-sha256": document["input"].get("checkpoint_sha256", "0" * 64),
        "oracle-runtime-sha256": canonical_json_sha256(tool.EXPECTED_ORACLE_RUNTIME),
        "runtime-sha256": canonical_json_sha256(
            document["environment"].get("runtime", {})
        ),
        "tokenizer-hash": "0" * 64,
        "processor-hash": "0" * 64,
        "component-index-hash": document["input"]["component_index_sha256"],
        "component-index-hashes": component_index_hashes,
        "artifact-set-sha256": "d" * 64,
        "device": document["environment"]["device"],
        "generator-device": document["environment"]["device"],
        "dtype": document["environment"]["dtype"],
        "attention-backend": document["environment"]["attention_backend"],
        "capture-command": document["adapter"]["command"],
        "workflow": "private-conformance-v1",
        "reference-order": "source-then-reference-v1",
        "seed": "42",
        "allocation-version": "canonical-draw-order-v1",
        "float-policy": "coupled-bfloat16-v1",
        "video-shift": "8.0",
        "audio-shift": "5.0",
        "endpoint-signature": "fl2va-v1",
    }
    values.update(manifest_layer["_pinned_provenance_values"])
    return [
        {"key": key, "value": values[key]}
        for key in manifest_layer["required_provenance"]
    ]


def layer_document(
    tool,
    role: str,
    authorization_sha: str,
    manifest_layer: dict[str, object],
) -> dict[str, object]:
    layer = manifest_layer["id"]
    component_indexes = copy.deepcopy(manifest_layer["_required_component_authorities"])
    component_summary_sha256 = (
        component_indexes[0]["sha256"]
        if len(component_indexes) == 1
        else tool.component_authority_set_sha256(component_indexes)
    )
    input_evidence: dict[str, object] = {
        "id": f"gpu-contract-{layer}",
        "sha256": hashlib.sha256(f"input:{layer}".encode()).hexdigest(),
        "component_index_sha256": component_summary_sha256,
        "component_indexes": component_indexes,
    }
    if layer == "audio-vae":
        input_evidence["checkpoint_sha256"] = (
            tool.AUDIO_VAE_CHECKPOINT_SHA256_BY_ROLE[role]
        )
    task = E2E_LAYER_TASKS.get(layer)
    if task is not None:
        conditioning = []
        if task == "fl2va":
            conditioning = [
                {
                    "role": "first-frame",
                    "sha256": hashlib.sha256(b"contract-first-frame").hexdigest(),
                },
                {
                    "role": "last-frame",
                    "sha256": hashlib.sha256(b"contract-last-frame").hexdigest(),
                },
            ]
        elif task == "ref2va":
            conditioning = [
                {
                    "role": "reference-image",
                    "sha256": hashlib.sha256(b"contract-reference-image").hexdigest(),
                },
                {
                    "role": "reference-audio",
                    "sha256": hashlib.sha256(b"contract-reference-audio").hexdigest(),
                },
            ]
        conformance = {
            "schema_version": CANONICAL_E2E_INPUT_SCHEMA,
            "task": task,
            "prompt_sha256": hashlib.sha256(b"contract prompt").hexdigest(),
            "negative_prompt_sha256": hashlib.sha256(b"").hexdigest(),
            "conditioning": conditioning,
            "seed": 42,
            "width": 1280,
            "height": 768,
            "frames": 362,
            "fps": 24,
            "video_sampler": {
                "algorithm": "minimax-h3-flow-euler-v1",
                "steps": 30,
                "guidance": "0",
                "shift": "12",
            },
            "audio_sampler": {
                "algorithm": "minimax-h3-flow-euler-v1",
                "steps": 30,
                "guidance": "0",
                "shift": "3",
            },
        }
        input_evidence["conformance"] = conformance
        input_evidence["sha256"] = canonical_json_sha256(conformance)
    outputs = [
        output_fixture(layer, key, TEST_MEASUREMENT_KINDS[layer][key])
        for key in manifest_layer["required_measurements"]
    ]
    oracle_adapter = manifest_layer.get("oracle_adapter")
    command = (
        f"{oracle_adapter['command_prefix']} --device cuda:contract-test"
        if role == "oracle" and oracle_adapter is not None
        else f"contract-test {role} {layer} capture"
    )
    adapter = {
        "schema_version": "mold.minimax-h3.layer-adapter.v1",
        "id": (
            oracle_adapter["id"]
            if role == "oracle" and oracle_adapter is not None
            else f"contract-{role}-adapter"
        ),
        "command": command,
        "tensor_hash_encoding": "canonical-typed-le-v1",
    }
    if "adapter-implementation-sha256" in manifest_layer["required_provenance"]:
        adapter["implementation_sha256"] = (
            oracle_adapter["implementation_sha256"]
            if role == "oracle" and oracle_adapter is not None
            else "1" * 64
        )
    environment = {
        "device": "cuda:contract-test",
        "dtype": (
            VISUAL_VAE_ENVIRONMENT_DTYPE
            if layer == "visual-vae"
            else (
                CANONICAL_F32_DTYPE
                if manifest_layer["authority_tier"] == "exact-full-fp32"
                else CANONICAL_BF16_DTYPE
            )
        ),
        "attention_backend": "math",
        "acceleration_policy": {
            "enabled": ["math"],
            "disabled": sorted(manifest_layer["_excluded_accelerations"]),
        },
        "forbidden_accelerations_disabled": True,
    }
    if layer == "audio-vae":
        environment["runtime"] = {"python": "3.contract"}
    if role == "oracle" and manifest_layer["authority_tier"] == "exact-full-bf16":
        environment["oracle_runtime"] = copy.deepcopy(tool.EXPECTED_ORACLE_RUNTIME)
    document: dict[str, object] = {
        "schema_version": tool.LAYER_OUTPUT_SCHEMA_VERSION,
        "family": "minimax-h3",
        "case_id": "gpu-contract-case",
        "layer": layer,
        "authority_tier": manifest_layer["authority_tier"],
        "authorization_document_sha256": authorization_sha,
        "input": input_evidence,
        "producer": {
            "role": role,
            "implementation": f"contract-{role}",
            "source_id": "diffusers" if role == "oracle" else "mold",
            "revision": (
                tool.EXPECTED_REVISIONS["diffusers"] if role == "oracle" else SOURCE_SHA
            ),
        },
        "adapter": adapter,
        "environment": environment,
        "outputs": outputs,
    }
    document["provenance"] = provenance_fixture(tool, manifest_layer, document)
    if role == "oracle":
        document["comparison"] = [
            comparison_fixture(key, TEST_MEASUREMENT_KINDS[layer][key], layer)
            for key in manifest_layer["required_measurements"]
        ]
    return document


def exact_layer_ids(tool) -> list[str]:
    return list(exact_manifest_layers(tool))


def bundle_fixture(
    tool,
    fixture_root: pathlib.Path,
    authorization_sha: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    manifest_sha = sha256(tool.MANIFEST_PATH)
    manifest_layers = exact_manifest_layers(tool)
    excluded_accelerations = sorted(
        tool.validate_manifest()["numerical_authority"]["excluded_accelerations"]
    )
    bundle_paths: list[pathlib.Path] = []
    for role in ("oracle", "mold"):
        fixtures = []
        for layer, manifest_layer in manifest_layers.items():
            document = layer_document(tool, role, authorization_sha, manifest_layer)
            evidence_path = fixture_root / role / f"{layer}.json"
            write_json(evidence_path, document)
            output = document["outputs"][0]
            comparison = comparison_fixture(
                output["key"], TEST_MEASUREMENT_KINDS[layer][output["key"]], layer
            )
            fixtures.append(
                {
                    "id": f"gpu-contract-{layer}-{role}",
                    "layer": layer,
                    "authority_tier": document["authority_tier"],
                    "relative_path": str(evidence_path.relative_to(fixture_root)),
                    "sha256": sha256(evidence_path),
                    "component_index_sha256": document["input"][
                        "component_index_sha256"
                    ],
                    "component_indexes": copy.deepcopy(
                        document["input"]["component_indexes"]
                    ),
                    "tensor": {
                        "shape": output["shape"],
                        "dtype": output["dtype"],
                        "min": output["statistics"]["min"],
                        "max": output["statistics"]["max"],
                        "mean": output["statistics"]["mean"],
                        "std": output["statistics"]["std"],
                        "sampled_values": [
                            sample["value"] for sample in output["samples"]
                        ],
                    },
                    "tolerance": {
                        "absolute": comparison["absolute"],
                        "relative": comparison["relative"],
                        "metric": comparison["metric"],
                    },
                }
            )
        bundle = {
            "schema_version": tool.BUNDLE_SCHEMA_VERSION,
            "manifest_sha256": manifest_sha,
            "authorization_document_sha256": authorization_sha,
            "capture_environment": {
                "framework": "diffusers" if role == "oracle" else "mold",
                "framework_revision": (
                    tool.EXPECTED_REVISIONS["diffusers"]
                    if role == "oracle"
                    else SOURCE_SHA
                ),
                "device": "cuda:contract-test",
                "dtype": CANONICAL_MIXED_CAPTURE_DTYPE,
                "attention_backend": "math",
                "acceleration_policy": {
                    "enabled": ["math"],
                    "disabled": excluded_accelerations,
                },
                "command": f"contract-test {role} capture",
                "forbidden_accelerations_disabled": True,
            },
            "fixtures": fixtures,
        }
        if role == "oracle":
            bundle["capture_environment"]["oracle_runtime"] = copy.deepcopy(
                tool.EXPECTED_ORACLE_RUNTIME
            )
            bundle["capture_environment"]["oracle_adapters"] = [
                {
                    "layer": layer,
                    "id": contract["oracle_adapter"]["id"],
                    "command": (
                        f"{contract['oracle_adapter']['command_prefix']} "
                        "--device cuda:contract-test"
                    ),
                    "implementation_sha256": contract["oracle_adapter"][
                        "implementation_sha256"
                    ],
                }
                for layer, contract in sorted(manifest_layers.items())
                if contract.get("oracle_adapter") is not None
            ]
        bundle_path = fixture_root / f"{role}-bundle.json"
        write_json(bundle_path, bundle)
        bundle_paths.append(bundle_path)
    return bundle_paths[0], bundle_paths[1]


def campaign_environment(
    fixture_root: pathlib.Path,
    authorization: pathlib.Path,
    oracle_bundle: pathlib.Path,
    mold_bundle: pathlib.Path,
) -> dict[str, str]:
    return {
        "MOLD_H3_FIXTURE_ROOT": str(fixture_root),
        "MOLD_H3_AUTHORIZATION_RECORD": str(authorization),
        "MOLD_H3_ORACLE_BUNDLE": str(oracle_bundle),
        "MOLD_H3_MOLD_BUNDLE": str(mold_bundle),
        "MOLD_H3_SOURCE_SHA": SOURCE_SHA,
    }


def mutate_evidence(
    fixture_root: pathlib.Path,
    bundle_path: pathlib.Path,
    mutate_document: Callable[[dict[str, object]], None] | None = None,
    mutate_fixture: Callable[[dict[str, object]], None] | None = None,
    fixture_layer: str | None = None,
) -> tuple[dict[str, object], pathlib.Path]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    fixture = (
        bundle["fixtures"][0]
        if fixture_layer is None
        else next(
            candidate
            for candidate in bundle["fixtures"]
            if candidate["layer"] == fixture_layer
        )
    )
    evidence_path = fixture_root / fixture["relative_path"]
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    if mutate_document is not None:
        mutate_document(document)
        write_json(evidence_path, document)
        fixture["sha256"] = sha256(evidence_path)
    if mutate_fixture is not None:
        mutate_fixture(fixture)
    write_json(bundle_path, bundle)
    return bundle, evidence_path


def record_by_key(records: list[dict[str, object]], key: str) -> dict[str, object]:
    return next(record for record in records if record["key"] == key)


def rehash_e2e_input(document: dict[str, object]) -> None:
    document["input"]["sha256"] = canonical_json_sha256(
        document["input"]["conformance"]
    )


def test_manifest_layer_contract(runner, tool, authorization_sha: str) -> None:
    manifest = tool.validate_manifest()
    manifest_layers = exact_manifest_layers(tool)
    assert set(runner.exact_layer_contracts(manifest)) == set(manifest_layers)
    assert manifest_layers["tokenizer-processor"]["_pinned_provenance_values"] == {
        "tokenizer-hash": (
            "00c15d010418b4b43af4bf87555435e46a5bf6c1fc2cd4cc1c8c545375ca1547"
        ),
        "processor-hash": (
            "44e3ff907049587aa580bb17f0374bb78e704e74efbf4626ed6366f4d15be287"
        ),
        "transformers-revision": tool.EXPECTED_REVISIONS["transformers"],
        "oracle-runtime-sha256": canonical_json_sha256(tool.EXPECTED_ORACLE_RUNTIME),
    }
    bfloat16_max = float.fromhex("0x1.fep+127")
    assert runner.is_exact_bfloat16(bfloat16_max)
    assert runner.is_exact_bfloat16(-bfloat16_max)
    assert not runner.is_exact_bfloat16(0.1)
    assert runner.is_finite_float64(float.fromhex("0x1.fffffffffffffp+1023"))
    assert runner.is_finite_float64(2**80)
    assert not runner.is_finite_float64(2**80 + 1)

    remapped_manifest = copy.deepcopy(manifest)
    tokenizer_contract = next(
        layer
        for layer in remapped_manifest["fixture_layers"]
        if layer["id"] == "tokenizer-processor"
    )
    tokenizer_contract["pinned_provenance"][0]["component_indexes"] = [
        "official-license"
    ]
    expect_failure(
        lambda: runner.exact_layer_contracts(remapped_manifest),
        "invalid pinned provenance",
    )
    narrowed_alias_manifest = copy.deepcopy(manifest)
    narrowed_alias_manifest["numerical_authority"]["excluded_acceleration_aliases"][
        "fp8"
    ].remove("float8")
    expect_failure(
        lambda: runner.exact_layer_contracts(narrowed_alias_manifest),
        "invalid acceleration exclusions",
    )
    excluded_accelerations = runner.manifest_excluded_accelerations(manifest)
    component_authorities = {
        component["id"]: component["sha256"]
        for component in manifest["component_indexes"]
    }

    for layer, manifest_layer in manifest_layers.items():
        assert manifest_layer["_required_component_authorities"]
        assert set(manifest_layer["role_invariant_provenance"]).issubset(
            manifest_layer["required_provenance"]
        )
        for key, kind in TEST_MEASUREMENT_KINDS[layer].items():
            assert runner.MEASUREMENT_DTYPES[layer][key] == DTYPE_BY_KIND[kind]

        for role in ("oracle", "mold"):
            document = layer_document(tool, role, authorization_sha, manifest_layer)
            tool.validate_layer_output(document, f"valid {role} {layer}")
            runner.validate_manifest_layer_evidence(document, manifest_layer, role)

            wrong_encoding = copy.deepcopy(document)
            wrong_encoding["adapter"]["tensor_hash_encoding"] = "partial-record-v1"
            expect_failure(
                lambda wrong_encoding=wrong_encoding, manifest_layer=manifest_layer, role=role: (
                    runner.validate_manifest_layer_evidence(
                        wrong_encoding, manifest_layer, role
                    )
                ),
                "tensor hash encoding is not canonical",
            )

            for provenance_key in manifest_layer["_pinned_provenance_values"]:
                unpinned = copy.deepcopy(document)
                record_by_key(unpinned["provenance"], provenance_key)["value"] = "f" * (
                    40 if provenance_key.endswith("revision") else 64
                )
                expect_failure(
                    lambda unpinned=unpinned, manifest_layer=manifest_layer, role=role: (
                        runner.validate_manifest_layer_evidence(
                            unpinned, manifest_layer, role
                        )
                    ),
                    f"provenance {provenance_key!r} is not authority-pinned",
                )

        if len(manifest_layer["_required_component_authorities"]) > 1:
            reordered_components = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            reordered_components["input"]["component_indexes"].reverse()
            tool.validate_layer_output(
                reordered_components, f"reordered component set {layer}"
            )
            runner.validate_manifest_layer_evidence(
                reordered_components, manifest_layer, "oracle"
            )

        unrelated_component_id = next(
            identifier
            for identifier in component_authorities
            if identifier not in manifest_layer["required_component_indexes"]
        )
        unrelated = layer_document(tool, "oracle", authorization_sha, manifest_layer)
        unrelated["input"]["component_indexes"][0] = {
            "id": unrelated_component_id,
            "sha256": component_authorities[unrelated_component_id],
        }
        refresh_component_summary(unrelated)
        for provenance_key in ("component-index-hash", "component-index-hashes"):
            if provenance_key not in manifest_layer["required_provenance"]:
                continue
            record_by_key(unrelated["provenance"], provenance_key)["value"] = (
                runner.structured_provenance_value(unrelated, provenance_key)
            )
        tool.validate_layer_output(unrelated, f"globally valid unrelated {layer}")
        expect_failure(
            lambda unrelated=unrelated, manifest_layer=manifest_layer: (
                runner.validate_manifest_layer_evidence(
                    unrelated, manifest_layer, "oracle"
                )
            ),
            "component authorities differ from the manifest",
        )

        extra_component = layer_document(
            tool, "oracle", authorization_sha, manifest_layer
        )
        extra_component["input"]["component_indexes"].append(
            {
                "id": unrelated_component_id,
                "sha256": component_authorities[unrelated_component_id],
            }
        )
        refresh_component_summary(extra_component)
        tool.validate_layer_output(extra_component, f"extra component {layer}")
        expect_failure(
            lambda extra_component=extra_component, manifest_layer=manifest_layer: (
                runner.validate_manifest_layer_evidence(
                    extra_component, manifest_layer, "oracle"
                )
            ),
            "component authorities differ from the manifest",
        )

        if len(manifest_layer["_required_component_authorities"]) > 1:
            missing_one_component = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            missing_one_component["input"]["component_indexes"].pop()
            refresh_component_summary(missing_one_component)
            tool.validate_layer_output(
                missing_one_component, f"missing one component {layer}"
            )
            expect_failure(
                lambda missing_one_component=missing_one_component, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        missing_one_component, manifest_layer, "oracle"
                    )
                ),
                "component authorities differ from the manifest",
            )

        missing_components = layer_document(
            tool, "oracle", authorization_sha, manifest_layer
        )
        del missing_components["input"]["component_indexes"]
        missing_components["input"]["component_index_sha256"] = manifest_layer[
            "_required_component_authorities"
        ][0]["sha256"]
        tool.validate_layer_output(
            missing_components, f"legacy component summary {layer}"
        )
        expect_failure(
            lambda missing_components=missing_components, manifest_layer=manifest_layer: (
                runner.validate_manifest_layer_evidence(
                    missing_components, manifest_layer, "oracle"
                )
            ),
            "non-empty component authority list",
        )

        if layer in E2E_LAYER_TASKS:
            for role in ("oracle", "mold"):
                unsigned_boundary = layer_document(
                    tool, role, authorization_sha, manifest_layer
                )
                unsigned_boundary["input"]["conformance"]["seed"] = (
                    runner.UNSIGNED_INT64_MAX
                )
                rehash_e2e_input(unsigned_boundary)
                runner.validate_manifest_layer_evidence(
                    unsigned_boundary, manifest_layer, role
                )

                unsigned_overflow = layer_document(
                    tool, role, authorization_sha, manifest_layer
                )
                unsigned_overflow["input"]["conformance"]["seed"] = (
                    runner.UNSIGNED_INT64_MAX + 1
                )
                rehash_e2e_input(unsigned_overflow)
                expect_failure(
                    lambda unsigned_overflow=unsigned_overflow, manifest_layer=manifest_layer, role=role: (
                        runner.validate_manifest_layer_evidence(
                            unsigned_overflow, manifest_layer, role
                        )
                    ),
                    "seed must be unsigned int64",
                )

                dimension_overflow = layer_document(
                    tool, role, authorization_sha, manifest_layer
                )
                dimension_overflow["input"]["conformance"]["width"] = 2**63
                rehash_e2e_input(dimension_overflow)
                expect_failure(
                    lambda dimension_overflow=dimension_overflow, manifest_layer=manifest_layer, role=role: (
                        runner.validate_manifest_layer_evidence(
                            dimension_overflow, manifest_layer, role
                        )
                    ),
                    "width must be a positive signed int64",
                )

            unhashed_input = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            unhashed_input["input"]["conformance"]["seed"] = 43
            expect_failure(
                lambda unhashed_input=unhashed_input, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        unhashed_input, manifest_layer, "oracle"
                    )
                ),
                "input hash does not match canonical evidence",
            )

            semantic_mutations: list[
                tuple[Callable[[dict[str, object]], None], str]
            ] = [
                (
                    lambda document: document["input"]["conformance"].update(
                        {
                            "task": (
                                "fl2va" if E2E_LAYER_TASKS[layer] == "t2va" else "t2va"
                            )
                        }
                    ),
                    "end-to-end input task is invalid",
                ),
                (
                    lambda document: document["input"]["conformance"].update(
                        {"width": 1279}
                    ),
                    "dimensions violate",
                ),
                (
                    lambda document: document["input"]["conformance"].update(
                        {"frames": 123}
                    ),
                    "frame contract is invalid",
                ),
                (
                    lambda document: document["input"]["conformance"].update(
                        {"fps": 25}
                    ),
                    "frame contract is invalid",
                ),
                (
                    lambda document: document["input"]["conformance"][
                        "video_sampler"
                    ].update({"guidance": "1"}),
                    "sampler contract is invalid",
                ),
                (
                    lambda document: document["input"]["conformance"][
                        "audio_sampler"
                    ].update({"steps": 29}),
                    "sampler contract is invalid",
                ),
                (
                    lambda document: document["input"]["conformance"].update(
                        {"negative_prompt_sha256": "f" * 64}
                    ),
                    "negative prompt must be absent",
                ),
            ]
            if layer == "end-to-end-t2va":
                semantic_mutations.append(
                    (
                        lambda document: document["input"]["conformance"].update(
                            {
                                "conditioning": [
                                    {"role": "first-frame", "sha256": "f" * 64}
                                ]
                            }
                        ),
                        "T2VA input must not carry conditioning media",
                    )
                )
            elif layer == "end-to-end-fl2va":
                semantic_mutations.append(
                    (
                        lambda document: document["input"]["conformance"].update(
                            {"conditioning": []}
                        ),
                        "FL2VA conditioning order is invalid",
                    )
                )
            else:
                semantic_mutations.append(
                    (
                        lambda document: document["input"]["conformance"].update(
                            {"conditioning": []}
                        ),
                        "Ref2VA conditioning order is invalid",
                    )
                )
            for mutate_input, expected_error in semantic_mutations:
                invalid_input = layer_document(
                    tool, "oracle", authorization_sha, manifest_layer
                )
                mutate_input(invalid_input)
                rehash_e2e_input(invalid_input)
                expect_failure(
                    lambda invalid_input=invalid_input, manifest_layer=manifest_layer: (
                        runner.validate_manifest_layer_evidence(
                            invalid_input, manifest_layer, "oracle"
                        )
                    ),
                    expected_error,
                )

        generic = layer_document(tool, "oracle", authorization_sha, manifest_layer)
        generic["outputs"] = [output_fixture(layer, "activation", "activation")]
        generic["comparison"] = [comparison_fixture("activation", "activation")]
        expect_failure(
            lambda generic=generic, manifest_layer=manifest_layer: (
                runner.validate_manifest_layer_evidence(
                    generic, manifest_layer, "oracle"
                )
            ),
            "measurement keys differ from the manifest",
        )

        reordered = layer_document(tool, "oracle", authorization_sha, manifest_layer)
        reordered["outputs"][0], reordered["outputs"][1] = (
            reordered["outputs"][1],
            reordered["outputs"][0],
        )
        expect_failure(
            lambda reordered=reordered, manifest_layer=manifest_layer: (
                runner.validate_manifest_layer_evidence(
                    reordered, manifest_layer, "oracle"
                )
            ),
            "measurement order differs from the manifest",
        )

        for provenance_key in manifest_layer["required_provenance"]:
            missing = layer_document(tool, "oracle", authorization_sha, manifest_layer)
            missing["provenance"] = [
                record
                for record in missing["provenance"]
                if record["key"] != provenance_key
            ]
            expect_failure(
                lambda missing=missing, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        missing, manifest_layer, "oracle"
                    )
                ),
                "provenance keys differ from the manifest",
            )

            renamed = layer_document(tool, "oracle", authorization_sha, manifest_layer)
            record_by_key(renamed["provenance"], provenance_key)["key"] = (
                f"{provenance_key}-renamed"
            )
            expect_failure(
                lambda renamed=renamed, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        renamed, manifest_layer, "oracle"
                    )
                ),
                "provenance keys differ from the manifest",
            )

            noncanonical = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            record = record_by_key(noncanonical["provenance"], provenance_key)
            record["value"] = f" {record['value']}"
            expect_failure(
                lambda noncanonical=noncanonical, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        noncanonical, manifest_layer, "oracle"
                    )
                ),
                "is not canonical",
            )

            empty = layer_document(tool, "oracle", authorization_sha, manifest_layer)
            record_by_key(empty["provenance"], provenance_key)["value"] = ""
            expect_failure(
                lambda empty=empty, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        empty, manifest_layer, "oracle"
                    )
                ),
                "is not canonical",
            )

        structured_drifts = {
            "source-revision": "b" * 40,
            "component-index-hash": "b" * 64,
            "component-index-hashes": "b" * 64,
            "device": "cuda:other",
            "generator-device": "cuda:other",
            "dtype": "float32",
            "attention-backend": "flash",
            "capture-command": "different capture command",
            "checkpoint-sha256": "b" * 64,
            "adapter-implementation-sha256": "b" * 64,
            "runtime-sha256": "b" * 64,
        }
        for provenance_key, value in structured_drifts.items():
            if provenance_key not in manifest_layer["required_provenance"]:
                continue
            drifted = layer_document(tool, "oracle", authorization_sha, manifest_layer)
            if provenance_key == "dtype":
                value = (
                    CANONICAL_BF16_DTYPE
                    if drifted["environment"]["dtype"] == CANONICAL_F32_DTYPE
                    else CANONICAL_F32_DTYPE
                )
            record_by_key(drifted["provenance"], provenance_key)["value"] = value
            expect_failure(
                lambda drifted=drifted, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        drifted, manifest_layer, "oracle"
                    )
                ),
                "does not match structured evidence",
            )

        if layer == "tokenizer-processor":
            missing_acceleration_policy = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            del missing_acceleration_policy["environment"]["acceleration_policy"]
            expect_failure(
                lambda: runner.validate_manifest_layer_evidence(
                    missing_acceleration_policy,
                    manifest_layer,
                    "oracle",
                    excluded_accelerations,
                ),
                "lacks structured acceleration evidence",
            )

        for excluded_acceleration in sorted(excluded_accelerations):
            missing_exclusion = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            missing_exclusion["environment"]["acceleration_policy"]["disabled"].remove(
                excluded_acceleration
            )
            expect_failure(
                lambda missing_exclusion=missing_exclusion, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        missing_exclusion,
                        manifest_layer,
                        "oracle",
                        excluded_accelerations,
                    )
                ),
                "does not disable every manifest-excluded acceleration",
            )

            enabled_exclusion = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            enabled_exclusion["environment"]["acceleration_policy"]["enabled"].append(
                excluded_acceleration
            )
            expect_failure(
                lambda enabled_exclusion=enabled_exclusion, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        enabled_exclusion,
                        manifest_layer,
                        "oracle",
                        excluded_accelerations,
                    )
                ),
                "enables a manifest-excluded acceleration",
            )

        for excluded_alias in ("float8-e4m3fn", "sage-attn", "euler-ancestral"):
            enabled_alias = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            enabled_alias["environment"]["acceleration_policy"]["enabled"].append(
                excluded_alias
            )
            expect_failure(
                lambda enabled_alias=enabled_alias, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        enabled_alias,
                        manifest_layer,
                        "oracle",
                        excluded_accelerations,
                    )
                ),
                "enables a manifest-excluded acceleration",
            )

        if layer == "tokenizer-processor":
            for policy_mutation in ("duplicate", "uppercase"):
                noncanonical_policy = layer_document(
                    tool, "oracle", authorization_sha, manifest_layer
                )
                disabled = noncanonical_policy["environment"]["acceleration_policy"][
                    "disabled"
                ]
                if policy_mutation == "duplicate":
                    disabled.append(disabled[0])
                else:
                    disabled[0] = disabled[0].upper()
                expect_failure(
                    lambda noncanonical_policy=noncanonical_policy: (
                        runner.validate_manifest_layer_evidence(
                            noncanonical_policy,
                            manifest_layer,
                            "oracle",
                            excluded_accelerations,
                        )
                    ),
                    "structured acceleration evidence is not canonical",
                )

        if "attention-backend" in manifest_layer["required_provenance"]:
            excluded_backend = next(iter(excluded_accelerations))
            excluded = layer_document(tool, "oracle", authorization_sha, manifest_layer)
            excluded["environment"]["attention_backend"] = excluded_backend
            record_by_key(excluded["provenance"], "attention-backend")["value"] = (
                excluded_backend
            )
            expect_failure(
                lambda excluded=excluded, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        excluded,
                        manifest_layer,
                        "oracle",
                        excluded_accelerations,
                    )
                ),
                "manifest-excluded acceleration",
            )
            for excluded_alias in (
                "float8-e4m3fn",
                "sage-attn",
                "euler-ancestral",
            ):
                excluded = layer_document(
                    tool, "oracle", authorization_sha, manifest_layer
                )
                excluded["environment"]["attention_backend"] = excluded_alias
                record_by_key(excluded["provenance"], "attention-backend")["value"] = (
                    excluded_alias
                )
                expect_failure(
                    lambda excluded=excluded, manifest_layer=manifest_layer: (
                        runner.validate_manifest_layer_evidence(
                            excluded,
                            manifest_layer,
                            "oracle",
                            excluded_accelerations,
                        )
                    ),
                    "manifest-excluded acceleration",
                )

        for measurement in manifest_layer["required_measurements"]:
            missing = layer_document(tool, "oracle", authorization_sha, manifest_layer)
            missing["outputs"] = [
                output for output in missing["outputs"] if output["key"] != measurement
            ]
            missing["comparison"] = [
                policy
                for policy in missing["comparison"]
                if policy["key"] != measurement
            ]
            expect_failure(
                lambda missing=missing, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        missing, manifest_layer, "oracle"
                    )
                ),
                "measurement keys differ from the manifest",
            )

            renamed = layer_document(tool, "oracle", authorization_sha, manifest_layer)
            record_by_key(renamed["outputs"], measurement)["key"] = (
                f"{measurement}-renamed"
            )
            record_by_key(renamed["comparison"], measurement)["key"] = (
                f"{measurement}-renamed"
            )
            expect_failure(
                lambda renamed=renamed, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        renamed, manifest_layer, "oracle"
                    )
                ),
                "measurement keys differ from the manifest",
            )

            kind = TEST_MEASUREMENT_KINDS[layer][measurement]
            wrong_dtype = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            output = record_by_key(wrong_dtype["outputs"], measurement)
            output["dtype"] = {
                "integer": CANONICAL_METRIC_DTYPE,
                "activation": CANONICAL_METRIC_DTYPE,
                "float16": CANONICAL_F32_DTYPE,
                "float32": CANONICAL_F16_DTYPE,
                "metric": CANONICAL_BF16_DTYPE,
            }[kind]
            expect_failure(
                lambda wrong_dtype=wrong_dtype, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        wrong_dtype, manifest_layer, "oracle"
                    )
                ),
                "dtype must be",
            )

            for role in ("oracle", "mold"):
                for summary_key in ("mean", "std"):
                    summary_overflow = layer_document(
                        tool, role, authorization_sha, manifest_layer
                    )
                    record_by_key(summary_overflow["outputs"], measurement)[
                        "statistics"
                    ][summary_key] = 10**400
                    expect_failure(
                        lambda summary_overflow=summary_overflow, manifest_layer=manifest_layer, role=role: (
                            runner.validate_manifest_layer_evidence(
                                summary_overflow, manifest_layer, role
                            )
                        ),
                        "summary is outside the finite float64 domain",
                    )

                if kind == "activation":
                    for invalid_value in (0.1, 10**100):
                        for location in ("min", "max", "sample"):
                            invalid_activation = layer_document(
                                tool, role, authorization_sha, manifest_layer
                            )
                            activation_output = record_by_key(
                                invalid_activation["outputs"], measurement
                            )
                            if location == "sample":
                                activation_output["samples"][0]["value"] = invalid_value
                            else:
                                activation_output["statistics"][location] = (
                                    invalid_value
                                )
                            expect_failure(
                                lambda invalid_activation=invalid_activation, manifest_layer=manifest_layer, role=role: (
                                    runner.validate_manifest_layer_evidence(
                                        invalid_activation, manifest_layer, role
                                    )
                                ),
                                "must be exact bfloat16",
                            )
                elif kind in {"float16", "float32"}:
                    for invalid_value in (0.1, 10**100):
                        for location in ("min", "max", "sample"):
                            invalid_float = layer_document(
                                tool, role, authorization_sha, manifest_layer
                            )
                            float_output = record_by_key(
                                invalid_float["outputs"], measurement
                            )
                            if location == "sample":
                                float_output["samples"][0]["value"] = invalid_value
                            else:
                                float_output["statistics"][location] = invalid_value
                            expect_failure(
                                lambda invalid_float=invalid_float, manifest_layer=manifest_layer, role=role: (
                                    runner.validate_manifest_layer_evidence(
                                        invalid_float, manifest_layer, role
                                    )
                                ),
                                f"must be exact {kind}",
                            )
                elif kind == "metric":
                    for invalid_value in (10**400, 2**80 + 1):
                        for location in ("min", "max", "sample"):
                            invalid_metric = layer_document(
                                tool, role, authorization_sha, manifest_layer
                            )
                            metric_output = record_by_key(
                                invalid_metric["outputs"], measurement
                            )
                            if location == "sample":
                                metric_output["samples"][0]["value"] = invalid_value
                            else:
                                metric_output["statistics"][location] = invalid_value
                            expect_failure(
                                lambda invalid_metric=invalid_metric, manifest_layer=manifest_layer, role=role: (
                                    runner.validate_manifest_layer_evidence(
                                        invalid_metric, manifest_layer, role
                                    )
                                ),
                                "must be finite float64",
                            )

            if kind == "integer":
                fractional_sample = layer_document(
                    tool, "oracle", authorization_sha, manifest_layer
                )
                record_by_key(fractional_sample["outputs"], measurement)["samples"][0][
                    "value"
                ] = 0.25
                expect_failure(
                    lambda fractional_sample=fractional_sample, manifest_layer=manifest_layer: (
                        runner.validate_manifest_layer_evidence(
                            fractional_sample, manifest_layer, "oracle"
                        )
                    ),
                    "integer min/max and samples",
                )

                fractional_mold_sample = layer_document(
                    tool, "mold", authorization_sha, manifest_layer
                )
                record_by_key(fractional_mold_sample["outputs"], measurement)[
                    "samples"
                ][0]["value"] = 0.25
                expect_failure(
                    lambda fractional_mold_sample=fractional_mold_sample, manifest_layer=manifest_layer: (
                        runner.validate_manifest_layer_evidence(
                            fractional_mold_sample, manifest_layer, "mold"
                        )
                    ),
                    "integer min/max and samples",
                )

                for statistic, value in (("min", 0.25), ("max", 2.25)):
                    fractional_bound = layer_document(
                        tool, "oracle", authorization_sha, manifest_layer
                    )
                    record_by_key(fractional_bound["outputs"], measurement)[
                        "statistics"
                    ][statistic] = value
                    expect_failure(
                        lambda fractional_bound=fractional_bound, manifest_layer=manifest_layer: (
                            runner.validate_manifest_layer_evidence(
                                fractional_bound, manifest_layer, "oracle"
                            )
                        ),
                        "integer min/max and samples",
                    )

                fractional_summary = layer_document(
                    tool, "oracle", authorization_sha, manifest_layer
                )
                summary_statistics = record_by_key(
                    fractional_summary["outputs"], measurement
                )["statistics"]
                summary_statistics["mean"] = 1.5
                summary_statistics["std"] = 0.5
                runner.validate_manifest_layer_evidence(
                    fractional_summary, manifest_layer, "oracle"
                )

                for role in ("oracle", "mold"):
                    bounded = layer_document(
                        tool, role, authorization_sha, manifest_layer
                    )
                    bounded_output = record_by_key(bounded["outputs"], measurement)
                    bounded_output["statistics"]["min"] = runner.SIGNED_INT64_MIN
                    bounded_output["statistics"]["max"] = runner.SIGNED_INT64_MAX
                    bounded_output["samples"][0]["value"] = runner.SIGNED_INT64_MIN
                    bounded_output["samples"][1]["value"] = runner.SIGNED_INT64_MAX
                    runner.validate_manifest_layer_evidence(
                        bounded, manifest_layer, role
                    )

                    for field, value in (
                        ("min", runner.SIGNED_INT64_MIN - 1),
                        ("max", runner.SIGNED_INT64_MAX + 1),
                    ):
                        overflow = layer_document(
                            tool, role, authorization_sha, manifest_layer
                        )
                        record_by_key(overflow["outputs"], measurement)["statistics"][
                            field
                        ] = value
                        expect_failure(
                            lambda overflow=overflow, manifest_layer=manifest_layer, role=role: (
                                runner.validate_manifest_layer_evidence(
                                    overflow, manifest_layer, role
                                )
                            ),
                            "signed int64",
                        )

                    sample_overflow = layer_document(
                        tool, role, authorization_sha, manifest_layer
                    )
                    record_by_key(sample_overflow["outputs"], measurement)["samples"][
                        0
                    ]["value"] = runner.SIGNED_INT64_MAX + 1
                    expect_failure(
                        lambda sample_overflow=sample_overflow, manifest_layer=manifest_layer, role=role: (
                            runner.validate_manifest_layer_evidence(
                                sample_overflow, manifest_layer, role
                            )
                        ),
                        "signed int64",
                    )

            wrong_policy = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            policy = record_by_key(wrong_policy["comparison"], measurement)
            piecewise_qwen = layer == "qwen-layer-50" and measurement in {
                "statistics",
                "sampled-values",
            }
            if kind == "integer":
                policy["absolute"] = 0.25
                expected_policy_error = "integer policy"
            else:
                policy["absolute"] = (
                    runner.REVIEWED_ABSOLUTE_TOLERANCE_CAPS.get(
                        (layer, measurement), runner.MAX_EXACT_ABSOLUTE_TOLERANCE
                    )
                    + 1.0
                )
                expected_policy_error = (
                    "reviewed Qwen magnitude policy"
                    if piecewise_qwen
                    else "bounded protected policy"
                )
            expect_failure(
                lambda wrong_policy=wrong_policy, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        wrong_policy, manifest_layer, "oracle"
                    )
                ),
                expected_policy_error,
            )

            if kind != "integer":
                for policy_key in ("absolute", "relative"):
                    overflow_policy = layer_document(
                        tool, "oracle", authorization_sha, manifest_layer
                    )
                    record_by_key(overflow_policy["comparison"], measurement)[
                        policy_key
                    ] = 10**400
                    expect_failure(
                        lambda overflow_policy=overflow_policy, manifest_layer=manifest_layer: (
                            runner.validate_manifest_layer_evidence(
                                overflow_policy, manifest_layer, "oracle"
                            )
                        ),
                        expected_policy_error,
                    )

            wrong_hash_policy = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            hash_policy = record_by_key(wrong_hash_policy["comparison"], measurement)
            hash_policy["hash_policy"] = (
                "record-only" if kind == "integer" else "not-reviewed"
            )
            expect_failure(
                lambda wrong_hash_policy=wrong_hash_policy, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        wrong_hash_policy, manifest_layer, "oracle"
                    )
                ),
                (
                    "integer policy"
                    if kind == "integer"
                    else expected_policy_error
                ),
            )

        oracle = layer_document(tool, "oracle", authorization_sha, manifest_layer)
        mold = layer_document(tool, "mold", authorization_sha, manifest_layer)
        first_policy = oracle["comparison"][0]
        fixture = {
            "tolerance": {
                "absolute": first_policy["absolute"],
                "relative": first_policy["relative"],
                "metric": first_policy["metric"],
            }
        }
        runner.validate_oracle_mold_policy_parity(
            oracle, fixture, mold, fixture, manifest_layer
        )
        mismatched_mold = copy.deepcopy(mold)
        mismatched_mold["outputs"][0]["dtype"] = "uint8"
        expect_failure(
            lambda: runner.validate_oracle_mold_policy_parity(
                oracle, fixture, mismatched_mold, fixture, manifest_layer
            ),
            "dtypes differ",
        )

        mismatched_hash = copy.deepcopy(mold)
        mismatched_hash["outputs"][0]["content_sha256"] = "f" * 64
        if first_policy["hash_policy"] == "exact":
            expect_failure(
                lambda: runner.validate_oracle_mold_policy_parity(
                    oracle, fixture, mismatched_hash, fixture, manifest_layer
                ),
                "content hashes differ",
            )
        else:
            runner.validate_oracle_mold_policy_parity(
                oracle, fixture, mismatched_hash, fixture, manifest_layer
            )
            exact_key = next(
                (
                    policy["key"]
                    for policy in oracle["comparison"]
                    if policy["hash_policy"] == "exact"
                ),
                None,
            )
            if exact_key is not None:
                record_by_key(mismatched_hash["outputs"], exact_key)[
                    "content_sha256"
                ] = "e" * 64
                expect_failure(
                    lambda: runner.validate_oracle_mold_policy_parity(
                        oracle, fixture, mismatched_hash, fixture, manifest_layer
                    ),
                    "content hashes differ",
                )

        if layer in E2E_LAYER_TASKS:
            mismatched_input = copy.deepcopy(mold)
            mismatched_input["input"]["conformance"]["seed"] = 43
            rehash_e2e_input(mismatched_input)
            runner.validate_manifest_layer_evidence(
                mismatched_input, manifest_layer, "mold"
            )
            expect_failure(
                lambda: runner.validate_oracle_mold_policy_parity(
                    oracle, fixture, mismatched_input, fixture, manifest_layer
                ),
                "input hashes differ",
            )

        for provenance_key in manifest_layer["role_invariant_provenance"]:
            mismatched_provenance = copy.deepcopy(mold)
            record_by_key(mismatched_provenance["provenance"], provenance_key)[
                "value"
            ] += "-mold-drift"
            expect_failure(
                lambda mismatched_provenance=mismatched_provenance: (
                    runner.validate_oracle_mold_policy_parity(
                        oracle,
                        fixture,
                        mismatched_provenance,
                        fixture,
                        manifest_layer,
                    )
                ),
                f"provenance {provenance_key!r} differs",
            )


def test_runner_contract(runner, tool, temporary: pathlib.Path) -> None:
    fixture_root = temporary / "fixtures"
    fixture_root.mkdir()
    authorization, reviewed_source, authorization_sha = authorization_fixture(
        tool, temporary
    )
    test_manifest_layer_contract(runner, tool, authorization_sha)
    original_sha256_file = tool.sha256_file
    original_load_tool = runner.load_tool

    def controlled_sha256_file(path: pathlib.Path) -> str:
        if pathlib.Path(path).resolve() == reviewed_source.resolve():
            return REVIEWED_AUTHORIZATION_EVIDENCE_SHA256
        return original_sha256_file(path)

    tool.sha256_file = controlled_sha256_file
    runner.load_tool = lambda: tool

    oversized = temporary / "oversized-layer.json"
    oversized.write_bytes(b"{" + b" " * 16 + b"}")
    expect_failure(
        lambda: runner.read_json_once(oversized, "oversized layer", 16),
        "bounded regular-file contract",
    )

    def reset_campaign() -> tuple[dict[str, str], pathlib.Path, pathlib.Path]:
        oracle_bundle, mold_bundle = bundle_fixture(
            tool, fixture_root, authorization_sha
        )
        return (
            campaign_environment(
                fixture_root, authorization, oracle_bundle, mold_bundle
            ),
            oracle_bundle,
            mold_bundle,
        )

    try:
        environment, oracle_bundle, mold_bundle = reset_campaign()
        assert set(exact_layer_ids(tool)) == {
            fixture["layer"]
            for fixture in json.loads(oracle_bundle.read_text(encoding="utf-8"))[
                "fixtures"
            ]
        }

        expect_failure(
            lambda: runner.resolve_external_file(
                "authorization record", str(TOOL_PATH)
            ),
            "must live outside the Mold repository",
        )
        expect_failure(
            lambda: runner.run_campaign({}, lambda: None),
            "MOLD_H3_FIXTURE_ROOT is required",
        )
        missing_authorization = dict(environment)
        del missing_authorization["MOLD_H3_AUTHORIZATION_RECORD"]
        expect_failure(
            lambda: runner.run_campaign(missing_authorization, lambda: None),
            "MOLD_H3_AUTHORIZATION_RECORD is required",
        )

        probes = 0

        def counted_probe() -> None:
            nonlocal probes
            probes += 1

        result = runner.run_campaign(environment, counted_probe, lambda: SOURCE_SHA)
        assert result == {
            "comparisons": 11,
            "notes": 0,
            "source_sha": SOURCE_SHA,
        }
        assert probes == 1

        environment, oracle_bundle, _ = reset_campaign()
        oracle_runtime = json.loads(oracle_bundle.read_text(encoding="utf-8"))
        oracle_runtime["capture_environment"]["oracle_runtime"]["torch"] = "0.0.0"
        write_json(oracle_bundle, oracle_runtime)
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "oracle bundle runtime identity is not exact",
        )

        environment, oracle_bundle, _ = reset_campaign()
        oracle_adapter = json.loads(oracle_bundle.read_text(encoding="utf-8"))
        tokenizer_adapter = next(
            record
            for record in oracle_adapter["capture_environment"]["oracle_adapters"]
            if record["layer"] == "tokenizer-processor"
        )
        tokenizer_adapter["implementation_sha256"] = "0" * 64
        write_json(oracle_bundle, oracle_adapter)
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "adapter record for layer tokenizer-processor is not exact",
        )

        excluded_accelerations = runner.manifest_excluded_accelerations(
            tool.validate_manifest()
        )
        layer_contracts = runner.exact_layer_contracts(tool.validate_manifest())
        _, valid_oracle_path, _ = reset_campaign()
        valid_oracle = json.loads(valid_oracle_path.read_text(encoding="utf-8"))

        missing_adapter = copy.deepcopy(valid_oracle)
        del missing_adapter["capture_environment"]["oracle_adapters"]
        expect_failure(
            lambda: runner.validate_capture_environment(
                tool,
                missing_adapter,
                "oracle",
                SOURCE_SHA,
                excluded_accelerations,
                layer_contracts,
            ),
            "missing manifest-required adapter records",
        )

        extra_adapter = copy.deepcopy(valid_oracle)
        extra_record = copy.deepcopy(
            extra_adapter["capture_environment"]["oracle_adapters"][0]
        )
        extra_record["layer"] = "zz-extra-layer"
        extra_adapter["capture_environment"]["oracle_adapters"].append(extra_record)
        expect_failure(
            lambda: runner.validate_capture_environment(
                tool,
                extra_adapter,
                "oracle",
                SOURCE_SHA,
                excluded_accelerations,
                layer_contracts,
            ),
            "adapter layers differ",
        )

        duplicate_adapter = copy.deepcopy(valid_oracle)
        duplicate_adapter["capture_environment"]["oracle_adapters"].append(
            copy.deepcopy(
                duplicate_adapter["capture_environment"]["oracle_adapters"][0]
            )
        )
        expect_failure(
            lambda: runner.validate_capture_environment(
                tool,
                duplicate_adapter,
                "oracle",
                SOURCE_SHA,
                excluded_accelerations,
                layer_contracts,
            ),
            "repeats adapter record",
        )

        mixed_adapter = copy.deepcopy(valid_oracle)
        mixed_tokenizer_adapter = next(
            record
            for record in mixed_adapter["capture_environment"]["oracle_adapters"]
            if record["layer"] == "tokenizer-processor"
        )
        mixed_tokenizer_adapter["id"] = "different-layer-adapter"
        expect_failure(
            lambda: runner.validate_capture_environment(
                tool,
                mixed_adapter,
                "oracle",
                SOURCE_SHA,
                excluded_accelerations,
                layer_contracts,
            ),
            "adapter record for layer tokenizer-processor is not exact",
        )

        environment, _, mold_bundle = reset_campaign()

        def drift_processor_grid(document: dict[str, object]) -> None:
            grid = record_by_key(document["outputs"], "processor-grids")
            grid["content_sha256"] = "f" * 64

        mutate_evidence(
            fixture_root,
            mold_bundle,
            drift_processor_grid,
            fixture_layer="tokenizer-processor",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "output 'processor-grids' content hashes differ",
        )

        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: "b" * 40),
            "checkout source SHA",
        )
        expect_failure(
            lambda: runner.run_campaign(
                environment,
                lambda: (_ for _ in ()).throw(
                    runner.GpuConformanceFailure("CUDA probe failed")
                ),
                lambda: SOURCE_SHA,
            ),
            "CUDA probe failed",
        )

        drifted_source = dict(environment)
        drifted_source["MOLD_H3_SOURCE_SHA"] = "b" * 40
        expect_failure(
            lambda: runner.run_campaign(drifted_source, lambda: None, lambda: "b" * 40),
            "Mold bundle framework revision",
        )

        environment, _, mold_bundle = reset_campaign()
        incomplete = json.loads(mold_bundle.read_text(encoding="utf-8"))
        incomplete["fixtures"].pop()
        write_json(mold_bundle, incomplete)
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "complete exact layer coverage",
        )

        environment, _, mold_bundle = reset_campaign()

        def relabel_synthetic_layer(document: dict[str, object]) -> None:
            document["layer"] = "dual-sampler"

        mutate_evidence(
            fixture_root,
            mold_bundle,
            relabel_synthetic_layer,
            lambda fixture: fixture.update({"layer": "dual-sampler"}),
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "manifest authority tier",
        )

        environment, _, mold_bundle = reset_campaign()
        mutate_evidence(
            fixture_root,
            mold_bundle,
            lambda document: document.update(
                {"authority_tier": "quantized-structural"}
            ),
            lambda fixture: fixture.update({"authority_tier": "quantized-structural"}),
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "authority tier drifts from the manifest",
        )

        environment, _, mold_bundle = reset_campaign()
        capture_dtype = json.loads(mold_bundle.read_text(encoding="utf-8"))
        capture_dtype["capture_environment"]["dtype"] = "float32"
        write_json(mold_bundle, capture_dtype)
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "capture environment dtype",
        )

        environment, _, mold_bundle = reset_campaign()
        excluded_accelerations = runner.manifest_excluded_accelerations(
            tool.validate_manifest()
        )
        valid_capture_bundle = json.loads(mold_bundle.read_text(encoding="utf-8"))
        missing_capture_policy = copy.deepcopy(valid_capture_bundle)
        del missing_capture_policy["capture_environment"]["acceleration_policy"]
        expect_failure(
            lambda: runner.validate_capture_environment(
                tool,
                missing_capture_policy,
                "mold",
                SOURCE_SHA,
                excluded_accelerations,
            ),
            "lacks structured acceleration evidence",
        )
        for excluded_acceleration in sorted(excluded_accelerations):
            missing_exclusion = copy.deepcopy(valid_capture_bundle)
            missing_exclusion["capture_environment"]["acceleration_policy"][
                "disabled"
            ].remove(excluded_acceleration)
            expect_failure(
                lambda missing_exclusion=missing_exclusion: (
                    runner.validate_capture_environment(
                        tool,
                        missing_exclusion,
                        "mold",
                        SOURCE_SHA,
                        excluded_accelerations,
                    )
                ),
                "does not disable every manifest-excluded acceleration",
            )

            enabled_exclusion = copy.deepcopy(valid_capture_bundle)
            enabled_exclusion["capture_environment"]["acceleration_policy"][
                "enabled"
            ].append(excluded_acceleration)
            expect_failure(
                lambda enabled_exclusion=enabled_exclusion: (
                    runner.validate_capture_environment(
                        tool,
                        enabled_exclusion,
                        "mold",
                        SOURCE_SHA,
                        excluded_accelerations,
                    )
                ),
                "enables a manifest-excluded acceleration",
            )

        duplicate_exclusion = copy.deepcopy(valid_capture_bundle)
        duplicate_exclusion["capture_environment"]["acceleration_policy"][
            "disabled"
        ].append(sorted(excluded_accelerations)[0])
        expect_failure(
            lambda: runner.validate_capture_environment(
                tool,
                duplicate_exclusion,
                "mold",
                SOURCE_SHA,
                excluded_accelerations,
            ),
            "structured acceleration evidence is not canonical",
        )

        uppercase_exclusion = copy.deepcopy(valid_capture_bundle)
        uppercase_exclusion["capture_environment"]["acceleration_policy"]["disabled"][
            0
        ] = uppercase_exclusion["capture_environment"]["acceleration_policy"][
            "disabled"
        ][0].upper()
        expect_failure(
            lambda: runner.validate_capture_environment(
                tool,
                uppercase_exclusion,
                "mold",
                SOURCE_SHA,
                excluded_accelerations,
            ),
            "structured acceleration evidence is not canonical",
        )

        excluded_backend = sorted(excluded_accelerations)[0]
        capture_backend = json.loads(mold_bundle.read_text(encoding="utf-8"))
        capture_backend["capture_environment"]["attention_backend"] = excluded_backend
        write_json(mold_bundle, capture_backend)
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "manifest-excluded acceleration",
        )

        for excluded_alias in ("float8-e4m3fn", "euler-ancestral"):
            environment, _, mold_bundle = reset_campaign()
            capture_alias = json.loads(mold_bundle.read_text(encoding="utf-8"))
            capture_alias["capture_environment"]["acceleration_policy"][
                "enabled"
            ].append(excluded_alias)
            write_json(mold_bundle, capture_alias)
            expect_failure(
                lambda: runner.run_campaign(
                    environment, lambda: None, lambda: SOURCE_SHA
                ),
                "enables a manifest-excluded acceleration",
            )

        environment, _, mold_bundle = reset_campaign()

        def drift_layer_acceleration_summary(document: dict[str, object]) -> None:
            document["environment"]["acceleration_policy"]["enabled"].append(
                "dense-math"
            )

        mutate_evidence(
            fixture_root,
            mold_bundle,
            drift_layer_acceleration_summary,
            fixture_layer="qwen-layer-50",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "layer acceleration policy differs from its bundle",
        )

        environment, _, mold_bundle = reset_campaign()
        mutate_evidence(
            fixture_root,
            mold_bundle,
            lambda document: document["environment"].update({"dtype": "float32"}),
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "layer environment dtype",
        )

        environment, _, mold_bundle = reset_campaign()

        def use_excluded_backend(document: dict[str, object]) -> None:
            document["environment"]["attention_backend"] = excluded_backend
            record_by_key(document["provenance"], "attention-backend")["value"] = (
                excluded_backend
            )

        mutate_evidence(
            fixture_root,
            mold_bundle,
            use_excluded_backend,
            fixture_layer="transformer-block",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "manifest-excluded acceleration",
        )

        environment, _, mold_bundle = reset_campaign()

        def use_excluded_backend_alias(document: dict[str, object]) -> None:
            document["environment"]["attention_backend"] = "sage-attn"
            record_by_key(document["provenance"], "attention-backend")["value"] = (
                "sage-attn"
            )

        mutate_evidence(
            fixture_root,
            mold_bundle,
            use_excluded_backend_alias,
            fixture_layer="transformer-block",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "manifest-excluded acceleration",
        )

        environment, _, mold_bundle = reset_campaign()
        mutate_evidence(
            fixture_root,
            mold_bundle,
            lambda document: document["outputs"][0].update({"dtype": "uint8"}),
            lambda fixture: fixture["tensor"].update({"dtype": "uint8"}),
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "dtype must be",
        )

        for output_key, location, invalid_value, expected_error in (
            (
                "sampled-values",
                "sample",
                10**100,
                "must be exact bfloat16",
            ),
            (
                "statistics",
                "sample",
                10**400,
                "must be finite float64",
            ),
            (
                "statistics",
                "sample",
                2**80 + 1,
                "must be finite float64",
            ),
            (
                "sampled-values",
                "mean",
                10**400,
                "summary is outside the finite float64 domain",
            ),
        ):
            environment, _, mold_bundle = reset_campaign()

            def violate_numeric_domain(
                document: dict[str, object],
                output_key: str = output_key,
                location: str = location,
                invalid_value: int = invalid_value,
            ) -> None:
                output = record_by_key(document["outputs"], output_key)
                if location == "sample":
                    output["samples"][0]["value"] = invalid_value
                else:
                    output["statistics"][location] = invalid_value
                    output["statistics"]["max"] = invalid_value

            mutate_evidence(
                fixture_root,
                mold_bundle,
                violate_numeric_domain,
                fixture_layer="qwen-layer-50",
            )
            expect_failure(
                lambda: runner.run_campaign(
                    environment, lambda: None, lambda: SOURCE_SHA
                ),
                expected_error,
            )

        environment, _, mold_bundle = reset_campaign()
        mutate_evidence(
            fixture_root,
            mold_bundle,
            mutate_fixture=lambda fixture: fixture["tensor"].update({"mean": 0.125}),
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "bundle tensor summary",
        )

        environment, oracle_bundle, _ = reset_campaign()
        mutate_evidence(
            fixture_root,
            oracle_bundle,
            mutate_fixture=lambda fixture: fixture["tolerance"].update(
                {"absolute": 0.000003}
            ),
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "bundle tolerance summary",
        )

        environment, _, mold_bundle = reset_campaign()
        mutate_evidence(
            fixture_root,
            mold_bundle,
            mutate_fixture=lambda fixture: fixture["tolerance"].update(
                {"relative": 0.000002}
            ),
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "differs from oracle policy",
        )

        environment, oracle_bundle, _ = reset_campaign()
        mutate_evidence(
            fixture_root,
            oracle_bundle,
            lambda document: document["comparison"][0].update({"absolute": 1.0}),
            lambda fixture: fixture["tolerance"].update({"absolute": 1.0}),
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "policy",
        )

        environment, oracle_bundle, _ = reset_campaign()

        def overflow_floating_policy(document: dict[str, object]) -> None:
            record_by_key(document["comparison"], "sampled-values")["absolute"] = (
                10**400
            )

        mutate_evidence(
            fixture_root,
            oracle_bundle,
            overflow_floating_policy,
            fixture_layer="qwen-layer-50",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "reviewed Qwen magnitude policy",
        )

        environment, oracle_bundle, _ = reset_campaign()

        def widen_qwen_magnitude_policy(document: dict[str, object]) -> None:
            record_by_key(document["comparison"], "sampled-values")[
                "large_relative"
            ] = 0.5

        mutate_evidence(
            fixture_root,
            oracle_bundle,
            widen_qwen_magnitude_policy,
            fixture_layer="qwen-layer-50",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "reviewed Qwen magnitude policy",
        )

        environment, _, mold_bundle = reset_campaign()

        def change_floating_content_hash(document: dict[str, object]) -> None:
            record_by_key(document["outputs"], "sampled-values")["content_sha256"] = (
                "f" * 64
            )

        mutate_evidence(
            fixture_root,
            mold_bundle,
            change_floating_content_hash,
            fixture_layer="qwen-layer-50",
        )
        result = runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA)
        assert result["notes"] >= 1

        environment, _, mold_bundle = reset_campaign()

        def change_seed(document: dict[str, object]) -> None:
            record_by_key(document["provenance"], "seed")["value"] = "43"

        mutate_evidence(
            fixture_root,
            mold_bundle,
            change_seed,
            fixture_layer="noise-allocation",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "provenance 'seed' differs",
        )

        for provenance_key in ("tokenizer-hash", "processor-hash"):
            environment, oracle_bundle, mold_bundle = reset_campaign()

            def forge_pinned_provenance(
                document: dict[str, object], provenance_key: str = provenance_key
            ) -> None:
                record_by_key(document["provenance"], provenance_key)["value"] = (
                    "f" * 64
                )

            mutate_evidence(
                fixture_root,
                oracle_bundle,
                forge_pinned_provenance,
                fixture_layer="tokenizer-processor",
            )
            mutate_evidence(
                fixture_root,
                mold_bundle,
                forge_pinned_provenance,
                fixture_layer="tokenizer-processor",
            )
            expect_failure(
                lambda: runner.run_campaign(
                    environment, lambda: None, lambda: SOURCE_SHA
                ),
                f"provenance {provenance_key!r} is not authority-pinned",
            )

        environment, _, mold_bundle = reset_campaign()

        def change_e2e_seed_without_hash(document: dict[str, object]) -> None:
            document["input"]["conformance"]["seed"] = 43

        mutate_evidence(
            fixture_root,
            mold_bundle,
            change_e2e_seed_without_hash,
            fixture_layer="end-to-end-t2va",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "input hash does not match canonical evidence",
        )

        environment, _, mold_bundle = reset_campaign()

        def change_e2e_seed_and_hash(document: dict[str, object]) -> None:
            document["input"]["conformance"]["seed"] = 43
            rehash_e2e_input(document)

        mutate_evidence(
            fixture_root,
            mold_bundle,
            change_e2e_seed_and_hash,
            fixture_layer="end-to-end-t2va",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "input hashes differ",
        )

        environment, _, mold_bundle = reset_campaign()
        manifest = tool.validate_manifest()
        component_authorities = {
            component["id"]: component["sha256"]
            for component in manifest["component_indexes"]
        }
        required_qwen_components = next(
            layer["required_component_indexes"]
            for layer in manifest["fixture_layers"]
            if layer["id"] == "qwen-layer-50"
        )
        unrelated_component_id = next(
            identifier
            for identifier in component_authorities
            if identifier not in required_qwen_components
        )
        unrelated_component = {
            "id": unrelated_component_id,
            "sha256": component_authorities[unrelated_component_id],
        }

        def use_unrelated_component(document: dict[str, object]) -> None:
            document["input"]["component_indexes"] = [unrelated_component]
            document["input"]["component_index_sha256"] = unrelated_component["sha256"]
            record_by_key(document["provenance"], "component-index-hashes")["value"] = (
                f"{unrelated_component['id']}={unrelated_component['sha256']}"
            )

        def bundle_unrelated_component(fixture: dict[str, object]) -> None:
            fixture["component_indexes"] = [unrelated_component]
            fixture["component_index_sha256"] = unrelated_component["sha256"]

        mutate_evidence(
            fixture_root,
            mold_bundle,
            use_unrelated_component,
            bundle_unrelated_component,
            fixture_layer="qwen-layer-50",
        )
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "component authorities differ from the manifest",
        )

        environment, _, mold_bundle = reset_campaign()
        mismatched_component = json.loads(mold_bundle.read_text(encoding="utf-8"))
        mismatched_component["fixtures"][0]["component_index_sha256"] = "0" * 64
        write_json(mold_bundle, mismatched_component)
        expect_failure(
            lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
            "component index does not match",
        )

        environment, _, _ = reset_campaign()
        wrong_record, _, wrong_sha = authorization_fixture(
            tool, temporary, name="self-consistent-wrong", reviewed=False
        )
        assert wrong_sha != REVIEWED_AUTHORIZATION_EVIDENCE_SHA256
        wrong_environment = dict(environment)
        wrong_environment["MOLD_H3_AUTHORIZATION_RECORD"] = str(wrong_record)
        expect_failure(
            lambda: runner.run_campaign(
                wrong_environment, lambda: None, lambda: SOURCE_SHA
            ),
            "reviewed authorization evidence",
        )

        environment, _, mold_bundle = reset_campaign()
        original_loader = runner.load_layer_documents
        mutated_after_load = False

        def mutating_loader(*args, **kwargs):
            nonlocal mutated_after_load
            loaded = original_loader(*args, **kwargs)
            role = args[4]
            if role == "mold" and not mutated_after_load:
                bundle = args[3]
                evidence_path = fixture_root / bundle["fixtures"][0]["relative_path"]
                evidence_path.write_text("{}\n", encoding="utf-8")
                mutated_after_load = True
            return loaded

        runner.load_layer_documents = mutating_loader
        try:
            result = runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA)
        finally:
            runner.load_layer_documents = original_loader
        assert mutated_after_load
        assert result["comparisons"] == 11
    finally:
        tool.sha256_file = original_sha256_file
        runner.load_tool = original_load_tool


def test_workflow_contract() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    actionlint = (REPO_ROOT / ".github" / "actionlint.yaml").read_text(encoding="utf-8")
    qualification_docs = (
        REPO_ROOT / "docs" / "qualification" / "minimax-h3-conformance.md"
    ).read_text(encoding="utf-8")
    normalized_docs = " ".join(qualification_docs.split())
    assert "workflow_dispatch:" in workflow
    for trigger in ("push:", "pull_request:", "schedule:", "workflow_run:"):
        assert f"  {trigger}" not in workflow
    assert "environment: minimax-h3-private-uat" in workflow
    assert "runs-on:\n      group: minimax-h3-private-conformance" in workflow
    assert "labels: [self-hosted, linux, x64, cuda, minimax-h3-private-uat]" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "expected_source_sha:" in workflow
    assert "\"$GITHUB_REF\" != 'refs/heads/main'" in workflow
    assert '"$GITHUB_SHA" != "$EXPECTED_SOURCE_SHA"' in workflow
    assert f"uses: actions/checkout@{CHECKOUT_V6_SHA} # v6" in workflow
    checkout_position = workflow.index(f"actions/checkout@{CHECKOUT_V6_SHA}")
    validator_position = workflow.index(
        "- name: Compare authorization-bound external captures"
    )
    assert checkout_position < validator_position
    for secret in (
        "MOLD_H3_FIXTURE_ROOT",
        "MOLD_H3_AUTHORIZATION_RECORD",
        "MOLD_H3_ORACLE_BUNDLE",
        "MOLD_H3_MOLD_BUNDLE",
    ):
        assert workflow.count(f"secrets.{secret}") == 1
        assert workflow.index(f"secrets.{secret}") > validator_position
    assert "MOLD_H3_SOURCE_SHA: ${{ github.sha }}" in workflow
    assert "python3 scripts/run-minimax-h3-gpu-conformance.py" in workflow
    assert "upload-artifact" not in workflow
    assert "h3-private-uat" not in workflow.replace("minimax-h3-private-uat", "")
    assert "    - cuda" in actionlint
    assert "    - minimax-h3-private-uat" in actionlint
    assert "unchecked administrator prerequisites" in normalized_docs
    assert "can create one without the intended protection" in normalized_docs
    assert "routing metadata, not an access-control boundary" in normalized_docs
    assert (
        "utensils/mold/.github/workflows/minimax-h3-private-conformance.yml@refs/heads/main"
        in normalized_docs
    )
    assert "runner-group restrictions exist" in normalized_docs
    assert "persistent public-repository runner" in normalized_docs
    assert "if the environment, approval, or secrets are absent" not in normalized_docs
    assert "Computed metric and statistical summaries are serialized as `float64`" in (
        normalized_docs
    )
    assert "Discrete tokenizer, shape, layout" in normalized_docs
    assert "structured capture attestation" in normalized_docs
    assert "mold.minimax-h3.component-authority-set.v1" in normalized_docs
    assert "mold.minimax-h3.e2e-input.v1" in normalized_docs
    assert "No fixed prompt, media set, seed, dimensions" in normalized_docs
    assert REVIEWED_AUTHORIZATION_EVIDENCE_SHA256 in qualification_docs

    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for path in (
        ".github/actionlint.yaml",
        ".github/workflows/minimax-h3-private-conformance.yml",
        "docs/qualification/minimax-h3-*.json",
        "scripts/run-minimax-h3-gpu-conformance.py",
        "scripts/tests/minimax-h3-gpu-conformance-contract.py",
    ):
        assert f"'{path}'" in ci
    assert "python3 scripts/tests/minimax-h3-gpu-conformance-contract.py" in ci


def main() -> int:
    runner = load_module(RUNNER_PATH, "minimax_h3_gpu_conformance")
    tool = load_module(TOOL_PATH, "minimax_h3_conformance")
    with tempfile.TemporaryDirectory(prefix="mold-h3-gpu-contract-") as value:
        test_runner_contract(runner, tool, pathlib.Path(value).resolve())
    test_workflow_contract()
    print("MiniMax H3 private GPU conformance contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
