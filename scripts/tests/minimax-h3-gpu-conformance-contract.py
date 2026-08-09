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
CANONICAL_INTEGER_DTYPE = "int64"
CANONICAL_METRIC_DTYPE = "float64"
TEST_MEASUREMENT_KINDS = {
    "tokenizer-processor": {
        "token-ids": "integer",
        "special-token-presentation": "integer",
        "processor-shapes": "integer",
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
        "pixel-normalization": "activation",
        "posterior-seed-42": "activation",
        "latent-statistics": "metric",
        "sampled-values": "activation",
        "tile-seams": "metric",
    },
    "audio-vae": {
        "stereo-packing": "integer",
        "latent-rate": "metric",
        "waveform-statistics": "metric",
        "phase-polarity": "integer",
        "sampled-values": "activation",
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
        "video-head": "activation",
        "audio-head": "activation",
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
    layers = {
        layer["id"]: layer
        for layer in manifest["fixture_layers"]
        if layer["authority_tier"] == "exact-full-bf16"
    }
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


def comparison_fixture(key: str, kind: str) -> dict[str, object]:
    integer = kind == "integer"
    return {
        "key": key,
        "absolute": 0 if integer else 0.000002,
        "relative": 0 if integer else 0.000001,
        "metric": "elementwise-atol-plus-rtol",
        "hash_policy": "exact" if integer else "record-only",
    }


def provenance_fixture(
    manifest_layer: dict[str, object], document: dict[str, object]
) -> list[dict[str, str]]:
    values = {
        "source-revision": document["producer"]["revision"],
        "tokenizer-hash": hashlib.sha256(b"contract-tokenizer").hexdigest(),
        "processor-hash": hashlib.sha256(b"contract-processor").hexdigest(),
        "component-index-hash": document["input"]["component_index_sha256"],
        "component-index-hashes": document["input"]["component_index_sha256"],
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
    component_sha = next(iter(tool.EXPECTED_COMPONENT_INDEXES.values()))[1]
    outputs = [
        output_fixture(layer, key, TEST_MEASUREMENT_KINDS[layer][key])
        for key in manifest_layer["required_measurements"]
    ]
    document: dict[str, object] = {
        "schema_version": tool.LAYER_OUTPUT_SCHEMA_VERSION,
        "family": "minimax-h3",
        "case_id": "gpu-contract-case",
        "layer": layer,
        "authority_tier": "exact-full-bf16",
        "authorization_document_sha256": authorization_sha,
        "input": {
            "id": f"gpu-contract-{layer}",
            "sha256": hashlib.sha256(f"input:{layer}".encode()).hexdigest(),
            "component_index_sha256": component_sha,
        },
        "producer": {
            "role": role,
            "implementation": f"contract-{role}",
            "source_id": "diffusers" if role == "oracle" else "mold",
            "revision": (
                tool.EXPECTED_REVISIONS["diffusers"] if role == "oracle" else SOURCE_SHA
            ),
        },
        "adapter": {
            "schema_version": "mold.minimax-h3.layer-adapter.v1",
            "id": f"contract-{role}-adapter",
            "command": f"contract-test {role} {layer} capture",
            "tensor_hash_encoding": "canonical-typed-le-v1",
        },
        "environment": {
            "device": "cuda:contract-test",
            "dtype": CANONICAL_BF16_DTYPE,
            "attention_backend": "math",
            "forbidden_accelerations_disabled": True,
        },
        "outputs": outputs,
    }
    document["provenance"] = provenance_fixture(manifest_layer, document)
    if role == "oracle":
        document["comparison"] = [
            comparison_fixture(key, TEST_MEASUREMENT_KINDS[layer][key])
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
    bundle_paths: list[pathlib.Path] = []
    for role in ("oracle", "mold"):
        fixtures = []
        for layer, manifest_layer in manifest_layers.items():
            document = layer_document(tool, role, authorization_sha, manifest_layer)
            evidence_path = fixture_root / role / f"{layer}.json"
            write_json(evidence_path, document)
            output = document["outputs"][0]
            comparison = comparison_fixture(
                output["key"], TEST_MEASUREMENT_KINDS[layer][output["key"]]
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
                "dtype": CANONICAL_BF16_DTYPE,
                "attention_backend": "math",
                "command": f"contract-test {role} capture",
                "forbidden_accelerations_disabled": True,
            },
            "fixtures": fixtures,
        }
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
) -> tuple[dict[str, object], pathlib.Path]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    fixture = bundle["fixtures"][0]
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


def test_manifest_layer_contract(runner, tool, authorization_sha: str) -> None:
    manifest = tool.validate_manifest()
    manifest_layers = exact_manifest_layers(tool)
    assert set(runner.exact_layer_contracts(manifest)) == set(manifest_layers)

    for layer, manifest_layer in manifest_layers.items():
        for key, kind in TEST_MEASUREMENT_KINDS[layer].items():
            assert runner.MEASUREMENT_DTYPES[layer][key] == DTYPE_BY_KIND[kind]

        for role in ("oracle", "mold"):
            document = layer_document(tool, role, authorization_sha, manifest_layer)
            tool.validate_layer_output(document, f"valid {role} {layer}")
            runner.validate_manifest_layer_evidence(document, manifest_layer, role)

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
        }
        for provenance_key, value in structured_drifts.items():
            if provenance_key not in manifest_layer["required_provenance"]:
                continue
            drifted = layer_document(tool, "oracle", authorization_sha, manifest_layer)
            record_by_key(drifted["provenance"], provenance_key)["value"] = value
            expect_failure(
                lambda drifted=drifted, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        drifted, manifest_layer, "oracle"
                    )
                ),
                "does not match structured evidence",
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
            output["dtype"] = (
                CANONICAL_METRIC_DTYPE if kind != "metric" else CANONICAL_BF16_DTYPE
            )
            expect_failure(
                lambda wrong_dtype=wrong_dtype, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        wrong_dtype, manifest_layer, "oracle"
                    )
                ),
                "dtype must be",
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

            wrong_policy = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            policy = record_by_key(wrong_policy["comparison"], measurement)
            if kind == "integer":
                policy["absolute"] = 0.25
                expected_policy_error = "integer policy"
            else:
                policy["absolute"] = 1.0
                expected_policy_error = "bounded protected policy"
            expect_failure(
                lambda wrong_policy=wrong_policy, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        wrong_policy, manifest_layer, "oracle"
                    )
                ),
                expected_policy_error,
            )

            wrong_hash_policy = layer_document(
                tool, "oracle", authorization_sha, manifest_layer
            )
            hash_policy = record_by_key(wrong_hash_policy["comparison"], measurement)
            hash_policy["hash_policy"] = "record-only" if kind == "integer" else "exact"
            expect_failure(
                lambda wrong_hash_policy=wrong_hash_policy, manifest_layer=manifest_layer: (
                    runner.validate_manifest_layer_evidence(
                        wrong_hash_policy, manifest_layer, "oracle"
                    )
                ),
                ("integer policy" if kind == "integer" else "bounded protected policy"),
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
        runner.validate_oracle_mold_policy_parity(oracle, fixture, mold, fixture)
        mismatched_mold = copy.deepcopy(mold)
        mismatched_mold["outputs"][0]["dtype"] = "float32"
        expect_failure(
            lambda: runner.validate_oracle_mold_policy_parity(
                oracle, fixture, mismatched_mold, fixture
            ),
            "dtypes differ",
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
