#!/usr/bin/env python3
"""Mutation tests for the weight-free MiniMax H3 conformance boundary."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("minimax_h3_conformance", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_failure(action: Callable[[], object], fragment: str) -> None:
    try:
        action()
    except Exception as error:  # noqa: BLE001 - this is an adversarial contract test
        assert fragment in str(error), (fragment, str(error))
    else:
        raise AssertionError(f"expected failure containing {fragment!r}")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_manifest_drift(tool, temporary: pathlib.Path) -> None:
    original_path = tool.MANIFEST_PATH
    manifest = json.loads(original_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["revision"] = "0" * 40
    changed = temporary / "changed-manifest.json"
    write_json(changed, manifest)
    tool.MANIFEST_PATH = changed
    try:
        expect_failure(tool.validate_manifest, "source revisions drifted")
    finally:
        tool.MANIFEST_PATH = original_path


def test_synthetic_drift(tool, temporary: pathlib.Path) -> None:
    original_path = tool.SYNTHETIC_PATH
    fixture = json.loads(original_path.read_text(encoding="utf-8"))
    fixture["scheduler"]["video_shift"] = 11.0
    changed = temporary / "changed-synthetic.json"
    write_json(changed, fixture)
    tool.SYNTHETIC_PATH = changed
    try:
        expect_failure(tool.validate_manifest, "synthetic fixture does not match")
    finally:
        tool.SYNTHETIC_PATH = original_path


def checked_comparison_pair(tool) -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads(tool.SYNTHETIC_ORACLE_PATH.read_text(encoding="utf-8")),
        json.loads(tool.SYNTHETIC_MOLD_PATH.read_text(encoding="utf-8")),
    )


def output_with_key(document: dict[str, object], key: str) -> dict[str, object]:
    outputs = document["outputs"]
    assert isinstance(outputs, list)
    return next(output for output in outputs if output["key"] == key)


def test_synthetic_comparison_uses_fixture_math(tool) -> None:
    fixture = json.loads(tool.SYNTHETIC_PATH.read_text(encoding="utf-8"))
    oracle, _ = checked_comparison_pair(tool)
    coupled_update = fixture["coupled_update"]
    for key in ("video_next", "audio_next"):
        output = output_with_key(oracle, key)
        samples = output["samples"]
        assert isinstance(samples, list)
        actual = [sample["value"] for sample in samples]
        assert actual == coupled_update[key], (
            f"synthetic {key} evidence is not the fixture's coupled Euler output: "
            f"{actual!r} != {coupled_update[key]!r}"
        )


def test_comparison_diagnostics(tool) -> None:
    oracle, mold = checked_comparison_pair(tool)
    notes = tool.compare_layer_outputs(oracle, mold)
    assert len(notes) == 1
    assert "hash mismatch" in notes[0]
    assert "policy=record-only" in notes[0]

    shape_mold = copy.deepcopy(mold)
    output_with_key(shape_mold, "audio_next")["shape"] = [5]
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, shape_mold),
        "shape mismatch: oracle=[4], mold=[5]",
    )

    dtype_mold = copy.deepcopy(mold)
    output_with_key(dtype_mold, "audio_next")["dtype"] = "float16"
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, dtype_mold),
        "dtype mismatch: oracle='float32', mold='float16'",
    )

    keys_mold = copy.deepcopy(mold)
    renamed = output_with_key(keys_mold, "audio_next")
    renamed["key"] = "unexpected_output"
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, keys_mold),
        "missing Mold outputs=['audio_next'], extra Mold outputs=['unexpected_output']",
    )

    samples_mold = copy.deepcopy(mold)
    sampled_output = output_with_key(samples_mold, "audio_next")
    sampled_output["shape"] = [5]
    sampled_output["samples"][0]["index"] = [4]
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, samples_mold),
        "sample key mismatch: missing Mold indexes=[[0]], extra Mold indexes=[[4]]",
    )

    hash_mold = copy.deepcopy(mold)
    output_with_key(hash_mold, "video_next")["content_sha256"] = "f" * 64
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, hash_mold),
        "hash mismatch: oracle=d775157b364dba9b441d9f37d774ca51e48bfdd8b360fe9fcc40632cc53e848a",
    )

    tolerance_mold = copy.deepcopy(mold)
    output_with_key(tolerance_mold, "audio_next")["samples"][0]["value"] = 0.25
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, tolerance_mold),
        "sample=[0] tolerance exceeded",
    )

    nan_mold = copy.deepcopy(mold)
    output_with_key(nan_mold, "audio_next")["statistics"]["mean"] = float("nan")
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, nan_mold),
        "is NaN",
    )

    infinity_mold = copy.deepcopy(mold)
    output_with_key(infinity_mold, "audio_next")["samples"][0]["value"] = float("inf")
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, infinity_mold),
        "is Inf",
    )

    missing_field = copy.deepcopy(mold)
    del missing_field["producer"]["revision"]
    expect_failure(
        lambda: tool.compare_layer_outputs(oracle, missing_field),
        "missing required keys ['revision']",
    )

    drifted_oracle = copy.deepcopy(oracle)
    drifted_oracle["producer"]["revision"] = "0" * 40
    expect_failure(
        lambda: tool.compare_layer_outputs(drifted_oracle, mold),
        "oracle source revision is not pinned",
    )


def valid_authorization(
    tool, temporary: pathlib.Path
) -> tuple[pathlib.Path, dict[str, object]]:
    source_document = temporary / "minimax-authorization.txt"
    source_document.write_text(
        "synthetic external authorization evidence\n", encoding="utf-8"
    )
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
        "source_document_path": str(source_document),
        "source_document_sha256": sha256(source_document),
        "review_reference": "external-test-review",
    }
    record_path = temporary / "authorization-record.json"
    write_json(record_path, record)
    return record_path, record


def test_authorization_and_external_root(tool, temporary: pathlib.Path) -> None:
    expect_failure(
        lambda: tool.canonical_external_directory("fixture root", str(REPO_ROOT)),
        "must live outside",
    )
    record_path, record = valid_authorization(tool, temporary)
    assert tool.validate_authorization(record_path) == record

    missing_scope = dict(record)
    missing_scope["approved_scopes"] = ["fixture-capture"]
    missing_path = temporary / "missing-scope.json"
    write_json(missing_path, missing_scope)
    expect_failure(
        lambda: tool.validate_authorization(missing_path),
        "does not cover every conformance activity",
    )

    pathlib.Path(record["source_document_path"]).write_text(
        "mutated\n", encoding="utf-8"
    )
    expect_failure(
        lambda: tool.validate_authorization(record_path),
        "hash does not match",
    )


def test_external_bundle_hashes(tool, temporary: pathlib.Path) -> None:
    root = temporary / "fixtures"
    root.mkdir()
    evidence = root / "dual-sampler.json"
    evidence.write_text('{"fixture":"synthetic"}\n', encoding="utf-8")
    record_path, record = valid_authorization(tool, temporary)
    bundle = {
        "schema_version": tool.BUNDLE_SCHEMA_VERSION,
        "manifest_sha256": sha256(tool.MANIFEST_PATH),
        "authorization_document_sha256": record["source_document_sha256"],
        "capture_environment": {
            "framework": "diffusers",
            "framework_revision": tool.EXPECTED_REVISIONS["diffusers"],
            "device": "cpu-synthetic",
            "dtype": "float32",
            "attention_backend": "math",
            "command": "weight-free synthetic fixture",
            "forbidden_accelerations_disabled": True,
        },
        "fixtures": [
            {
                "id": "dual-sampler-synthetic",
                "layer": "dual-sampler",
                "authority_tier": "synthetic",
                "relative_path": evidence.name,
                "sha256": sha256(evidence),
                "component_index_sha256": "0" * 64,
                "tensor": {
                    "shape": [4],
                    "dtype": "float32",
                    "min": 0.0,
                    "max": 1.0,
                    "mean": 0.5,
                    "std": 0.5,
                    "sampled_values": [1.0, 0.0],
                },
                "tolerance": {
                    "absolute": 0.0,
                    "relative": 0.0,
                    "metric": "exact-float32",
                },
            }
        ],
    }
    bundle_path = root / "bundle.json"
    write_json(bundle_path, bundle)
    manifest = tool.validate_manifest()
    tool.validate_bundle(manifest, str(root), str(bundle_path), str(record_path))

    evidence.write_text("mutated\n", encoding="utf-8")
    expect_failure(
        lambda: tool.validate_bundle(
            manifest, str(root), str(bundle_path), str(record_path)
        ),
        "fixture evidence hash mismatch",
    )


def test_comparison_authorization_boundary(tool, temporary: pathlib.Path) -> None:
    root = temporary / "comparison-fixtures"
    root.mkdir()
    oracle, mold = checked_comparison_pair(tool)
    oracle_path = root / "oracle.json"
    mold_path = root / "mold.json"
    write_json(oracle_path, oracle)
    write_json(mold_path, mold)

    expect_failure(
        lambda: tool.compare_output_files(str(oracle_path), str(mold_path)),
        "requires an approved external fixture root and authorization record",
    )
    authorization_path, _ = valid_authorization(tool, temporary)
    notes = tool.compare_output_files(
        str(oracle_path),
        str(mold_path),
        str(root),
        str(authorization_path),
    )
    assert len(notes) == 1

    expect_failure(
        lambda: tool.compare_output_files(
            str(tool.SYNTHETIC_ORACLE_PATH),
            str(tool.SYNTHETIC_MOLD_PATH),
            str(root),
        ),
        "must be supplied together",
    )

    authorization = tool.validate_authorization(authorization_path)
    unauthorized_oracle = copy.deepcopy(oracle)
    unauthorized_mold = copy.deepcopy(mold)
    for document in (unauthorized_oracle, unauthorized_mold):
        document["authority_tier"] = "exact-full-bf16"
        document["authorization_document_sha256"] = "0" * 64
    write_json(oracle_path, unauthorized_oracle)
    write_json(mold_path, unauthorized_mold)
    expect_failure(
        lambda: tool.compare_output_files(
            str(oracle_path),
            str(mold_path),
            str(root),
            str(authorization_path),
        ),
        "oracle output is not bound to the authorization evidence",
    )
    assert authorization["source_document_sha256"] != "0" * 64


def test_ci_routing() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "'scripts/minimax-h3-conformance.py'" in workflow
    assert "'scripts/tests/minimax-h3-conformance-contract.py'" in workflow
    assert "'tests/fixtures/minimax_h3/**'" in workflow
    assert "'docs/qualification/minimax-h3-*.json'" in workflow
    assert "python3 scripts/tests/minimax-h3-conformance-contract.py" in workflow


def main() -> int:
    tool = load_tool()
    tool.validate_manifest()
    printed = subprocess.run(
        [sys.executable, str(TOOL_PATH), "print-synthetic"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert json.loads(printed) == json.loads(tool.SYNTHETIC_PATH.read_bytes())
    compared = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "compare",
            "--oracle",
            str(tool.SYNTHETIC_ORACLE_PATH),
            "--mold",
            str(tool.SYNTHETIC_MOLD_PATH),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "policy=record-only" in compared
    assert "MiniMax H3 conformance compare passed" in compared
    test_ci_routing()

    with tempfile.TemporaryDirectory(prefix="mold-h3-contract-") as value:
        temporary = pathlib.Path(value).resolve()
        test_manifest_drift(tool, temporary)
        test_synthetic_drift(tool, temporary)
        test_synthetic_comparison_uses_fixture_math(tool)
        test_comparison_diagnostics(tool)
        test_comparison_authorization_boundary(tool, temporary)
        test_authorization_and_external_root(tool, temporary)
        test_external_bundle_hashes(tool, temporary)

    print("MiniMax H3 conformance contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
