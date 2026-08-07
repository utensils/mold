#!/usr/bin/env python3
"""Mutation tests for the weight-free MiniMax H3 conformance boundary."""

from __future__ import annotations

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


def valid_authorization(tool, temporary: pathlib.Path) -> tuple[pathlib.Path, dict[str, object]]:
    source_document = temporary / "minimax-authorization.txt"
    source_document.write_text("synthetic external authorization evidence\n", encoding="utf-8")
    record = {
        "schema_version": tool.AUTHORIZATION_SCHEMA_VERSION,
        "family": "minimax-h3",
        "decision": "approved",
        "license_revision": tool.EXPECTED_REVISIONS["minimax-official-model"],
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

    pathlib.Path(record["source_document_path"]).write_text("mutated\n", encoding="utf-8")
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
        lambda: tool.validate_bundle(manifest, str(root), str(bundle_path), str(record_path)),
        "fixture evidence hash mismatch",
    )


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

    with tempfile.TemporaryDirectory(prefix="mold-h3-contract-") as value:
        temporary = pathlib.Path(value).resolve()
        test_manifest_drift(tool, temporary)
        test_synthetic_drift(tool, temporary)
        test_authorization_and_external_root(tool, temporary)
        test_external_bundle_hashes(tool, temporary)

    print("MiniMax H3 conformance contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
