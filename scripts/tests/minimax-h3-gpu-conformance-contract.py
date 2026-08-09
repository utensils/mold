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


def authorization_fixture(tool, root: pathlib.Path) -> tuple[pathlib.Path, str]:
    source = root / "authorization-source.txt"
    source.write_text("private conformance contract fixture\n", encoding="utf-8")
    source_sha = sha256(source)
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
    path = root / "authorization.json"
    write_json(path, record)
    return path, source_sha


def layer_document(
    tool,
    role: str,
    authorization_sha: str,
) -> dict[str, object]:
    fixture_path = (
        tool.SYNTHETIC_ORACLE_PATH if role == "oracle" else tool.SYNTHETIC_MOLD_PATH
    )
    document = json.loads(fixture_path.read_text(encoding="utf-8"))
    document["case_id"] = "gpu-contract-case"
    document["authority_tier"] = "exact-full-bf16"
    document["authorization_document_sha256"] = authorization_sha
    document["environment"] = {
        "device": "cuda:contract-test",
        "dtype": "bfloat16",
        "attention_backend": "math",
        "forbidden_accelerations_disabled": True,
    }
    if role == "mold":
        document["producer"]["revision"] = SOURCE_SHA
    return document


def bundle_fixture(
    tool,
    fixture_root: pathlib.Path,
    authorization_sha: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    manifest_sha = sha256(tool.MANIFEST_PATH)
    bundle_paths: list[pathlib.Path] = []
    for role in ("oracle", "mold"):
        document = layer_document(tool, role, authorization_sha)
        evidence_path = fixture_root / role / "gpu-contract-case.json"
        write_json(evidence_path, document)
        output = document["outputs"][0]
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
                "dtype": "bfloat16",
                "attention_backend": "math",
                "command": f"contract-test {role} capture",
                "forbidden_accelerations_disabled": True,
            },
            "fixtures": [
                {
                    "id": f"gpu-contract-case-{role}",
                    "layer": document["layer"],
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
                        "absolute": 0.000002,
                        "relative": 0.000001,
                        "metric": "elementwise-atol-plus-rtol",
                    },
                }
            ],
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


def test_runner_contract(runner, tool, temporary: pathlib.Path) -> None:
    fixture_root = temporary / "fixtures"
    fixture_root.mkdir()
    authorization, authorization_sha = authorization_fixture(tool, temporary)
    oracle_bundle, mold_bundle = bundle_fixture(tool, fixture_root, authorization_sha)
    environment = campaign_environment(
        fixture_root, authorization, oracle_bundle, mold_bundle
    )

    expect_failure(
        lambda: runner.resolve_external_file("authorization record", str(TOOL_PATH)),
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
    assert result == {"comparisons": 1, "notes": 1, "source_sha": SOURCE_SHA}
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

    mold_bundle_value = json.loads(mold_bundle.read_text(encoding="utf-8"))
    mismatched_component = copy.deepcopy(mold_bundle_value)
    mismatched_component["fixtures"][0]["component_index_sha256"] = "0" * 64
    write_json(mold_bundle, mismatched_component)
    expect_failure(
        lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
        "component index does not match",
    )
    write_json(mold_bundle, mold_bundle_value)

    mold_document_path = (
        fixture_root / mold_bundle_value["fixtures"][0]["relative_path"]
    )
    mold_document = json.loads(mold_document_path.read_text(encoding="utf-8"))
    quantized_document = copy.deepcopy(mold_document)
    quantized_document["authority_tier"] = "quantized-structural"
    write_json(mold_document_path, quantized_document)
    quantized_bundle = copy.deepcopy(mold_bundle_value)
    quantized_bundle["fixtures"][0]["authority_tier"] = "quantized-structural"
    quantized_bundle["fixtures"][0]["sha256"] = sha256(mold_document_path)
    write_json(mold_bundle, quantized_bundle)
    expect_failure(
        lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
        "exact-full-bf16",
    )

    synthetic_document = copy.deepcopy(mold_document)
    synthetic_document["authority_tier"] = "synthetic"
    synthetic_document["authorization_document_sha256"] = None
    write_json(mold_document_path, synthetic_document)
    synthetic_bundle = copy.deepcopy(mold_bundle_value)
    synthetic_bundle["fixtures"][0]["authority_tier"] = "synthetic"
    synthetic_bundle["fixtures"][0]["sha256"] = sha256(mold_document_path)
    write_json(mold_bundle, synthetic_bundle)
    expect_failure(
        lambda: runner.run_campaign(environment, lambda: None, lambda: SOURCE_SHA),
        "synthetic evidence is forbidden",
    )


def test_workflow_contract() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    actionlint = (REPO_ROOT / ".github" / "actionlint.yaml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    for trigger in ("push:", "pull_request:", "schedule:", "workflow_run:"):
        assert f"  {trigger}" not in workflow
    assert "environment: minimax-h3-private-uat" in workflow
    assert (
        "runs-on: [self-hosted, linux, x64, cuda, minimax-h3-private-uat]" in workflow
    )
    assert "permissions:\n  contents: read" in workflow
    assert "expected_source_sha:" in workflow
    assert "\"$GITHUB_REF\" != 'refs/heads/main'" in workflow
    assert '"$GITHUB_SHA" != "$EXPECTED_SOURCE_SHA"' in workflow
    for secret in (
        "MOLD_H3_FIXTURE_ROOT",
        "MOLD_H3_AUTHORIZATION_RECORD",
        "MOLD_H3_ORACLE_BUNDLE",
        "MOLD_H3_MOLD_BUNDLE",
    ):
        assert f"secrets.{secret}" in workflow
    assert "MOLD_H3_SOURCE_SHA: ${{ github.sha }}" in workflow
    assert "python3 scripts/run-minimax-h3-gpu-conformance.py" in workflow
    assert "upload-artifact" not in workflow
    assert "h3-private-uat" not in workflow.replace("minimax-h3-private-uat", "")
    assert "    - cuda" in actionlint
    assert "    - minimax-h3-private-uat" in actionlint

    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for path in (
        ".github/actionlint.yaml",
        ".github/workflows/minimax-h3-private-conformance.yml",
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
