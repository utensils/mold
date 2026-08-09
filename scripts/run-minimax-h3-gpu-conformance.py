#!/usr/bin/env python3
"""Validate an authorization-bound MiniMax H3 GPU conformance campaign.

This runner consumes already captured external evidence. Adapter commands in
the evidence are provenance only and are never executed here. The public Mold
product remains fail-closed and no evidence is copied into the checkout or CI.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
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
EXACT_AUTHORITY_TIER = "exact-full-bf16"


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


def call_redacted(tool: Any, context: str, action: Callable[[], Any]) -> Any:
    try:
        return action()
    except (tool.ConformanceFailure, OSError, ValueError):
        fail(f"{context} failed; inspect evidence only on the protected runner")


def validate_capture_environment(
    tool: Any,
    bundle: dict[str, Any],
    role: str,
    source_sha: str,
) -> None:
    capture = bundle["capture_environment"]
    if not is_cuda_environment(capture["device"]):
        fail(f"{role} bundle does not declare a CUDA capture environment")
    if role == "oracle":
        expected_framework = "diffusers"
        expected_revision = tool.EXPECTED_REVISIONS["diffusers"]
    else:
        expected_framework = "mold"
        expected_revision = source_sha
    if capture["framework"] != expected_framework:
        fail(f"{role} bundle framework is not {expected_framework}")
    if capture["framework_revision"] != expected_revision:
        fail(f"{role.capitalize()} bundle framework revision is not exact")


def load_layer_documents(
    tool: Any,
    fixture_root: pathlib.Path,
    authorization_sha: str,
    bundle: dict[str, Any],
    role: str,
    source_sha: str,
) -> dict[tuple[str, str], pathlib.Path]:
    documents: dict[tuple[str, str], pathlib.Path] = {}
    for position, fixture in enumerate(bundle["fixtures"], start=1):
        if fixture["authority_tier"] == "synthetic":
            fail(f"{role} bundle synthetic evidence is forbidden in the GPU campaign")
        if fixture["authority_tier"] != EXACT_AUTHORITY_TIER:
            fail(f"{role} bundle evidence must be {EXACT_AUTHORITY_TIER}")
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
        document = call_redacted(
            tool,
            f"{role} layer evidence validation",
            lambda: tool.validate_layer_output(
                tool.load_json(evidence_path), f"{role} evidence {position}"
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
        if document["authority_tier"] == "synthetic":
            fail(f"{role} layer synthetic evidence is forbidden in the GPU campaign")
        if document["authority_tier"] != EXACT_AUTHORITY_TIER:
            fail(f"{role} layer evidence must be {EXACT_AUTHORITY_TIER}")
        if document["authorization_document_sha256"] != authorization_sha:
            fail(f"{role} layer is not bound to the authorization evidence")
        if not is_cuda_environment(document["environment"]["device"]):
            fail(f"{role} layer does not declare a CUDA execution environment")
        if fixture["layer"] != document["layer"]:
            fail(f"{role} bundle layer does not match its evidence")
        if fixture["authority_tier"] != document["authority_tier"]:
            fail(f"{role} bundle authority tier does not match its evidence")
        if (
            fixture["component_index_sha256"]
            != document["input"]["component_index_sha256"]
        ):
            fail(f"{role} bundle component index does not match its evidence")
        key = (document["case_id"], document["layer"])
        if key in documents:
            fail(f"{role} bundle contains duplicate case/layer evidence")
        documents[key] = evidence_path
    return documents


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
    oracle_bundle_path = resolve_bundle_path(
        fixture_root, "oracle bundle", values["MOLD_H3_ORACLE_BUNDLE"]
    )
    mold_bundle_path = resolve_bundle_path(
        fixture_root, "Mold bundle", values["MOLD_H3_MOLD_BUNDLE"]
    )
    if oracle_bundle_path == mold_bundle_path:
        fail("oracle and Mold bundles must be distinct files")

    gpu_probe()

    for label, path in (
        ("oracle bundle", oracle_bundle_path),
        ("Mold bundle", mold_bundle_path),
    ):
        call_redacted(
            tool,
            f"{label} validation",
            lambda path=path: tool.validate_bundle(
                manifest,
                str(fixture_root),
                str(path),
                str(authorization_path),
            ),
        )
    oracle_bundle = call_redacted(
        tool, "oracle bundle reload", lambda: tool.load_json(oracle_bundle_path)
    )
    mold_bundle = call_redacted(
        tool, "Mold bundle reload", lambda: tool.load_json(mold_bundle_path)
    )
    source_sha = values["MOLD_H3_SOURCE_SHA"]
    validate_capture_environment(tool, oracle_bundle, "oracle", source_sha)
    validate_capture_environment(tool, mold_bundle, "mold", source_sha)
    authorization_sha = authorization["source_document_sha256"]
    oracle_documents = load_layer_documents(
        tool,
        fixture_root,
        authorization_sha,
        oracle_bundle,
        "oracle",
        source_sha,
    )
    mold_documents = load_layer_documents(
        tool,
        fixture_root,
        authorization_sha,
        mold_bundle,
        "mold",
        source_sha,
    )
    if not oracle_documents or set(oracle_documents) != set(mold_documents):
        fail(
            "oracle and Mold comparison sets differ "
            f"(oracle={len(oracle_documents)}, mold={len(mold_documents)})"
        )

    notes = 0
    for position, key in enumerate(sorted(oracle_documents), start=1):
        comparison_notes = call_redacted(
            tool,
            f"authorized comparison {position}",
            lambda key=key: tool.compare_output_files(
                str(oracle_documents[key]),
                str(mold_documents[key]),
                str(fixture_root),
                str(authorization_path),
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
