#!/usr/bin/env python3
"""Hermetic schema and evidence validator for LTX-2.5 CUDA verification reports.

The schema walker, keyword audit, and hashing helpers are imported from
scripts/validate-cuda-qualification-report.py so the two CUDA validators
cannot drift on how a schema is read; nothing in that sibling is modified.
Every `*_path` that has a `*_sha256` sibling — in the report, in each row
manifest, and in each ComfyUI manifest — is re-hashed here, so a mutated
stdout.log, server.log, media file, or manifest is rejected even when the
report itself still parses.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

SIBLING = Path(__file__).resolve().parent / "validate-cuda-qualification-report.py"
_spec = importlib.util.spec_from_file_location("cuda_qualification_validator", SIBLING)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"cannot import {SIBLING}")
_sibling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sibling)

ValidationFailure = _sibling.ValidationFailure
validate_schema = _sibling.validate_schema
audit_schema_keywords = _sibling.audit_schema_keywords
sha256_file = _sibling.sha256_file
load_json = _sibling.load_json

ROW_STATUSES = ("passed", "failed", "blocked", "not_run")
REASON_SOURCES = ("admission", "runtime_readiness", "http_status", "oom_envelope")
SHA_RE = re.compile(r"[0-9a-f]{64}")


def fail(message: str) -> None:
    print(f"invalid LTX-2.5 CUDA verification report: {message}", file=sys.stderr)
    raise SystemExit(1)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def bind_hash_pairs(node: Any, label: str, root: Path | None) -> None:
    """Re-hash every `<stem>_path` / `<stem>_sha256` pair found in `node`.

    `media.path` / `media.sha256` is the one pair that does not carry a stem;
    it is handled as `path` + `sha256` on the same object.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "path" or key.endswith("_path"):
                stem = key[: -len("_path")] if key.endswith("_path") else ""
                sha_key = f"{stem}_sha256" if stem else "sha256"
                if sha_key not in node or node[sha_key] is None:
                    continue
                expected = node[sha_key]
                if not isinstance(value, str) or not value:
                    raise ValidationFailure(f"{label}.{key}: path is missing")
                if not isinstance(expected, str) or SHA_RE.fullmatch(expected) is None:
                    raise ValidationFailure(f"{label}.{sha_key}: invalid recorded SHA-256")
                path = Path(value)
                if not path.is_file():
                    raise ValidationFailure(f"{label}.{key}: evidence file is missing: {value}")
                if root is not None and not inside(path, root):
                    raise ValidationFailure(f"{label}.{key}: evidence lives outside {root}")
                if sha256_file(path) != expected:
                    raise ValidationFailure(f"{label}.{key}: evidence checksum mismatch")
            elif isinstance(value, (dict, list)):
                bind_hash_pairs(value, f"{label}.{key}", root)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            bind_hash_pairs(value, f"{label}[{index}]", root)


def validate_row(row: dict[str, Any], evidence_dir: Path, matrix_ids: set[str]) -> None:
    row_id = row["id"]
    label = f"rows[{row_id}]"
    if row_id not in matrix_ids:
        raise ValidationFailure(f"{label}: not a matrix row")
    if row["status"] not in ROW_STATUSES:
        raise ValidationFailure(f"{label}: unknown status")
    manifest_path = Path(row["manifest_path"])
    expected_manifest = evidence_dir / "rows" / row_id / "manifest.json"
    if manifest_path.resolve() != expected_manifest.resolve():
        raise ValidationFailure(f"{label}: manifest_path is not {expected_manifest}")
    if not manifest_path.is_file():
        raise ValidationFailure(f"{label}: row manifest is missing")
    if sha256_file(manifest_path) != row["manifest_sha256"]:
        raise ValidationFailure(f"{label}: row manifest checksum mismatch")
    manifest = load_json(manifest_path, label)
    if manifest.get("schema_version") != "mold.ltx25.cuda.row.v1":
        raise ValidationFailure(f"{label}: row manifest schema mismatch")
    for key in ("id", "model", "case", "profile", "status"):
        if manifest.get(key) != row.get(key):
            raise ValidationFailure(f"{label}: manifest {key} disagrees with the report")
    status = row["status"]
    if status == "blocked":
        if not row.get("reason") or row.get("reason_source") not in REASON_SOURCES:
            raise ValidationFailure(f"{label}: blocked row needs reason and reason_source")
    if status in ("failed", "not_run") and not row.get("reason"):
        raise ValidationFailure(f"{label}: {status} row needs a reason")
    if status in ("passed", "failed", "blocked"):
        if "stdout_log_path" not in manifest:
            raise ValidationFailure(f"{label}: attempted row retains no stdout.log")
        bind_hash_pairs(manifest, label, evidence_dir / "rows" / row_id)
    if status == "passed" and row["kind"] in ("cancellation", "fatal_cuda"):
        # Those rows retain logs rather than media; the hashes were bound above.
        return
    if status == "passed":
        media = row.get("media")
        if not isinstance(media, dict):
            raise ValidationFailure(f"{label}: passed row has no media")
        bind_hash_pairs({"media": media}, label, evidence_dir / "rows" / row_id)
        if Path(media["path"]).stat().st_size == 0:
            raise ValidationFailure(f"{label}: media is empty")
        expected = row.get("provenance_expected", [])
        observed = row.get("provenance_observed", [])
        if [item.get("line") for item in observed] != list(expected):
            raise ValidationFailure(f"{label}: observed provenance does not cover the expectation")
        generation = row.get("generation")
        if not isinstance(generation, dict) or generation.get("backend") != "cuda":
            raise ValidationFailure(f"{label}: Library row is not a CUDA generation")
        if generation.get("source") != "cli" or generation.get("metadata_synthetic") != 0:
            raise ValidationFailure(f"{label}: Library row is not a CLI-recorded real generation")
        server_log = Path(manifest["server_log_path"])
        text = server_log.read_text(encoding="utf-8", errors="replace")
        full_text = ""
        full_path = manifest.get("server_log_full_path")
        if isinstance(full_path, str) and Path(full_path).is_file():
            full_text = Path(full_path).read_text(encoding="utf-8", errors="replace")
        for item in observed:
            line = item["line"]
            if item["scope"] == "slice" and line not in text:
                raise ValidationFailure(f"{label}: {line!r} is absent from the retained server log slice")
            if item["scope"] == "process" and line not in text and line not in full_text:
                raise ValidationFailure(f"{label}: {line!r} is absent from the retained server logs")


def validate_comfy(reference: dict[str, Any], label: str) -> None:
    status = reference["status"]
    if status == "not_run":
        if not reference.get("reason"):
            raise ValidationFailure(f"{label}: not_run needs a reason")
        return
    manifest_path = Path(reference["manifest_path"])
    if not manifest_path.is_file():
        raise ValidationFailure(f"{label}: ComfyUI manifest is missing")
    if sha256_file(manifest_path) != reference["manifest_sha256"]:
        raise ValidationFailure(f"{label}: ComfyUI manifest checksum mismatch")
    manifest = load_json(manifest_path, label)
    if manifest.get("schema_version") != "mold.ltx25.comfy-cuda-reference.v1":
        raise ValidationFailure(f"{label}: ComfyUI manifest schema mismatch")
    if manifest.get("backend") != "CUDA" or manifest.get("status") != status:
        raise ValidationFailure(f"{label}: ComfyUI manifest backend/status mismatch")
    bind_hash_pairs(manifest, label, None)
    if status == "operator_deferred":
        cause = manifest.get("deferred", {}).get("guard_cause")
        if cause not in (
            "pressure_unreadable",
            "host_memory",
            "server_rss",
            "timeout",
            "gpu_unreadable",
            "torch_cuda_unavailable",
        ):
            raise ValidationFailure(f"{label}: unknown ComfyUI guard cause {cause!r}")


def validate_report(report_path: Path, schema_path: Path) -> None:
    schema = load_json(schema_path, "schema")
    audit_schema_keywords(schema)
    report = load_json(report_path, "report")
    validate_schema(report, schema, schema)

    # The runner names the evidence directory after the report stem
    # (`<name>.json` -> `<name>.d`), mirroring the Metal capture.
    evidence_dir = report_path.resolve().with_suffix(".d")
    if not evidence_dir.is_dir():
        raise ValidationFailure(f"evidence directory is missing: {evidence_dir}")

    matrix = report["matrix"]
    matrix_path = Path(matrix["path"])
    if not matrix_path.is_file() or sha256_file(matrix_path) != matrix["sha256"]:
        raise ValidationFailure("matrix fixture is missing or does not match its recorded hash")
    matrix_doc = load_json(matrix_path, "matrix")
    matrix_ids = {row["id"] for row in matrix_doc.get("rows", [])}
    if matrix["rows"] != len(matrix_ids):
        raise ValidationFailure("matrix row count disagrees with the fixture")

    rows = report["rows"]
    seen = {row["id"] for row in rows}
    if seen != matrix_ids:
        raise ValidationFailure("report rows do not cover the matrix exactly once")
    for row in rows:
        validate_row(row, evidence_dir, matrix_ids)

    counts = {status: sum(1 for row in rows if row["status"] == status) for status in ROW_STATUSES}
    if report["summary"] != counts:
        raise ValidationFailure(f"summary {report['summary']} disagrees with rows {counts}")

    for gate in report["gates"]:
        bind_hash_pairs({"log_path": gate["log_path"], "log_sha256": gate["log_sha256"]},
                        f"gates[{gate['label']}]", evidence_dir)
        if gate["status"] == "failed":
            raise ValidationFailure(f"gate {gate['label']} failed")

    for key in ("int8", "gguf_q4"):
        validate_comfy(report["comfy_reference"][key], f"comfy_reference.{key}")

    build = report["host"]["build"]
    binary = Path(build["binary_path"])
    if binary.is_file() and sha256_file(binary) != build["binary_sha256"]:
        raise ValidationFailure("qualified binary no longer matches its recorded hash")
    if build["candle_rev"] != build["cargo_lock_candle_rev"]:
        raise ValidationFailure("build candle revision disagrees with Cargo.lock")

    status = report["qualification_status"]
    if report["source_tree_state"] == "contract_test":
        if status != "not_qualified_contract_test":
            raise ValidationFailure("contract-test reports never qualify hardware")
    elif counts["failed"] > 0:
        if status != "failed":
            raise ValidationFailure("a report with failed rows must be marked failed")
    elif counts["passed"] == 0:
        if status != "incomplete":
            raise ValidationFailure("a report with no passed rows is incomplete")
    elif status != "passed":
        raise ValidationFailure("a report with passed rows and no failures is passed")


def main() -> None:
    args = sys.argv[1:]
    if len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--schema"):
        fail("usage: validate-ltx25-cuda-report.py <report.json> [--schema <schema.json>]")
    report_path = Path(args[0])
    schema_path = (
        Path(args[2])
        if len(args) == 3
        else Path(__file__).resolve().parent.parent
        / "docs/qualification/ltx25-cuda-verification.schema.json"
    )
    try:
        validate_report(report_path, schema_path)
    except (ValidationFailure, KeyError, TypeError, ValueError, OSError) as error:
        fail(str(error))
    print("LTX-2.5 CUDA verification report schema, relationships, and evidence: ok")


if __name__ == "__main__":
    main()
