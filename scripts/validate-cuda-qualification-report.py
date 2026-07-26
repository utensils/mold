#!/usr/bin/env python3
"""Hermetic schema and cryptographic evidence validator for CUDA qualification."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any


class ValidationFailure(Exception):
    pass


def fail(message: str) -> None:
    print(f"invalid CUDA qualification report: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SUPPORTED_SCHEMA_KEYWORDS = {
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "title",
    "description",
    "type",
    "required",
    "properties",
    "additionalProperties",
    "items",
    "const",
    "enum",
    "pattern",
    "minLength",
    "minItems",
    "minimum",
    "format",
    "allOf",
    "if",
    "then",
}


def audit_schema_keywords(schema: Any, location: str = "#") -> None:
    if not isinstance(schema, dict):
        raise ValidationFailure(f"{location}: schema node is not an object")
    unsupported = set(schema) - SUPPORTED_SCHEMA_KEYWORDS
    if unsupported:
        raise ValidationFailure(
            f"{location}: unsupported schema keyword(s): {', '.join(sorted(unsupported))}"
        )
    for map_keyword in ("properties", "$defs"):
        values = schema.get(map_keyword, {})
        if not isinstance(values, dict):
            raise ValidationFailure(f"{location}/{map_keyword}: expected object")
        for name, child in values.items():
            audit_schema_keywords(child, f"{location}/{map_keyword}/{name}")
    if "items" in schema:
        audit_schema_keywords(schema["items"], f"{location}/items")
    for list_keyword in ("allOf",):
        values = schema.get(list_keyword, [])
        if not isinstance(values, list):
            raise ValidationFailure(f"{location}/{list_keyword}: expected array")
        for index, child in enumerate(values):
            audit_schema_keywords(child, f"{location}/{list_keyword}/{index}")
    for conditional in ("if", "then"):
        if conditional in schema:
            audit_schema_keywords(schema[conditional], f"{location}/{conditional}")


def resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValidationFailure(f"unsupported non-local schema reference: {reference}")
    node: Any = root
    for encoded in reference[2:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or key not in node:
            raise ValidationFailure(f"unresolved schema reference: {reference}")
        node = node[key]
    if not isinstance(node, dict):
        raise ValidationFailure(f"schema reference is not an object: {reference}")
    return node


def json_type_matches(instance: Any, expected: str) -> bool:
    return {
        "object": isinstance(instance, dict),
        "array": isinstance(instance, list),
        "string": isinstance(instance, str),
        "boolean": isinstance(instance, bool),
        "integer": isinstance(instance, int) and not isinstance(instance, bool),
        "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
        "null": instance is None,
    }.get(expected, False)


def validate_schema(
    instance: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    location: str = "$",
) -> None:
    if "$ref" in schema:
        validate_schema(instance, resolve_ref(root, schema["$ref"]), root, location)
    for child in schema.get("allOf", []):
        validate_schema(instance, child, root, location)

    if "if" in schema:
        try:
            validate_schema(instance, schema["if"], root, location)
            condition_matches = True
        except ValidationFailure:
            condition_matches = False
        if condition_matches and "then" in schema:
            validate_schema(instance, schema["then"], root, location)

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = [expected_type] if isinstance(expected_type, str) else expected_type
        if not isinstance(expected_types, list) or not all(
            isinstance(value, str) for value in expected_types
        ):
            raise ValidationFailure(f"{location}: invalid schema type declaration")
        if not any(json_type_matches(instance, value) for value in expected_types):
            raise ValidationFailure(f"{location}: expected type {expected_types}")

    if "const" in schema and instance != schema["const"]:
        raise ValidationFailure(f"{location}: value does not match const")
    if "enum" in schema and instance not in schema["enum"]:
        raise ValidationFailure(f"{location}: value is not in enum")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(value, str) for value in required
        ):
            raise ValidationFailure(f"{location}: invalid required declaration")
        missing = [key for key in required if key not in instance]
        if missing:
            raise ValidationFailure(f"{location}: missing required keys {missing}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValidationFailure(f"{location}: invalid properties declaration")
        if schema.get("additionalProperties") is False:
            extras = set(instance) - set(properties)
            if extras:
                raise ValidationFailure(
                    f"{location}: unexpected properties {sorted(extras)}"
                )
        for key, child_schema in properties.items():
            if key in instance:
                validate_schema(instance[key], child_schema, root, f"{location}.{key}")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise ValidationFailure(f"{location}: array is too short")
        if "items" in schema:
            for index, value in enumerate(instance):
                validate_schema(value, schema["items"], root, f"{location}[{index}]")

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise ValidationFailure(f"{location}: string is too short")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise ValidationFailure(f"{location}: string does not match pattern")
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(instance.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValidationFailure(f"{location}: invalid date-time") from error
            if parsed.tzinfo is None:
                raise ValidationFailure(f"{location}: date-time lacks timezone")
        elif "format" in schema:
            raise ValidationFailure(
                f"{location}: unsupported schema format {schema['format']}"
            )

    if (
        isinstance(instance, (int, float))
        and not isinstance(instance, bool)
        and "minimum" in schema
        and instance < schema["minimum"]
    ):
        raise ValidationFailure(f"{location}: value is below minimum")


def exact_path(value: Any, expected: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationFailure(f"{label}: path is missing")
    actual = Path(value).resolve()
    if actual != expected.resolve():
        raise ValidationFailure(f"{label}: path is not the exact expected evidence path")
    if not actual.is_file():
        raise ValidationFailure(f"{label}: evidence file is missing")
    return actual


def hash_bound_file(
    value: Any,
    expected_path: Path,
    expected_sha: Any,
    label: str,
) -> Path:
    path = exact_path(value, expected_path, label)
    if not isinstance(expected_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha
    ):
        raise ValidationFailure(f"{label}: invalid recorded SHA-256")
    if sha256_file(path) != expected_sha:
        raise ValidationFailure(f"{label}: evidence checksum mismatch")
    return path


def parse_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise ValidationFailure("SHA256SUMS contains a malformed entry")
        name = fields[1].lstrip("*")
        if name.startswith("./"):
            name = name[2:]
        if not name or name in result:
            raise ValidationFailure("SHA256SUMS contains a duplicate or empty filename")
        result[name] = fields[0]
    return result


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationFailure(f"{label}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValidationFailure(f"{label}: JSON root is not an object")
    return value


def validate_artifact(
    key: str,
    artifact: dict[str, Any],
    provenance: dict[str, Any],
    checksums: dict[str, str],
) -> None:
    target = {"sm86": "sm_86", "sm89": "sm_89"}[key]
    filename = f"mold-x86_64-unknown-linux-gnu-cuda-{key}.tar.gz"
    manifest_artifact = provenance.get("artifacts", {}).get(key, {})
    if (
        artifact.get("cuda_target") != target
        or manifest_artifact.get("cuda_target") != target
        or manifest_artifact.get("filename") != filename
    ):
        raise ValidationFailure(f"{key}: target or filename provenance mismatch")
    binary_path = Path(artifact["path"]).resolve()
    archive_path = Path(artifact["archive_path"]).resolve()
    if not binary_path.is_file() or not archive_path.is_file():
        raise ValidationFailure(f"{key}: binary or archive is missing")
    actual_binary = sha256_file(binary_path)
    actual_archive = sha256_file(archive_path)
    expected_binary = manifest_artifact.get("binary_sha256")
    expected_archive = manifest_artifact.get("archive_sha256")
    if not all(
        artifact.get(field)
        for field in (
            "trusted_checksum_verified",
            "elf_target_verified",
            "ptx_target_verified",
            "source_identity_verified",
        )
    ):
        raise ValidationFailure(f"{key}: verification boolean is false")
    if not (
        expected_binary
        == artifact.get("expected_sha256")
        == artifact.get("actual_sha256")
        == actual_binary
    ):
        raise ValidationFailure(f"{key}: binary hash is not provenance-bound")
    if not (
        expected_archive
        == checksums.get(filename)
        == artifact.get("expected_archive_sha256")
        == artifact.get("actual_archive_sha256")
        == actual_archive
    ):
        raise ValidationFailure(f"{key}: archive hash is not provenance-bound")
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != 1 or members[0].name != "mold" or not members[0].isfile():
                raise ValidationFailure(f"{key}: archive must contain exactly mold")
            member = archive.extractfile(members[0])
            if member is None or hashlib.sha256(member.read()).hexdigest() != actual_binary:
                raise ValidationFailure(f"{key}: archive member is not the exact binary")
    except (tarfile.TarError, OSError) as error:
        raise ValidationFailure(f"{key}: invalid archive") from error


def validate_compute_observation(
    path: Path,
    root_pid: int,
    observed_pid: int,
    selected_uuid: str,
    label: str,
) -> None:
    try:
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValidationFailure(f"{label}: invalid compute observation CSV") from error
    expected_fields = {
        "polled_at_utc",
        "generation_root_pid",
        "observed_pid",
        "gpu_uuid",
    }
    if not rows or set(rows[0]) != expected_fields:
        raise ValidationFailure(f"{label}: compute observation has no bound rows")
    if not any(
        row["generation_root_pid"] == str(root_pid)
        and row["observed_pid"] == str(observed_pid)
        and row["gpu_uuid"] == selected_uuid
        for row in rows
    ):
        raise ValidationFailure(f"{label}: PID/GPU observation relationship is absent")


def validate_report(report_path: Path, schema_path: Path) -> None:
    schema = load_json(schema_path, "schema")
    audit_schema_keywords(schema)
    report = load_json(report_path, "report")
    validate_schema(report, schema, schema)

    evidence_dir = Path(f"{report_path.resolve()}.d")
    release_tag = report["release_tag"]
    source_sha = report["source_sha"]
    provenance_report = report["provenance"]
    provenance_path = hash_bound_file(
        provenance_report["official_release_manifest_path"],
        evidence_dir / "mold-release-provenance.json",
        provenance_report["official_release_manifest_sha256"],
        "release provenance",
    )
    checksums_path = hash_bound_file(
        provenance_report["checksum_manifest_path"],
        evidence_dir / "SHA256SUMS",
        provenance_report["checksum_manifest_sha256"],
        "checksum manifest",
    )
    provenance = load_json(provenance_path, "release provenance")
    checksums = parse_checksums(checksums_path)
    if checksums.get("mold-release-provenance.json") != sha256_file(provenance_path):
        raise ValidationFailure("SHA256SUMS does not bind release provenance")
    if (
        provenance.get("schema_version") != "mold.release.provenance.v1"
        or provenance.get("release_tag") != release_tag
        or provenance.get("source_sha") != source_sha
    ):
        raise ValidationFailure("release provenance identity mismatch")
    expected_base = (
        f"https://github.com/utensils/mold/releases/download/{release_tag}"
    )
    if (
        provenance_report["official_release_manifest_url"]
        != f"{expected_base}/mold-release-provenance.json"
        or provenance_report["checksum_manifest_url"]
        != f"{expected_base}/SHA256SUMS"
        or not provenance_report["official_release_manifest_verified"]
    ):
        raise ValidationFailure("official release URL or verification state mismatch")

    for key in ("sm86", "sm89"):
        validate_artifact(key, report["artifacts"][key], provenance, checksums)
        if source_sha[:7] not in report["artifacts"][key]["version_output"]:
            raise ValidationFailure(f"{key}: version output does not identify source")

    devices = report["host"]["devices"]
    device_uuids = {device["uuid"] for device in devices}
    if any(
        "RTX 3090" not in device["name"].upper()
        or device["compute_capability"] != "8.6"
        for device in devices
    ):
        raise ValidationFailure("qualification device is not RTX 3090 compute capability 8.6")

    required_tests = {
        "sm86_attention_image_smoke": ("image", ".png"),
        "sm86_ptx_image_smoke": ("image", ".png"),
        "sm86_video_smoke": ("video", ".mp4"),
        "sm86_chained_video_smoke": ("video", ".mp4"),
    }
    tests = report["tests"]
    for name, (media_kind, extension) in required_tests.items():
        result = tests[name]
        if result["status"] != "passed":
            continue
        if result["selected_gpu_uuid"] not in device_uuids:
            raise ValidationFailure(f"{name}: selected GPU is not qualified")
        output = hash_bound_file(
            result["output_path"],
            evidence_dir / f"{name}{extension}",
            result["output_sha256"],
            f"{name} output",
        )
        if output.stat().st_size == 0:
            raise ValidationFailure(f"{name}: output is empty")
        log = hash_bound_file(
            result["log_path"],
            evidence_dir / f"{name}.log",
            result["log_sha256"],
            f"{name} log",
        )
        observation = hash_bound_file(
            result["compute_observation_path"],
            evidence_dir / f"{name}-compute-observations.csv",
            result["compute_observation_sha256"],
            f"{name} compute observation",
        )
        validate_compute_observation(
            observation,
            result["generation_root_pid"],
            result["observed_compute_pid"],
            result["selected_gpu_uuid"],
            name,
        )
        if media_kind == "image" and re.search(
            r"mold_inference::attention: attention backend selected "
            r"backend=Math(?:\s|$)",
            log.read_text(encoding="utf-8", errors="replace"),
        ) is None:
            raise ValidationFailure(f"{name}: exact math attention selection is absent")
        if media_kind == "video" and result["frame_count"] < 1:
            raise ValidationFailure(f"{name}: decoded video has no frames")

        if name == "sm86_ptx_image_smoke":
            probe_path = hash_bound_file(
                result["embedded_ptx_probe_path"],
                evidence_dir / f"{name}-embedded-ptx.json",
                result["embedded_ptx_probe_sha256"],
                f"{name} PTX probe",
            )
            probe = load_json(probe_path, f"{name} PTX probe")
            if (
                probe.get("expected_target") != "sm_86"
                or probe.get("observed_targets") != ["sm_86"]
                or probe.get("malformed_modules") != []
                or probe.get("incomplete_modules") != []
                or not probe.get("loaded")
                or probe.get("artifact_sha256")
                != report["artifacts"]["sm86"]["actual_sha256"]
                or not any(
                    attempt.get("loaded") for attempt in probe.get("attempts", [])
                )
            ):
                raise ValidationFailure(f"{name}: loaded PTX is not artifact-bound")

    negative = report["compatibility_controls"]["sm89_driver_negative"]
    if negative["selected_gpu_uuid"] not in device_uuids:
        raise ValidationFailure("sm89 negative control selected an unqualified GPU")
    command_path = hash_bound_file(
        negative["command_path"],
        evidence_dir / "sm89-driver-negative.command",
        negative["command_sha256"],
        "sm89 negative command",
    )
    if command_path.read_text(encoding="utf-8").strip() != negative["command"].strip():
        raise ValidationFailure("sm89 negative command text is not evidence-bound")
    negative_probe_path = hash_bound_file(
        negative["probe_path"],
        evidence_dir / "sm89-driver-negative.json",
        negative["probe_sha256"],
        "sm89 negative probe",
    )
    if negative["status"] == "passed":
        negative_probe = load_json(negative_probe_path, "sm89 negative probe")
        attempts = negative_probe.get("attempts", [])
        if (
            negative["exit_code"] != 0
            or not negative["expected_incompatibility"]
            or negative_probe.get("expected_target") != "sm_89"
            or negative_probe.get("observed_targets") != ["sm_89"]
            or negative_probe.get("malformed_modules") != []
            or negative_probe.get("incomplete_modules") != []
            or negative_probe.get("artifact_sha256")
            != report["artifacts"]["sm89"]["actual_sha256"]
            or negative_probe.get("loaded")
            or not attempts
            or any(
                attempt.get("cuda_result") not in (209, 218)
                or attempt.get("loaded") is not False
                for attempt in attempts
            )
        ):
            raise ValidationFailure("sm89 exact-ELF negative Driver evidence is invalid")

    if report["hardware_qualified"]:
        if set(tests) != set(required_tests):
            raise ValidationFailure("qualified report has an unexpected test set")
        if any(result["status"] != "passed" for result in tests.values()):
            raise ValidationFailure("qualified report contains a non-passing workload")
        if negative["status"] != "passed":
            raise ValidationFailure("qualified report lacks sm89 negative Driver control")


def main() -> None:
    args = sys.argv[1:]
    if len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--schema"):
        fail(
            "usage: validate-cuda-qualification-report.py "
            "<report.json> [--schema <schema.json>]"
        )
    report_path = Path(args[0])
    schema_path = (
        Path(args[2])
        if len(args) == 3
        else Path(__file__).resolve().parent.parent
        / "docs/qualification/cuda-sm86-report.schema.json"
    )
    try:
        validate_report(report_path, schema_path)
    except (ValidationFailure, KeyError, TypeError, ValueError, OSError) as error:
        fail(str(error))
    print("CUDA qualification report schema, relationships, and evidence: ok")


if __name__ == "__main__":
    main()
