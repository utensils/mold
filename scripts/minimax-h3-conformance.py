#!/usr/bin/env python3
"""Validate MiniMax H3 conformance pins and external fixture evidence.

The checked-in contract is intentionally weight-free. Real checkpoint evidence
must live outside the repository and is accepted only with a separately managed
authorization record whose source document is content-addressed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "minimax_h3"
MANIFEST_PATH = FIXTURE_DIR / "conformance-manifest.json"
SYNTHETIC_PATH = FIXTURE_DIR / "synthetic-v1.json"
MANIFEST_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "qualification" / "minimax-h3-conformance-manifest.schema.json"
)
BUNDLE_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "qualification" / "minimax-h3-fixture-bundle.schema.json"
)

SCHEMA_VERSION = "mold.minimax-h3.conformance-manifest.v1"
SYNTHETIC_SCHEMA_VERSION = "mold.minimax-h3.synthetic.v1"
BUNDLE_SCHEMA_VERSION = "mold.minimax-h3.fixture-bundle.v1"
AUTHORIZATION_SCHEMA_VERSION = "mold.minimax-h3.authorization.v1"

EXPECTED_REVISIONS = {
    "minimax-official-code": "8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea",
    "minimax-official-model": "bfc8ed0353f5a9733be73e6b2c98ec0948195b86",
    "diffusers": "9c6a68c32b3b2a64db91800b624d33cec6e25ab8",
    "comfyui": "a464ac33588ae182f81a090d910cfbf21e255b73",
    "comfy-checkpoints": "eb8a16107c595128b3a578f82d2ce2f75920c355",
    "sglang": "0c3a76fa0a5bfab410b645f4143e7e8e3cc25c77",
    "vllm-omni": "3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc",
}

EXPECTED_LICENSE_SHA256 = "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44"

EXPECTED_EXCLUDED_ACCELERATIONS = {
    "torch-compile",
    "fp8",
    "cache-dit",
    "sageattention",
    "stochastic-sampling",
    "approximate-attention",
    "tf32",
}

EXPECTED_COMPONENT_INDEXES = {
    "official-license": (
        "LICENSE",
        EXPECTED_LICENSE_SHA256,
    ),
    "official-model-index": (
        "model_index.json",
        "5a587fe13b2371427415ac892463142683aefcd8d322e274a3a095eac37ac7d2",
    ),
    "official-modular-index": (
        "modular_model_index.json",
        "a2b6a210e482ffb78e613b553f570c44e101afce6741bd4ed91429d0559af031",
    ),
    "official-fl2va-transformer-index": (
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e",
    ),
    "official-ref2va-transformer-index": (
        "transformer_ref/diffusion_pytorch_model.safetensors.index.json",
        "ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e",
    ),
    "official-text-encoder-index": (
        "text_encoder/model.safetensors.index.json",
        "06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a",
    ),
    "official-video-vae-index": (
        "vae/diffusion_pytorch_model.safetensors.index.json",
        "15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca",
    ),
    "official-audio-vae-config": (
        "audio_vae/config.json",
        "9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6",
    ),
}

EXPECTED_LAYERS = {
    "tokenizer-processor",
    "qwen-layer-50",
    "visual-vae",
    "audio-vae",
    "token-refiner",
    "transformer-block",
    "packed-layout",
    "dual-sampler",
    "noise-allocation",
    "end-to-end-t2va",
    "end-to-end-fl2va",
    "end-to-end-ref2va",
}

FORBIDDEN_FIXTURE_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".flac",
    ".gguf",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".mp4",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
    ".webp",
}


class ConformanceFailure(Exception):
    """A fail-closed conformance contract violation."""


def fail(message: str) -> None:
    raise ConformanceFailure(message)


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail(f"cannot hash {path}: {error}")
    return digest.hexdigest()


def schema_helper() -> Any:
    helper_path = REPO_ROOT / "scripts" / "validate-cuda-qualification-report.py"
    spec = importlib.util.spec_from_file_location("mold_schema_helper", helper_path)
    if spec is None or spec.loader is None:
        fail(f"cannot load hermetic schema helper: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def validate_schema(instance: Any, schema_path: pathlib.Path) -> None:
    helper = schema_helper()
    schema = load_json(schema_path)
    try:
        helper.audit_schema_keywords(schema)
        helper.validate_schema(instance, schema, schema)
    except helper.ValidationFailure as error:
        fail(f"schema validation failed for {schema_path.name}: {error}")


def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def shifted_schedule(grid_points: int, shift: float) -> list[float]:
    values: list[float] = []
    for index in range(grid_points):
        base = f32(1.0 - index / (grid_points - 1))
        numerator = f32(f32(shift) * base)
        denominator = f32(1.0 + f32(f32(shift - 1.0) * base))
        shifted = f32(numerator / denominator)
        if not values or shifted != values[-1]:
            values.append(shifted)
    return values


def euler_update(
    sample: list[float], velocity: list[float], sigma: float, sigma_next: float
) -> list[float]:
    timestep = f32(1.0 - sigma)
    sigma_from_timestep = f32(1.0 - timestep)
    ratio = f32(sigma_next / sigma)
    one_minus_ratio = f32(1.0 - ratio)
    output = []
    for sample_value, velocity_value in zip(sample, velocity, strict=True):
        sample_value = f32(sample_value)
        velocity_value = f32(velocity_value)
        denoised = f32(sample_value + f32(sigma_from_timestep * velocity_value))
        output.append(
            f32(f32(ratio * sample_value) + f32(one_minus_ratio * denoised))
        )
    return output


def align_frames(requested: int) -> int:
    while requested % 17 != 5:
        requested += 1
    return requested


def synthetic_fixture() -> dict[str, Any]:
    frame_grid = []
    for nominal_seconds in (5, 10, 15):
        requested = nominal_seconds * 24
        aligned = align_frames(requested)
        frame_grid.append(
            {
                "nominal_seconds": nominal_seconds,
                "requested_frames": requested,
                "aligned_frames": aligned,
                "video_latent_frames": ((aligned - 5) // 17) * 5 + 2,
                "audio_latents_per_channel": round(aligned / 24 * 40),
                "actual_duration_seconds": aligned / 24,
            }
        )

    video_sigmas = shifted_schedule(4, 12.0)
    audio_sigmas = shifted_schedule(4, 3.0)
    sample = [0.25, -0.5, 1.0, -2.0]
    video_velocity = [0.5, 0.25, -1.5, 2.0]
    audio_velocity = [-0.25, 0.75, 0.5, -1.0]
    step_index = 1

    spans = [5.0 / 3.0 * value for value in (1, 4, 4, 4, 4, 1)]
    temporal_positions = [3.0]
    for span in spans:
        temporal_positions.append(temporal_positions[-1] + span)

    return {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "authority": {
            "source": "diffusers-minimax-h3",
            "revision": EXPECTED_REVISIONS["diffusers"],
            "execution": "weight-free-python-float32-emulation",
        },
        "frame_grid": frame_grid,
        "scheduler": {
            "grid_points": 4,
            "transformer_evaluations": 3,
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "video_sigmas": video_sigmas,
            "audio_sigmas": audio_sigmas,
            "video_timesteps": [f32(1.0 - value) for value in video_sigmas[:-1]],
            "audio_timesteps": [f32(1.0 - value) for value in audio_sigmas[:-1]],
            "terminal_zero_included": True,
            "ancestral_noise_reinjected": False,
        },
        "coupled_update": {
            "step_index": step_index,
            "sample": sample,
            "video_velocity": video_velocity,
            "audio_velocity": audio_velocity,
            "video_next": euler_update(
                sample,
                video_velocity,
                video_sigmas[step_index],
                video_sigmas[step_index + 1],
            ),
            "audio_next": euler_update(
                sample,
                audio_velocity,
                audio_sigmas[step_index],
                audio_sigmas[step_index + 1],
            ),
            "velocity_sign": "data-ward-plus",
            "update_dtype": "float32",
        },
        "packed_layout": {
            "order": [
                "text",
                "reference-audio",
                "reference-video",
                "target-audio",
                "target-video",
            ],
            "row_counts": [3, 4, 6, 8, 10],
            "row_offsets": [0, 3, 7, 13, 21, 31],
            "modality_tags": {
                "video": 0,
                "text": 1,
                "audio": 2,
            },
            "video_soundtrack_precedes_visual_rows": True,
        },
        "temporal_rope": {
            "origin": 3.0,
            "frame_rescale": 5.0 / 3.0,
            "frames_per_latent_pattern": [1, 4, 4, 4, 4],
            "positions_for_seven_latents": temporal_positions,
        },
        "noise_allocation_order": [
            {"domain": "condition-posterior", "shape": [1, 24, 1, 2, 2]},
            {"domain": "condition-noise", "shape": [1, 24, 1, 2, 2]},
            {"domain": "target-video", "shape": [1, 24, 2, 2, 2]},
            {"domain": "target-audio", "shape": [2, 3, 32]},
        ],
    }


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_manifest() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    validate_schema(manifest, MANIFEST_SCHEMA_PATH)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail("unexpected conformance manifest schema version")

    revisions = {source["id"]: source["revision"] for source in manifest["sources"]}
    if len(manifest["sources"]) != len(EXPECTED_REVISIONS) or revisions != EXPECTED_REVISIONS:
        fail("source revisions drifted from the reviewed H3 authority set")

    excluded = set(manifest["numerical_authority"]["excluded_accelerations"])
    if (
        len(manifest["numerical_authority"]["excluded_accelerations"])
        != len(EXPECTED_EXCLUDED_ACCELERATIONS)
        or excluded != EXPECTED_EXCLUDED_ACCELERATIONS
    ):
        fail("ground-truth acceleration exclusions drifted")

    indexes = {
        item["id"]: (item["relative_path"], item["sha256"])
        for item in manifest["component_indexes"]
    }
    if (
        len(manifest["component_indexes"]) != len(EXPECTED_COMPONENT_INDEXES)
        or indexes != EXPECTED_COMPONENT_INDEXES
    ):
        fail("official component index fingerprints drifted")

    layers = {layer["id"] for layer in manifest["fixture_layers"]}
    if len(manifest["fixture_layers"]) != len(EXPECTED_LAYERS) or layers != EXPECTED_LAYERS:
        fail("fixture layer coverage is incomplete or contains an unreviewed layer")

    policy = manifest["capture_policy"]
    if not all(
        policy[field]
        for field in (
            "external_fixture_root_required",
            "authorization_record_required",
            "weights_in_repository_forbidden",
            "generated_media_in_repository_forbidden",
        )
    ):
        fail("real H3 capture policy is not fail closed")

    expected_synthetic = synthetic_fixture()
    checked_synthetic = load_json(SYNTHETIC_PATH)
    if checked_synthetic != expected_synthetic:
        fail("synthetic fixture does not match the deterministic generator")
    synthetic = manifest["synthetic_fixture"]
    if synthetic["relative_path"] != "synthetic-v1.json":
        fail("synthetic fixture path is not repository-relative and fixed")
    if synthetic["sha256"] != sha256_file(SYNTHETIC_PATH):
        fail("synthetic fixture hash drifted")

    tracked = []
    try:
        output = subprocess.run(
            ["git", "ls-files", "-z", "--", str(FIXTURE_DIR.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        tracked = [REPO_ROOT / path.decode() for path in output.split(b"\0") if path]
    except (OSError, subprocess.CalledProcessError):
        tracked = [path for path in FIXTURE_DIR.rglob("*") if path.is_file()]
    forbidden = [path for path in tracked if path.suffix.lower() in FORBIDDEN_FIXTURE_SUFFIXES]
    if forbidden:
        fail(f"restricted model/media artifacts are present in the fixture tree: {forbidden}")
    return manifest


def canonical_external_directory(label: str, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        fail(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        fail(f"cannot resolve {label}: {error}")
    if not resolved.is_dir():
        fail(f"{label} is not a directory")
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        fail(f"{label} must live outside the Mold repository")
    return resolved


def validate_authorization(record_path: pathlib.Path) -> dict[str, Any]:
    if not record_path.is_absolute():
        fail("authorization record path must be absolute")
    record = load_json(record_path)
    required = {
        "schema_version",
        "family",
        "decision",
        "license_revision",
        "license_sha256",
        "approved_scopes",
        "source_document_path",
        "source_document_sha256",
        "review_reference",
    }
    if not isinstance(record, dict) or set(record) != required:
        fail("authorization record fields are incomplete or unreviewed")
    if record["schema_version"] != AUTHORIZATION_SCHEMA_VERSION:
        fail("authorization record schema is unsupported")
    if record["family"] != "minimax-h3" or record["decision"] != "approved":
        fail("authorization record does not approve MiniMax H3")
    if record["license_revision"] != EXPECTED_REVISIONS["minimax-official-model"]:
        fail("authorization record covers a different H3 license revision")
    if record["license_sha256"] != EXPECTED_LICENSE_SHA256:
        fail("authorization record covers different H3 license bytes")
    scopes = record["approved_scopes"]
    if not isinstance(scopes, list) or not {
        "checkpoint-execution",
        "fixture-capture",
        "generated-output-retention",
    }.issubset(scopes):
        fail("authorization record does not cover every conformance activity")
    source_document = pathlib.Path(record["source_document_path"])
    if not source_document.is_absolute() or not source_document.is_file():
        fail("authorization source document must be an existing absolute file")
    try:
        source_document.resolve().relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        fail("authorization source document must not be committed to the repository")
    if sha256_file(source_document) != record["source_document_sha256"]:
        fail("authorization source document hash does not match its record")
    if not isinstance(record["review_reference"], str) or not record["review_reference"].strip():
        fail("authorization record lacks an external review reference")
    return record


def parse_sources(values: list[str]) -> dict[str, pathlib.Path]:
    sources: dict[str, pathlib.Path] = {}
    for value in values:
        identifier, separator, raw_path = value.partition("=")
        if not separator or identifier in sources:
            fail(f"--source must be a unique id=/absolute/path pair, got {value!r}")
        sources[identifier] = pathlib.Path(raw_path)
    return sources


def verify_source_checkouts(manifest: dict[str, Any], values: list[str]) -> None:
    sources = parse_sources(values)
    if set(sources) != set(EXPECTED_REVISIONS):
        missing = sorted(set(EXPECTED_REVISIONS) - set(sources))
        extra = sorted(set(sources) - set(EXPECTED_REVISIONS))
        fail(f"source checkout set differs; missing={missing}, extra={extra}")
    for identifier, expected_revision in EXPECTED_REVISIONS.items():
        root = sources[identifier]
        if not root.is_absolute() or not root.is_dir():
            fail(f"source {identifier} is not an existing absolute directory")
        try:
            observed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            fail(f"cannot read source revision for {identifier}: {error}")
        if observed != expected_revision:
            fail(f"source {identifier} is {observed}, expected {expected_revision}")

    model_root = sources["minimax-official-model"]
    for item in manifest["component_indexes"]:
        path = model_root / item["relative_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            fail(f"component metadata drifted: {item['id']} ({path})")


def validate_bundle(
    manifest: dict[str, Any], fixture_root_value: str, bundle_value: str, authorization_value: str
) -> None:
    fixture_root = canonical_external_directory("fixture root", fixture_root_value)
    authorization_path = pathlib.Path(authorization_value)
    authorization = validate_authorization(authorization_path)
    bundle_path = pathlib.Path(bundle_value).resolve(strict=True)
    try:
        bundle_path.relative_to(fixture_root)
    except ValueError:
        fail("fixture bundle must be inside the approved external fixture root")
    bundle = load_json(bundle_path)
    validate_schema(bundle, BUNDLE_SCHEMA_PATH)
    if bundle["schema_version"] != BUNDLE_SCHEMA_VERSION:
        fail("external fixture bundle schema is unsupported")
    if bundle["manifest_sha256"] != sha256_file(MANIFEST_PATH):
        fail("external fixture bundle targets a different conformance manifest")
    if bundle["authorization_document_sha256"] != authorization["source_document_sha256"]:
        fail("external fixture bundle is not bound to the authorization evidence")
    fixture_layers = {fixture["layer"] for fixture in bundle["fixtures"]}
    unknown_layers = fixture_layers - EXPECTED_LAYERS
    if unknown_layers:
        fail(f"external fixture bundle contains unknown layers: {sorted(unknown_layers)}")
    if not bundle["fixtures"]:
        fail("external fixture bundle contains no evidence")
    for fixture in bundle["fixtures"]:
        evidence_path = (fixture_root / fixture["relative_path"]).resolve(strict=True)
        try:
            evidence_path.relative_to(fixture_root)
        except ValueError:
            fail(f"fixture evidence escapes its external root: {fixture['id']}")
        if not evidence_path.is_file() or sha256_file(evidence_path) != fixture["sha256"]:
            fail(f"fixture evidence hash mismatch: {fixture['id']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the weight-free H3 contract, pinned checkouts, or a separately "
            "authorized external fixture bundle."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-contract", help="validate checked-in pins and synthetic fixtures")

    verify = subparsers.add_parser("verify-sources", help="verify every pinned local source checkout")
    verify.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ID=/ABSOLUTE/PATH",
        help="repeat once for every source id in the conformance manifest",
    )

    bundle = subparsers.add_parser(
        "validate-bundle", help="validate authorized evidence outside the repository"
    )
    bundle.add_argument("--fixture-root", required=True)
    bundle.add_argument("--bundle", required=True)
    bundle.add_argument("--authorization-record", required=True)

    subparsers.add_parser("print-synthetic", help="print the deterministic synthetic fixture")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if args.command == "print-synthetic":
            sys.stdout.buffer.write(canonical_bytes(synthetic_fixture()))
            return 0
        manifest = validate_manifest()
        if args.command == "verify-sources":
            verify_source_checkouts(manifest, args.source)
        elif args.command == "validate-bundle":
            validate_bundle(
                manifest,
                args.fixture_root,
                args.bundle,
                args.authorization_record,
            )
        print(f"MiniMax H3 conformance {args.command} passed")
        return 0
    except (ConformanceFailure, OSError) as error:
        print(f"invalid MiniMax H3 conformance evidence: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
