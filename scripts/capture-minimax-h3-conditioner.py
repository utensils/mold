#!/usr/bin/env python3
"""Capture authorization-bound MiniMax H3 conditioner primitive evidence.

This tool is intentionally separate from every shipped Mold runtime. It opens
only the public tokenizer and processor metadata named by the checked
conformance manifest, executes a fixed synthetic-media case, and writes a
schema-valid oracle layer plus a partial bundle under an approved external
fixture root. It never opens checkpoint shards or emits evidence in the
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFORMANCE_TOOL_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
GPU_CONFORMANCE_TOOL_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"
MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "minimax_h3" / "conformance-manifest.json"
)

DIFFUSERS_REVISION = "9c6a68c32b3b2a64db91800b624d33cec6e25ab8"
TRANSFORMERS_REVISION = "42f189ded85d18d00b51161d694cafd325e32b91"
MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
REVIEWED_AUTHORIZATION_EVIDENCE_SHA256 = (
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d"
)
CASE_ID = "conditioner-primitives-v1"
INPUT_SCHEMA = "mold.minimax-h3.tokenizer-processor-input.v1"
TENSOR_HASH_ENCODING = "canonical-typed-le-v1"
COMPONENT_AUTHORITY_SET_SCHEMA = "mold.minimax-h3.component-authority-set.v1"
CAPTURE_COMMAND_PREFIX = (
    "python3 scripts/capture-minimax-h3-conditioner.py tokenizer-processor"
)
REQUIRED_ENVIRONMENT = (
    "MOLD_H3_FIXTURE_ROOT",
    "MOLD_H3_AUTHORIZATION_RECORD",
    "MOLD_H3_OFFICIAL_MODEL",
    "MOLD_H3_DIFFUSERS_CHECKOUT",
    "MOLD_H3_TRANSFORMERS_CHECKOUT",
)
REQUIRED_COMPONENT_IDS = (
    "official-model-index",
    "official-modular-index",
    "official-tokenizer-json",
    "official-tokenizer-config",
    "official-tokenizer-merges",
    "official-tokenizer-vocab",
    "official-processor-config",
    "official-processor-video-config",
    "official-processor-chat-template",
    "official-processor-tokenizer-json",
    "official-processor-tokenizer-config",
    "official-processor-merges",
    "official-processor-vocab",
)
EXCLUDED_ACCELERATION_ENV_ALIASES = (
    "torchcompile",
    "torchinductor",
    "inductor",
    "fp8",
    "float8",
    "e4m3",
    "e5m2",
    "cachedit",
    "sageattention",
    "sageattn",
    "stochasticsampling",
    "approximateattention",
    "approxattn",
    "tf32",
    "tensorfloat32",
)
FALSE_ENVIRONMENT_VALUES = frozenset({"", "0", "false", "no", "off", "disabled"})
CUDA_DEVICE_PATTERN = re.compile(r"^cuda:([0-9]+)$")


class CaptureFailure(Exception):
    """A fail-closed conditioner capture contract violation."""


def fail(message: str) -> None:
    raise CaptureFailure(message)


def load_repo_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot load repository module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        fail(f"capture evidence is not canonical JSON: {error}")


def document_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        fail(f"capture document is not finite JSON: {error}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        fail(f"cannot hash required capture input: {error}")
    return digest.hexdigest()


def is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def external_directory(label: str, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        fail(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        fail(f"{label} is unavailable")
    if not resolved.is_dir():
        fail(f"{label} is not a directory")
    if is_relative_to(resolved, REPO_ROOT):
        fail(f"{label} must live outside the Mold repository")
    return resolved


def external_file(label: str, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        fail(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        fail(f"{label} is unavailable")
    if not resolved.is_file():
        fail(f"{label} is not a file")
    if is_relative_to(resolved, REPO_ROOT):
        fail(f"{label} must live outside the Mold repository")
    return resolved


def command_output(arguments: Sequence[str], cwd: pathlib.Path) -> str:
    try:
        result = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        fail("capture source checkout probe failed")
    if result.returncode != 0:
        fail("capture source checkout probe failed")
    return result.stdout.strip()


def normalized_repository_url(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.removesuffix("/")


def verify_git_checkout(
    label: str,
    root: pathlib.Path,
    expected_revision: str,
    expected_repository: str,
) -> None:
    observed = command_output(["git", "rev-parse", "HEAD"], root)
    if observed != expected_revision:
        fail(f"{label} revision differs from the reviewed capture pin")
    repository = normalized_repository_url(
        command_output(["git", "remote", "get-url", "origin"], root)
    )
    if repository.removesuffix(".git") != expected_repository.removesuffix(".git"):
        fail(f"{label} repository identity differs from the reviewed capture pin")
    status = command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], root
    )
    if status:
        fail(f"{label} checkout must be clean before capture")


def manifest_component_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    components: dict[str, dict[str, str]] = {}
    for raw in manifest["component_indexes"]:
        identifier = raw["id"]
        if identifier in components:
            fail("checked manifest repeats component identities")
        components[identifier] = {
            "id": identifier,
            "relative_path": raw["relative_path"],
            "sha256": raw["sha256"],
        }
    return components


def huggingface_metadata_revision(model_root: pathlib.Path, relative_path: str) -> str:
    metadata = (
        model_root / ".cache" / "huggingface" / "download" / f"{relative_path}.metadata"
    )
    try:
        first_line = metadata.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError, UnicodeError):
        fail("official model snapshot lacks revision-bound component metadata")
    return first_line


def verify_model_components(
    model_root: pathlib.Path, manifest: Mapping[str, Any]
) -> list[dict[str, str]]:
    component_map = manifest_component_map(manifest)
    if set(REQUIRED_COMPONENT_IDS) - set(component_map):
        fail("checked manifest lacks conditioner component authority")
    verified: list[dict[str, str]] = []
    for identifier in REQUIRED_COMPONENT_IDS:
        component = component_map[identifier]
        relative_path = component["relative_path"]
        candidate = model_root / relative_path
        if candidate.is_symlink():
            fail("official model component may not be a symbolic link")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            fail(f"official model component is unavailable: {identifier}")
        if not resolved.is_file() or not is_relative_to(resolved, model_root):
            fail(f"official model component escapes the snapshot: {identifier}")
        if sha256_file(resolved) != component["sha256"]:
            fail(f"official model component hash drifted: {identifier}")
        if huggingface_metadata_revision(model_root, relative_path) != MODEL_REVISION:
            fail(f"official model component revision drifted: {identifier}")
        verified.append({"id": identifier, "sha256": component["sha256"]})
    return verified


def normalize_acceleration_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def reject_excluded_acceleration_environment(environment: Mapping[str, str]) -> None:
    for key, value in environment.items():
        normalized_key = normalize_acceleration_name(key)
        normalized_value = value.strip().lower()
        if normalized_value in FALSE_ENVIRONMENT_VALUES:
            continue
        if any(alias in normalized_key for alias in EXCLUDED_ACCELERATION_ENV_ALIASES):
            fail("capture environment enables or configures an excluded acceleration")


def component_authority_set_sha256(
    component_ids: Iterable[str], component_map: Mapping[str, Mapping[str, str]]
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": COMPONENT_AUTHORITY_SET_SCHEMA,
                "components": [
                    {
                        "id": identifier,
                        "sha256": component_map[identifier]["sha256"],
                    }
                    for identifier in sorted(component_ids)
                ],
            }
        )
    )


def bfloat16_bits_to_float(bits: int) -> float:
    if not 0 <= bits <= 0xFFFF:
        fail("bfloat16 capture value is outside its encoded domain")
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def sample_indexes(length: int) -> list[int]:
    if length <= 0:
        fail("capture tensors must be non-empty")
    return sorted({0, length // 4, length // 2, (3 * length) // 4, length - 1})


def finite_statistics(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        fail("capture tensors must be non-empty")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        fail("capture tensor contains a non-finite value")
    mean = math.fsum(converted) / len(converted)
    variance = math.fsum((value - mean) ** 2 for value in converted) / len(converted)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(variance),
    }


def integer_tensor_record(key: str, values: Sequence[int]) -> dict[str, Any]:
    signed_values = [int(value) for value in values]
    if not signed_values or any(
        not -(2**63) <= value <= 2**63 - 1 for value in signed_values
    ):
        fail(f"{key} evidence is outside signed int64")
    digest = hashlib.sha256()
    for value in signed_values:
        digest.update(struct.pack("<q", value))
    return {
        "key": key,
        "shape": [len(signed_values)],
        "dtype": "int64",
        "content_sha256": digest.hexdigest(),
        "statistics": finite_statistics(signed_values),
        "samples": [
            {"index": [index], "value": signed_values[index]}
            for index in sample_indexes(len(signed_values))
        ],
    }


def bfloat16_tensor_record(key: str, bits: Sequence[int]) -> dict[str, Any]:
    encoded = [int(value) for value in bits]
    if not encoded:
        fail(f"{key} evidence is empty")
    digest = hashlib.sha256()
    for value in encoded:
        if not 0 <= value <= 0xFFFF:
            fail(f"{key} evidence contains an invalid bfloat16 encoding")
        digest.update(struct.pack("<H", value))
    values = [bfloat16_bits_to_float(value) for value in encoded]
    if not all(math.isfinite(value) for value in values):
        fail(f"{key} evidence contains a non-finite bfloat16")
    return {
        "key": key,
        "shape": [len(values)],
        "dtype": "bfloat16",
        "content_sha256": digest.hexdigest(),
        "statistics": finite_statistics(values),
        "samples": [
            {"index": [index], "value": values[index]}
            for index in sample_indexes(len(values))
        ],
    }


def packed_vectors(vectors: Sequence[Sequence[int]]) -> list[int]:
    result = [len(vectors)]
    for vector in vectors:
        result.append(len(vector))
        result.extend(int(value) for value in vector)
    return result


def packed_shapes(named_shapes: Sequence[tuple[int, Sequence[int]]]) -> list[int]:
    result = [len(named_shapes)]
    seen: set[int] = set()
    for identifier, shape in named_shapes:
        if identifier in seen:
            fail("processor shape evidence repeats an identity")
        seen.add(identifier)
        dims = [int(value) for value in shape]
        if not dims or any(value <= 0 for value in dims):
            fail("processor shape evidence has an invalid dimension")
        result.extend([identifier, len(dims), *dims])
    return result


def acceleration_policy(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    disabled = list(manifest["numerical_authority"]["excluded_accelerations"])
    if len(disabled) != len(set(disabled)):
        fail("checked manifest repeats acceleration exclusions")
    return {
        "enabled": [
            "cuda",
            "deterministic-cuda-copy",
            "pytorch-bfloat16",
            "qwen3-vl-processor",
        ],
        "disabled": disabled,
    }


@dataclass(frozen=True)
class PrimitiveEvidence:
    token_ids: tuple[int, ...]
    special_presentation: tuple[int, ...]
    processor_shapes: tuple[int, ...]
    sampled_bfloat16_bits: tuple[int, ...]
    prompt_sha256: str
    image_sha256: str
    video_sha256: str
    sampled_source_indices: tuple[int, ...]
    sampled_timestamps_millis: tuple[int, ...]


def build_input_descriptor(evidence: PrimitiveEvidence) -> dict[str, Any]:
    return {
        "schema_version": INPUT_SCHEMA,
        "case_id": CASE_ID,
        "source_revisions": {
            "diffusers": DIFFUSERS_REVISION,
            "transformers": TRANSFORMERS_REVISION,
            "official_model": MODEL_REVISION,
        },
        "prompt": {
            "encoding": "utf-8",
            "sha256": evidence.prompt_sha256,
            "special_tokens_added": False,
            "chat_template_applied": False,
        },
        "image": {
            "generator": "mold-h3-rgb-pattern-v1",
            "shape": [256, 256, 3],
            "dtype": "uint8",
            "sha256": evidence.image_sha256,
        },
        "video": {
            "generator": "mold-h3-rgb-pattern-v1",
            "shape": [37, 64, 64, 3],
            "dtype": "uint8",
            "fps": 24,
            "sample_fps": 2,
            "sha256": evidence.video_sha256,
            "sampled_source_indices": list(evidence.sampled_source_indices),
            "sampled_timestamps_millis": list(evidence.sampled_timestamps_millis),
        },
    }


def layer_contract(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        layer
        for layer in manifest["fixture_layers"]
        if layer["id"] == "tokenizer-processor"
    ]
    if len(matches) != 1:
        fail("checked manifest does not contain one tokenizer/processor layer")
    layer = matches[0]
    if tuple(layer["required_component_indexes"]) != REQUIRED_COMPONENT_IDS:
        fail("tokenizer/processor component authority drifted")
    if tuple(layer["required_measurements"]) != (
        "token-ids",
        "special-token-presentation",
        "processor-shapes",
        "sampled-values",
    ):
        fail("tokenizer/processor measurement contract drifted")
    return layer


def build_layer_document(
    manifest: Mapping[str, Any],
    evidence: PrimitiveEvidence,
    authorization_sha256: str,
    device: str,
) -> dict[str, Any]:
    contract = layer_contract(manifest)
    component_map = manifest_component_map(manifest)
    components = [
        {"id": identifier, "sha256": component_map[identifier]["sha256"]}
        for identifier in REQUIRED_COMPONENT_IDS
    ]
    tokenizer_ids = (
        "official-tokenizer-json",
        "official-tokenizer-config",
        "official-tokenizer-merges",
        "official-tokenizer-vocab",
    )
    processor_ids = (
        "official-processor-config",
        "official-processor-video-config",
        "official-processor-chat-template",
        "official-processor-tokenizer-json",
        "official-processor-tokenizer-config",
        "official-processor-merges",
        "official-processor-vocab",
    )
    command = f"{CAPTURE_COMMAND_PREFIX} --device {device}"
    input_descriptor = build_input_descriptor(evidence)
    policy = acceleration_policy(manifest)
    outputs = [
        integer_tensor_record("token-ids", evidence.token_ids),
        integer_tensor_record(
            "special-token-presentation", evidence.special_presentation
        ),
        integer_tensor_record("processor-shapes", evidence.processor_shapes),
        bfloat16_tensor_record("sampled-values", evidence.sampled_bfloat16_bits),
    ]
    comparison = [
        {
            "key": key,
            "absolute": 0,
            "relative": 0,
            "metric": "elementwise-atol-plus-rtol",
            "hash_policy": "exact",
        }
        for key in contract["required_measurements"]
    ]
    return {
        "schema_version": "mold.minimax-h3.layer-output.v1",
        "family": "minimax-h3",
        "case_id": CASE_ID,
        "layer": "tokenizer-processor",
        "authority_tier": "exact-full-bf16",
        "authorization_document_sha256": authorization_sha256,
        "input": {
            "id": CASE_ID,
            "sha256": sha256_bytes(canonical_json_bytes(input_descriptor)),
            "component_index_sha256": components[0]["sha256"],
            "component_indexes": components,
        },
        "producer": {
            "role": "oracle",
            "implementation": (
                "diffusers-minimax-h3-transformers-42f189de-tokenizer-processor"
            ),
            "source_id": "diffusers",
            "revision": DIFFUSERS_REVISION,
        },
        "adapter": {
            "schema_version": "mold.minimax-h3.layer-adapter.v1",
            "id": "minimax-h3-tokenizer-processor-capture-v1",
            "command": command,
            "tensor_hash_encoding": TENSOR_HASH_ENCODING,
        },
        "environment": {
            "device": device,
            "dtype": "bfloat16",
            "attention_backend": "none-conditioner-preprocessing",
            "acceleration_policy": policy,
            "forbidden_accelerations_disabled": True,
        },
        "provenance": [
            {"key": "source-revision", "value": DIFFUSERS_REVISION},
            {
                "key": "tokenizer-hash",
                "value": component_authority_set_sha256(tokenizer_ids, component_map),
            },
            {
                "key": "processor-hash",
                "value": component_authority_set_sha256(processor_ids, component_map),
            },
            {"key": "device", "value": device},
            {"key": "capture-command", "value": command},
        ],
        "outputs": outputs,
        "comparison": comparison,
    }


def expected_tensor_summary(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shape": output["shape"],
        "dtype": output["dtype"],
        "min": output["statistics"]["min"],
        "max": output["statistics"]["max"],
        "mean": output["statistics"]["mean"],
        "std": output["statistics"]["std"],
        "sampled_values": [sample["value"] for sample in output["samples"]],
    }


def build_bundle(
    manifest: Mapping[str, Any],
    layer_document: Mapping[str, Any],
    layer_encoded: bytes,
    relative_layer_path: str,
    authorization_sha256: str,
    device: str,
) -> dict[str, Any]:
    first_output = layer_document["outputs"][0]
    first_policy = layer_document["comparison"][0]
    return {
        "schema_version": "mold.minimax-h3.fixture-bundle.v1",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "authorization_document_sha256": authorization_sha256,
        "capture_environment": {
            "framework": "diffusers",
            "framework_revision": DIFFUSERS_REVISION,
            "device": device,
            "dtype": "bfloat16",
            "attention_backend": "none-conditioner-preprocessing",
            "acceleration_policy": acceleration_policy(manifest),
            "command": layer_document["adapter"]["command"],
            "forbidden_accelerations_disabled": True,
        },
        "fixtures": [
            {
                "id": f"oracle-{CASE_ID}-tokenizer-processor",
                "layer": "tokenizer-processor",
                "authority_tier": "exact-full-bf16",
                "relative_path": relative_layer_path,
                "sha256": sha256_bytes(layer_encoded),
                "component_index_sha256": layer_document["input"][
                    "component_index_sha256"
                ],
                "component_indexes": layer_document["input"]["component_indexes"],
                "tensor": expected_tensor_summary(first_output),
                "tolerance": {
                    "absolute": first_policy["absolute"],
                    "relative": first_policy["relative"],
                    "metric": first_policy["metric"],
                },
            }
        ],
    }


def verify_module_origin(
    label: str, module_or_class: Any, checkout: pathlib.Path
) -> None:
    try:
        source = pathlib.Path(inspect.getfile(module_or_class)).resolve(strict=True)
    except (OSError, TypeError):
        fail(f"cannot bind {label} to its reviewed checkout")
    if not is_relative_to(source, checkout):
        fail(f"{label} imported outside its reviewed checkout")


def load_capture_runtime(
    diffusers_checkout: pathlib.Path, transformers_checkout: pathlib.Path
) -> SimpleNamespace:
    sys.dont_write_bytecode = True
    source_paths = [
        transformers_checkout / "src",
        diffusers_checkout / "src",
    ]
    if any(not path.is_dir() for path in source_paths):
        fail("capture checkout lacks its source package")
    for path in reversed(source_paths):
        sys.path.insert(0, str(path))
    for package in ("diffusers", "transformers"):
        if package in sys.modules:
            fail(f"{package} was imported before provenance was verified")
    try:
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
        transformers = importlib.import_module("transformers")
        diffusers = importlib.import_module("diffusers")
        qwen_module = importlib.import_module(
            "transformers.models.qwen3_vl.processing_qwen3_vl"
        )
        encoder_module = importlib.import_module(
            "diffusers.modular_pipelines.minimax_h3.encoders"
        )
    except (ImportError, RuntimeError) as error:
        fail(f"reviewed capture runtime cannot be imported: {error}")
    verify_module_origin("Transformers", transformers, transformers_checkout)
    verify_module_origin(
        "Qwen3-VL processor", qwen_module.Qwen3VLProcessor, transformers_checkout
    )
    verify_module_origin("Diffusers", diffusers, diffusers_checkout)
    verify_module_origin(
        "MiniMax H3 encoder",
        encoder_module.MiniMaxH3Ref2VATextEncoderStep,
        diffusers_checkout,
    )
    if not hasattr(qwen_module.Qwen3VLProcessor, "create_mm_token_type_ids"):
        fail("reviewed Transformers checkout lacks Qwen modality type authority")
    return SimpleNamespace(
        torch=torch,
        numpy=numpy,
        tokenizer_class=transformers.Qwen2TokenizerFast,
        processor_class=qwen_module.Qwen3VLProcessor,
        encoder_step=encoder_module.MiniMaxH3Ref2VATextEncoderStep,
        transformers_checkout=transformers_checkout,
    )


def configure_torch(runtime: SimpleNamespace, device: str) -> Any:
    torch = runtime.torch
    match = CUDA_DEVICE_PATTERN.fullmatch(device)
    if match is None:
        fail("capture device must be an indexed CUDA device such as cuda:0")
    if not torch.cuda.is_available():
        fail("CUDA is unavailable for protected capture")
    index = int(match.group(1))
    if index >= torch.cuda.device_count():
        fail("requested CUDA capture device is unavailable")
    torch.cuda.set_device(index)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    for name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_cudnn_sdp", False),
        ("enable_math_sdp", True),
    ):
        callback = getattr(torch.backends.cuda, name, None)
        if callback is not None:
            callback(enabled)
    if torch.backends.cuda.matmul.allow_tf32:
        fail("TF32 remained enabled after capture preflight")
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.allow_tf32:
        fail("cuDNN TF32 remained enabled after capture preflight")
    if not torch.are_deterministic_algorithms_enabled():
        fail("deterministic PyTorch execution remained disabled after preflight")
    for name, expected in (
        ("flash_sdp_enabled", False),
        ("mem_efficient_sdp_enabled", False),
        ("cudnn_sdp_enabled", False),
        ("math_sdp_enabled", True),
    ):
        probe = getattr(torch.backends.cuda, name, None)
        if probe is not None and bool(probe()) != expected:
            fail("PyTorch attention backend policy drifted after capture preflight")
    return torch.device(device)


def rgb_pattern(numpy: Any, frames: int, height: int, width: int) -> Any:
    frame_axis = numpy.arange(frames, dtype=numpy.uint16)[:, None, None]
    y_axis = numpy.arange(height, dtype=numpy.uint16)[None, :, None]
    x_axis = numpy.arange(width, dtype=numpy.uint16)[None, None, :]
    red = (3 * x_axis + 5 * y_axis + 19 * frame_axis) % 256
    green = (7 * x_axis + 11 * y_axis + 17 + 23 * frame_axis) % 256
    blue = (13 * x_axis + 17 * y_axis + 29 + 31 * frame_axis) % 256
    return numpy.stack((red, green, blue), axis=-1).astype(numpy.uint8)


def bfloat16_bits_from_tensor(torch: Any, tensor: Any, device: Any) -> list[int]:
    copied = tensor.detach().to(device=device, dtype=torch.bfloat16).contiguous()
    torch.cuda.synchronize(device)
    values = copied.to(device="cpu", dtype=torch.float32).reshape(-1).tolist()
    bits: list[int] = []
    for value in values:
        encoded = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        if encoded & 0xFFFF:
            fail("CUDA BF16 capture did not round-trip exactly")
        bits.append(encoded >> 16)
    return bits


def capture_primitives(
    runtime: SimpleNamespace, model_root: pathlib.Path, device: Any
) -> PrimitiveEvidence:
    torch = runtime.torch
    numpy = runtime.numpy
    tokenizer_root = model_root / "tokenizer"
    processor_root = model_root / "processor"
    try:
        tokenizer = runtime.tokenizer_class.from_pretrained(
            tokenizer_root, local_files_only=True
        )
        processor = runtime.processor_class.from_pretrained(
            processor_root, local_files_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        fail(f"pinned tokenizer/processor could not be loaded: {error}")
    verify_module_origin(
        "Qwen tokenizer", tokenizer.__class__, runtime.transformers_checkout
    )
    verify_module_origin(
        "Qwen processor tokenizer",
        processor.tokenizer.__class__,
        runtime.transformers_checkout,
    )
    verify_module_origin(
        "Qwen image processor",
        processor.image_processor.__class__,
        runtime.transformers_checkout,
    )
    verify_module_origin(
        "Qwen video processor",
        processor.video_processor.__class__,
        runtime.transformers_checkout,
    )

    prompt = "Raw prompt: coral robot, café, and literal <|vision_start|> bytes."
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    processor_prompt_ids = processor.tokenizer(prompt, add_special_tokens=False)[
        "input_ids"
    ]
    if prompt_ids != processor_prompt_ids:
        fail("processor tokenizer differs from the pinned standalone tokenizer")

    image = rgb_pattern(numpy, 1, 256, 256)[0]
    video = rgb_pattern(numpy, 37, 64, 64)
    temporal_patch = int(processor.video_processor.temporal_patch_size)
    sampled_frames, timestamps = runtime.encoder_step._sample_video_condition_frames(
        video, 24.0, 2.0, temporal_patch
    )
    expected_indices = (0, 12, 24, 36)
    if len(sampled_frames) != len(expected_indices) or any(
        not numpy.array_equal(frame, video[index])
        for frame, index in zip(sampled_frames, expected_indices)
    ):
        fail("pinned Diffusers video sampling authority drifted")
    timestamps_millis = tuple(round(value * 1000) for value in timestamps)
    if timestamps_millis != (250, 1250):
        fail("pinned Diffusers video timestamp authority drifted")

    try:
        image_features = processor.image_processor(images=[image], return_tensors="pt")
        video_features = processor.video_processor(
            videos=[numpy.stack(sampled_frames)],
            do_sample_frames=False,
            return_tensors="pt",
        )
    except (RuntimeError, TypeError, ValueError) as error:
        fail(f"pinned Qwen processor execution failed: {error}")

    merge_size = int(processor.image_processor.merge_size) ** 2
    image_grid = image_features["image_grid_thw"]
    video_grid = video_features["video_grid_thw"]
    image_tokens = int(image_grid[0].prod().item()) // merge_size
    video_tokens = (
        int(video_grid[0][1].item()) * int(video_grid[0][2].item()) // merge_size
    )
    if int(video_grid[0][0].item()) != len(timestamps):
        fail("pinned Qwen video processor changed temporal block geometry")

    references = [
        SimpleNamespace(kind="image", has_audio=False),
        SimpleNamespace(kind="video", has_audio=False),
    ]
    presentation_ids, h3_tags = runtime.encoder_step._build_presentation(
        tokenizer,
        prompt,
        references,
        [image_tokens],
        [video_tokens],
        [list(timestamps)],
    )
    mm_type_ids = processor.create_mm_token_type_ids([presentation_ids])[0]
    if len(mm_type_ids) != len(presentation_ids) or len(h3_tags) != len(
        presentation_ids
    ):
        fail("pinned presentation modality evidence has inconsistent length")

    special_ids = [
        tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        tokenizer.convert_tokens_to_ids("<|video_pad|>"),
    ]
    if any(value is None or value < 0 for value in special_ids):
        fail("pinned tokenizer lacks required H3 special tokens")

    image_pixels = image_features["pixel_values"]
    video_pixels = video_features["pixel_values_videos"]
    sampled_bits = [
        *bfloat16_bits_from_tensor(torch, image_pixels, device),
        *bfloat16_bits_from_tensor(torch, video_pixels, device),
    ]
    processor_shapes = packed_shapes(
        (
            (1, image.shape),
            (2, video.shape),
            (3, (len(sampled_frames), *sampled_frames[0].shape)),
            (4, tuple(image_pixels.shape)),
            (5, tuple(image_grid.shape)),
            (6, tuple(video_pixels.shape)),
            (7, tuple(video_grid.shape)),
        )
    )
    special_presentation = packed_vectors(
        (
            special_ids,
            h3_tags,
            mm_type_ids,
            list(expected_indices),
            list(timestamps_millis),
        )
    )
    return PrimitiveEvidence(
        token_ids=tuple(packed_vectors((prompt_ids, presentation_ids))),
        special_presentation=tuple(special_presentation),
        processor_shapes=tuple(processor_shapes),
        sampled_bfloat16_bits=tuple(sampled_bits),
        prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
        image_sha256=sha256_bytes(image.tobytes(order="C")),
        video_sha256=sha256_bytes(video.tobytes(order="C")),
        sampled_source_indices=expected_indices,
        sampled_timestamps_millis=timestamps_millis,
    )


def exclusive_write(path: pathlib.Path, encoded: bytes) -> None:
    temporary: pathlib.Path | None = None
    try:
        descriptor, temporary_value = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = pathlib.Path(temporary_value)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
    except FileExistsError:
        fail("capture output already exists; refusing to overwrite evidence")
    except OSError as error:
        fail(f"cannot commit external capture evidence: {error}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_capture(
    fixture_root: pathlib.Path,
    authorization_path: pathlib.Path,
    manifest: Mapping[str, Any],
    layer_document: Mapping[str, Any],
    device: str,
    conformance_tool: Any,
    gpu_tool: Any,
) -> tuple[pathlib.Path, pathlib.Path]:
    contracts = gpu_tool.exact_layer_contracts(manifest)
    gpu_tool.validate_manifest_layer_evidence(
        layer_document, contracts["tokenizer-processor"], "oracle"
    )
    conformance_tool.validate_schema(
        layer_document, conformance_tool.LAYER_OUTPUT_SCHEMA_PATH
    )
    layer_encoded = document_bytes(layer_document)
    capture_name = f"{CASE_ID}-{layer_document['input']['sha256'][:16]}"
    parent = fixture_root / "captures" / "tokenizer-processor"
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved_parent = parent.resolve(strict=True)
    if not is_relative_to(resolved_parent, fixture_root):
        fail("capture destination escapes MOLD_H3_FIXTURE_ROOT")
    capture_dir = parent / capture_name
    try:
        capture_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        fail("capture case already exists; refusing to overwrite evidence")
    except OSError as error:
        fail(f"cannot create external capture directory: {error}")
    layer_path = capture_dir / "oracle.json"
    bundle_path = capture_dir / "oracle-bundle.json"
    try:
        resolved_capture = capture_dir.resolve(strict=True)
        if not is_relative_to(resolved_capture, fixture_root):
            fail("capture directory escapes MOLD_H3_FIXTURE_ROOT")
        relative_layer_path = layer_path.relative_to(fixture_root).as_posix()
        bundle = build_bundle(
            manifest,
            layer_document,
            layer_encoded,
            relative_layer_path,
            layer_document["authorization_document_sha256"],
            device,
        )
        conformance_tool.validate_schema(bundle, conformance_tool.BUNDLE_SCHEMA_PATH)
        bundle_encoded = document_bytes(bundle)
        exclusive_write(layer_path, layer_encoded)
        exclusive_write(bundle_path, bundle_encoded)
        conformance_tool.validate_bundle(
            manifest,
            str(fixture_root),
            str(bundle_path),
            str(authorization_path),
        )
    except BaseException:
        shutil.rmtree(capture_dir, ignore_errors=True)
        raise
    return layer_path, bundle_path


def required_environment(environment: Mapping[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in REQUIRED_ENVIRONMENT:
        value = environment.get(name, "").strip()
        if not value:
            fail(f"{name} is required")
        values[name] = value
    return values


def validate_reviewed_authorization(
    conformance_tool: Any, authorization_path: pathlib.Path
) -> dict[str, Any]:
    authorization = conformance_tool.validate_authorization(authorization_path)
    if (
        authorization["source_document_sha256"]
        != REVIEWED_AUTHORIZATION_EVIDENCE_SHA256
    ):
        fail("authorization record does not bind the reviewed evidence")
    external_file(
        "authorization source document", authorization["source_document_path"]
    )
    return authorization


def run_tokenizer_processor_capture(
    environment: Mapping[str, str], device: str
) -> tuple[pathlib.Path, pathlib.Path]:
    reject_excluded_acceleration_environment(environment)
    values = required_environment(environment)
    conformance_tool = load_repo_module(
        "minimax_h3_conformance_capture", CONFORMANCE_TOOL_PATH
    )
    gpu_tool = load_repo_module(
        "minimax_h3_gpu_conformance_capture", GPU_CONFORMANCE_TOOL_PATH
    )
    manifest = conformance_tool.validate_manifest()
    if conformance_tool.EXPECTED_REVISIONS["diffusers"] != DIFFUSERS_REVISION:
        fail("capture Diffusers revision differs from the checked conformance pin")
    if conformance_tool.EXPECTED_REVISIONS["minimax-official-model"] != MODEL_REVISION:
        fail("capture model revision differs from the checked conformance pin")

    fixture_root = external_directory("fixture root", values["MOLD_H3_FIXTURE_ROOT"])
    authorization_path = external_file(
        "authorization record", values["MOLD_H3_AUTHORIZATION_RECORD"]
    )
    authorization = validate_reviewed_authorization(
        conformance_tool, authorization_path
    )

    model_root = external_directory(
        "official model snapshot", values["MOLD_H3_OFFICIAL_MODEL"]
    )
    diffusers_checkout = external_directory(
        "Diffusers checkout", values["MOLD_H3_DIFFUSERS_CHECKOUT"]
    )
    transformers_checkout = external_directory(
        "Transformers checkout", values["MOLD_H3_TRANSFORMERS_CHECKOUT"]
    )
    verify_git_checkout(
        "Diffusers",
        diffusers_checkout,
        DIFFUSERS_REVISION,
        "https://github.com/huggingface/diffusers",
    )
    verify_git_checkout(
        "Transformers",
        transformers_checkout,
        TRANSFORMERS_REVISION,
        "https://github.com/huggingface/transformers",
    )
    verify_model_components(model_root, manifest)

    runtime = load_capture_runtime(diffusers_checkout, transformers_checkout)
    torch_device = configure_torch(runtime, device)
    evidence = capture_primitives(runtime, model_root, torch_device)
    layer_document = build_layer_document(
        manifest,
        evidence,
        authorization["source_document_sha256"],
        device,
    )
    return write_capture(
        fixture_root,
        authorization_path,
        manifest,
        layer_document,
        device,
        conformance_tool,
        gpu_tool,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture authorization-bound external MiniMax H3 conditioner evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser(
        "tokenizer-processor",
        help="capture the pinned Diffusers tokenizer and multimodal processor case",
    )
    capture.add_argument(
        "--device",
        default=os.environ.get("MOLD_H3_CAPTURE_DEVICE", "cuda:0"),
        help="indexed CUDA device; defaults to MOLD_H3_CAPTURE_DEVICE or cuda:0",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command != "tokenizer-processor":
            fail("unsupported capture command")
        layer_path, bundle_path = run_tokenizer_processor_capture(
            os.environ, args.device
        )
        print(
            "MiniMax H3 conditioner capture complete: "
            f"layer={layer_path.name} bundle={bundle_path.name}"
        )
        return 0
    except (CaptureFailure, OSError, subprocess.SubprocessError) as error:
        print(f"MiniMax H3 conditioner capture rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
