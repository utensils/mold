#!/usr/bin/env python3
"""Capture paired MiniMax H3 token-refiner and first-transformer-block evidence."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-conditioner.py"
QWEN_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-qwen-layer50.py"
CONFORMANCE_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
GPU_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"
MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures" / "minimax_h3" / "conformance-manifest.json"
BASE_SHA256 = "006c761d6217430c78dcfc0b6c3a6a7d714924584a8fa79ea25407f82a876eda"
QWEN_SHA256 = "9801c718b88a852f928cdbbb0e520f47d6fb9728f03677261f0b14eaade30898"
DIFFUSERS_REVISION = "9c6a68c32b3b2a64db91800b624d33cec6e25ab8"
TRANSFORMERS_REVISION = "42f189ded85d18d00b51161d694cafd325e32b91"
MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
AUTHORIZATION_SHA256 = "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d"
RAW_SCHEMA = "mold.minimax-h3.transformer-raw-output.v1"
CAPTURE_MARKER = "mold.minimax-h3.private-uat-transformer-capture.v1"
AUTHORITY_SET_SHA256 = "b6a4e455ea28a0acbd3b4e2180bda495bb35b32e98da131682590a254148135b"
INPUT_DOMAIN = b"mold.minimax-h3.transformer-input.v1\0"
MOLD_BINARY = "h3_transformer_capture"
MOLD_FEATURES = "dev-bins,h3-private-uat,cuda"
TASKS = ("fl2va", "ref2va")
LAYERS = ("token-refiner", "transformer-block")
RAW_KEYS = (
    "token-refiner", "normalized-q", "normalized-k", "rotated-q", "rotated-k",
    "adaln", "block-output", "video-head", "audio-head",
)
RAW_CONTRACT = {
    "token-refiner": ((1, 2, 5376), "bfloat16"),
    "normalized-q": ((1, 6, 56, 128), "bfloat16"),
    "normalized-k": ((1, 6, 56, 128), "bfloat16"),
    "rotated-q": ((1, 6, 56, 128), "bfloat16"),
    "rotated-k": ((1, 6, 56, 128), "bfloat16"),
    "adaln": ((6, 32256), "bfloat16"),
    "block-output": ((1, 6, 5376), "bfloat16"),
    "video-head": ((1, 2, 96), "float32"),
    "audio-head": ((1, 2, 32), "float32"),
}
MEASUREMENTS = {
    "token-refiner": ("shape", "statistics", "sampled-values", "tolerance"),
    "transformer-block": (
        "qk-rms", "partial-mm-rope", "adaln", "video-head", "audio-head", "tolerance"
    ),
}
RELATIVE_TOLERANCE = 1.0 / 64.0
ABSOLUTE_TOLERANCE = {
    "token-refiner": {
        "statistics": 1.0 / 64.0,
        "sampled-values": 8.0,
    },
    "transformer-block": {
        "qk-rms": 1.0 / 64.0,
        "partial-mm-rope": 1.0 / 16.0,
        "adaln": 1.0 / 64.0,
        "video-head": 1.25,
        "audio-head": 0.5,
    },
}
RECORD_ONLY_MEASUREMENTS = {
    layer: frozenset(measurements) for layer, measurements in ABSOLUTE_TOLERANCE.items()
}
MEASUREMENT_DTYPES = {
    "token-refiner": {
        "shape": "int64", "statistics": "float64", "sampled-values": "bfloat16", "tolerance": "float64"
    },
    "transformer-block": {
        "qk-rms": "float64", "partial-mm-rope": "bfloat16", "adaln": "bfloat16",
        "video-head": "float32", "audio-head": "float32", "tolerance": "float64",
    },
}
REQUIRED_ENVIRONMENT = (
    "MOLD_H3_FIXTURE_ROOT", "MOLD_H3_AUTHORIZATION_RECORD", "MOLD_H3_OFFICIAL_MODEL",
    "MOLD_H3_DIFFUSERS_CHECKOUT", "MOLD_H3_TRANSFORMERS_CHECKOUT",
)


class CaptureFailure(Exception):
    pass


def fail(message: str) -> None:
    raise CaptureFailure(message)


def load_module(name: str, path: pathlib.Path, expected: str | None = None) -> Any:
    if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        fail(f"repository module {name} differs from its reviewed dependency")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot load repository module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


BASE = load_module("minimax_h3_transformer_base", BASE_PATH, BASE_SHA256)
QWEN = load_module("minimax_h3_transformer_staging", QWEN_PATH, QWEN_SHA256)


@dataclass(frozen=True)
class TensorPayload:
    key: str
    shape: tuple[int, ...]
    dtype: str
    encoded: bytes

    def values(self) -> list[float | int]:
        if self.dtype == "bfloat16":
            return [BASE.bfloat16_bits_to_float(item[0]) for item in struct.iter_unpack("<H", self.encoded)]
        encoding = {"float32": "<f", "float64": "<d", "int64": "<q"}[self.dtype]
        return [item[0] for item in struct.iter_unpack(encoding, self.encoded)]


@dataclass
class AuthenticatedTransformerFile:
    relative_path: str
    sha256: str
    identity: Any
    metadata_identity: Any
    descriptor: int


class AuthenticatedTransformerSnapshot:
    def __init__(self, files: Sequence[AuthenticatedTransformerFile]):
        self.files = tuple(files)
        self.by_relative_path = {item.relative_path: item for item in files}

    def __enter__(self) -> "AuthenticatedTransformerSnapshot":
        return self

    def __exit__(self, *_: Any) -> None:
        for item in self.files:
            os.close(item.descriptor)

    def descriptor_path(self, relative_path: str) -> pathlib.Path:
        try:
            descriptor = self.by_relative_path[relative_path].descriptor
        except KeyError:
            fail(f"transformer checkpoint index selected unreviewed artifact {relative_path}")
        return pathlib.Path(f"/dev/fd/{descriptor}")

    def retained_bytes(self, relative_path: str, maximum_bytes: int) -> bytes:
        item = self.by_relative_path[relative_path]
        if item.identity.size > maximum_bytes:
            fail(f"transformer component {relative_path} exceeds its size bound")
        os.lseek(item.descriptor, 0, os.SEEK_SET)
        chunks = []
        while chunk := os.read(item.descriptor, 1024 * 1024):
            chunks.append(chunk)
        encoded = b"".join(chunks)
        if len(encoded) != item.identity.size:
            fail(f"transformer component {relative_path} changed while reading")
        return encoded

    def revalidate(self) -> None:
        for item in self.files:
            opened = QWEN.ExternalFileIdentity.from_metadata(
                item.identity.path, os.fstat(item.descriptor)
            )
            if (
                opened != item.identity
                or QWEN.ExternalFileIdentity.capture(item.identity.path) != item.identity
            ):
                fail(f"transformer component {item.relative_path} changed during execution")
            os.lseek(item.descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            observed = 0
            while chunk := os.read(item.descriptor, 8 * 1024 * 1024):
                digest.update(chunk)
                observed += len(chunk)
            if observed != item.identity.size or digest.hexdigest() != item.sha256:
                fail(f"transformer component {item.relative_path} changed during execution")
            item.metadata_identity.revalidate()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def document_bytes(value: Any) -> bytes:
    return BASE.document_bytes(value)


def lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def required_environment(environment: Mapping[str, str]) -> dict[str, str]:
    values = {name: environment.get(name, "").strip() for name in REQUIRED_ENVIRONMENT}
    missing = [name for name, value in values.items() if not value]
    if missing:
        fail(f"{missing[0]} is required")
    return values


def reviewed_authorization_sha256(authorization: Mapping[str, Any]) -> str:
    observed = authorization.get("source_document_sha256")
    if observed != AUTHORIZATION_SHA256:
        fail("authorization differs from reviewed evidence")
    return observed


def component_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    layer = next(item for item in manifest["fixture_layers"] if item["id"] == "token-refiner")
    identifiers = tuple(layer["required_component_indexes"])
    block = next(item for item in manifest["fixture_layers"] if item["id"] == "transformer-block")
    if identifiers != tuple(block["required_component_indexes"]) or len(identifiers) != 32:
        fail("transformer layer authority sets drifted")
    components = BASE.manifest_component_map(manifest)
    if BASE.component_authority_set_sha256(identifiers, components) != AUTHORITY_SET_SHA256:
        fail("transformer authority set differs from the reviewed boundary")
    return identifiers


def component_authorities(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    components = BASE.manifest_component_map(manifest)
    return [{"id": identifier, "sha256": components[identifier]["sha256"]} for identifier in component_ids(manifest)]


def adapter_implementation_sha256(role: str) -> str:
    producer_sha256 = BASE.sha256_file(pathlib.Path(__file__).resolve())
    if role == "oracle":
        return producer_sha256
    return sha256_bytes(
        BASE.canonical_json_bytes(
            {
                "schema_version": "mold.minimax-h3.transformer-mold-adapter.v1",
                "producer_sha256": producer_sha256,
                "rust_adapter_sha256": BASE.sha256_file(
                    REPO_ROOT / "crates" / "mold-inference" / "src" / "bin"
                    / "h3_transformer_capture.rs"
                ),
                "cargo_manifest_sha256": BASE.sha256_file(
                    REPO_ROOT / "crates" / "mold-inference" / "Cargo.toml"
                ),
            }
        )
    )


def authenticate_transformer_snapshot(
    model_root: pathlib.Path, manifest: Mapping[str, Any]
) -> AuthenticatedTransformerSnapshot:
    components = BASE.manifest_component_map(manifest)
    files: list[AuthenticatedTransformerFile] = []
    try:
        for identifier in component_ids(manifest):
            component = components[identifier]
            relative_path = component["relative_path"]
            relative = pathlib.PurePosixPath(relative_path)
            if (
                relative.is_absolute()
                or relative.parts[0] not in {"transformer", "transformer_ref"}
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                fail("transformer authority contains a non-canonical path")
            candidate = model_root.joinpath(*relative.parts)
            if candidate.is_symlink():
                fail(f"transformer component {identifier} may not be a symbolic link")
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                fail(f"transformer component {identifier} is unavailable")
            if not resolved.is_file() or not BASE.is_relative_to(resolved, model_root):
                fail(f"transformer component {identifier} escapes the model snapshot")
            metadata_lines, metadata_identity = QWEN.model_metadata_lines(
                model_root, relative_path
            )
            if not metadata_lines or metadata_lines[0] != MODEL_REVISION:
                fail(f"transformer component {identifier} revision drifted")
            before = QWEN.ExternalFileIdentity.capture(resolved)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(resolved, os.O_RDONLY | nofollow)
            try:
                opened = QWEN.ExternalFileIdentity.from_metadata(
                    resolved, os.fstat(descriptor)
                )
                if opened != before:
                    fail(f"transformer component {identifier} changed while opening")
                digest = hashlib.sha256()
                observed = 0
                while chunk := os.read(descriptor, 8 * 1024 * 1024):
                    digest.update(chunk)
                    observed += len(chunk)
                if observed != opened.size or digest.hexdigest() != component["sha256"]:
                    fail(f"transformer component {identifier} differs from its reviewed pin")
                files.append(
                    AuthenticatedTransformerFile(
                        relative_path, component["sha256"], opened,
                        metadata_identity, descriptor,
                    )
                )
                descriptor = -1
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        return AuthenticatedTransformerSnapshot(files)
    except BaseException:
        for item in files:
            os.close(item.descriptor)
        raise


def layer_contract(manifest: Mapping[str, Any], layer: str) -> Mapping[str, Any]:
    matches = [item for item in manifest["fixture_layers"] if item["id"] == layer]
    if len(matches) != 1 or tuple(matches[0]["required_measurements"]) != MEASUREMENTS[layer]:
        fail(f"{layer} manifest contract drifted")
    component_ids(manifest)
    return matches[0]


def build_case(task: str) -> dict[str, Any]:
    if task not in TASKS:
        fail("transformer task must be fl2va or ref2va")
    descriptor = {
        "case_id": f"transformer-{task}-deterministic-v1",
        "task": task,
        "token_shape": [1, 2, 5376],
        "packed_shape": [1, 6, 5376],
        "adaln_shape": [2, 2688],
        "adaln_boundary": "post-silu-bfloat16",
        "positions": [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]],
        "adaln_indices": [1, 0, 2, 3, 5, 3],
        "timestep_indices": [0, 0, 0, 1, 1, 1],
        "video_indices": [1, 3],
        "audio_indices": [2, 4],
        "patterns": {"token": [31, 0.75], "packed": [37, 1.0], "adaln": [29, 0.5]},
    }
    descriptor["input_sha256"] = sha256_bytes(INPUT_DOMAIN + BASE.canonical_json_bytes(descriptor))
    return descriptor


def tensor_payload(torch: Any, key: str, tensor: Any) -> TensorPayload:
    copied = tensor.detach().to("cpu").contiguous()
    if not bool(torch.isfinite(copied.float()).all()):
        fail(f"{key} contains non-finite values")
    if copied.dtype == torch.bfloat16:
        encoded = copied.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
        dtype = "bfloat16"
    elif copied.dtype == torch.float32:
        encoded = copied.numpy().astype("<f4", copy=False).tobytes()
        dtype = "float32"
    else:
        fail(f"{key} has unsupported dtype {copied.dtype}")
    return TensorPayload(key, tuple(int(value) for value in copied.shape), dtype, encoded)


def deterministic(torch: Any, count: int, modulus: int, offset: float, device: Any) -> Any:
    values = torch.arange(count, dtype=torch.int64, device="cpu").remainder(modulus).float().div(16).sub(offset)
    return values.to(device=device, dtype=torch.bfloat16)


def load_runtime(diffusers: pathlib.Path, transformers: pathlib.Path) -> SimpleNamespace:
    runtime = BASE.load_capture_runtime(diffusers, transformers)
    module = importlib.import_module("diffusers.models.transformers.transformer_minimax_h3")
    safe = importlib.import_module("safetensors")
    for value in (
        module.MiniMaxH3TokenRefinerBlock, module.MiniMaxH3TransformerBlock,
        module.MiniMaxH3RotaryPosEmbed, module.MiniMaxH3AdaLayerNormOut,
    ):
        BASE.verify_module_origin("Diffusers MiniMax H3 transformer", value, diffusers)
    runtime.transformer_module = module
    runtime.safetensors = safe
    return runtime


def selected_state(
    runtime: SimpleNamespace,
    snapshot: AuthenticatedTransformerSnapshot,
    component_dir: str,
    prefix: str,
) -> dict[str, Any]:
    index_relative = f"{component_dir}/diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(snapshot.retained_bytes(index_relative, 16 * 1024 * 1024))
    names = [name for name in index["weight_map"] if name.startswith(prefix)]
    state: dict[str, Any] = {}
    for shard in sorted({index["weight_map"][name] for name in names}):
        selected_names = [name for name in names if index["weight_map"][name] == shard]
        descriptor_path = snapshot.descriptor_path(f"{component_dir}/{shard}")
        with runtime.safetensors.safe_open(descriptor_path, framework="pt", device="cpu") as opened:
            for name in selected_names:
                state[name[len(prefix):]] = opened.get_tensor(name)
    return state


class RecordingProcessor:
    def __init__(self, torch: Any):
        self.torch = torch
        self.normalized_q = None
        self.normalized_k = None
        self.rotated_q = None
        self.rotated_k = None

    def __call__(self, attn: Any, hidden: Any, rotary_emb: Any = None, attention_mask: Any = None) -> Any:
        torch = self.torch
        q = attn.norm_q(attn.to_q(hidden).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(hidden).unflatten(-1, (attn.heads, -1)))
        v = attn.to_v(hidden).unflatten(-1, (attn.heads, -1))
        self.normalized_q, self.normalized_k = q, k
        if rotary_emb is not None:
            apply_rope = importlib.import_module(
                "diffusers.models.transformers.transformer_minimax_h3"
            )._apply_rotary_emb
            q, k = apply_rope(q, *rotary_emb), apply_rope(k, *rotary_emb)
        self.rotated_q, self.rotated_k = q, k
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).flatten(2, 3).type_as(q)
        return attn.to_out[0](output)


def capture_oracle(
    runtime: SimpleNamespace,
    snapshot: AuthenticatedTransformerSnapshot,
    task: str,
    device_label: str,
) -> tuple[TensorPayload, ...]:
    torch = runtime.torch
    device = BASE.configure_torch(runtime, device_label)
    module = runtime.transformer_module
    component_dir = "transformer" if task == "fl2va" else "transformer_ref"
    refiner = module.MiniMaxH3TokenRefinerBlock(5376, 56, 128, 14336, 1e-5, 1e-5).to(dtype=torch.bfloat16)
    block = module.MiniMaxH3TransformerBlock(5376, 56, 128, 14336, 2688, 1e-5, 1e-5).to(dtype=torch.bfloat16)
    norm_out = module.MiniMaxH3AdaLayerNormOut(5376, 2688, 1e-5).to(dtype=torch.bfloat16)
    video_head = torch.nn.Linear(5376, 96, bias=True)
    audio_head = torch.nn.Linear(5376, 32, bias=True)
    refiner.load_state_dict(selected_state(runtime, snapshot, component_dir, "token_refiner.refiner_blocks.0."), strict=True)
    block.load_state_dict(selected_state(runtime, snapshot, component_dir, "transformer_blocks.0."), strict=True)
    norm_out.load_state_dict(selected_state(runtime, snapshot, component_dir, "norm_out."), strict=True)
    video_head.load_state_dict(selected_state(runtime, snapshot, component_dir, "proj_out."), strict=True)
    audio_head.load_state_dict(selected_state(runtime, snapshot, component_dir, "audio_proj_out."), strict=True)
    refiner, block, norm_out, video_head, audio_head = [item.eval().to(device) for item in (refiner, block, norm_out, video_head, audio_head)]
    token = deterministic(torch, 2 * 5376, 31, 0.75, device).reshape(1, 2, 5376)
    hidden = deterministic(torch, 6 * 5376, 37, 1.0, device).reshape(1, 6, 5376)
    adaln = deterministic(torch, 2 * 2688, 29, 0.5, device).reshape(2, 2688)
    positions = torch.tensor(build_case(task)["positions"], dtype=torch.float32, device=device)
    adaln_indices = torch.tensor(build_case(task)["adaln_indices"], dtype=torch.long, device=device)
    timestep_indices = torch.tensor(build_case(task)["timestep_indices"], dtype=torch.long, device=device)
    video_indices = torch.tensor(build_case(task)["video_indices"], dtype=torch.long, device=device)
    audio_indices = torch.tensor(build_case(task)["audio_indices"], dtype=torch.long, device=device)
    refiner_processor, block_processor = RecordingProcessor(torch), RecordingProcessor(torch)
    refiner.attn.set_processor(refiner_processor)
    block.attn.set_processor(block_processor)
    rope = module.MiniMaxH3RotaryPosEmbed(16, 10000.0).to(device)
    with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        token_output = refiner(token)
        parameters = block.adaln_proj.linear(adaln).view(-1, 6 * 5376).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = parameters
        normalized = block.norm1(hidden) * (1 + scale_a.index_select(0, adaln_indices)) + shift_a.index_select(0, adaln_indices)
        update = block.attn(normalized, rope(positions))
        block_hidden = hidden + gate_a.index_select(0, adaln_indices) * update
        normalized = block.norm2(block_hidden) * (1 + scale_m.index_select(0, adaln_indices)) + shift_m.index_select(0, adaln_indices)
        block_output = block_hidden + gate_m.index_select(0, adaln_indices) * block.ff(normalized)
        shift, scale = norm_out.linear(adaln).chunk(2, dim=-1)
        final = norm_out.norm(block_output) * (1 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)
        final = final.float()
        video = video_head(final).index_select(1, video_indices)
        audio = audio_head(final).index_select(1, audio_indices)
    return tuple(tensor_payload(torch, key, tensor) for key, tensor in (
        ("token-refiner", token_output), ("normalized-q", block_processor.normalized_q),
        ("normalized-k", block_processor.normalized_k), ("rotated-q", block_processor.rotated_q),
        ("rotated-k", block_processor.rotated_k), ("adaln", torch.cat(parameters, dim=1)),
        ("block-output", block_output), ("video-head", video), ("audio-head", audio),
    ))


def decode_raw_tensor(item: Mapping[str, Any]) -> TensorPayload:
    if set(item) != {"key", "dtype", "shape", "little_endian_base64"} or item["key"] not in RAW_KEYS:
        fail("Mold raw transformer tensor schema drifted")
    dtype = item["dtype"]
    expected_shape, expected_dtype = RAW_CONTRACT[item["key"]]
    if dtype != expected_dtype:
        fail("Mold raw transformer tensor dtype drifted")
    shape = tuple(item["shape"])
    encoded = base64.b64decode(item["little_endian_base64"], validate=True)
    if shape != expected_shape or len(encoded) != math.prod(shape) * (2 if dtype == "bfloat16" else 4):
        fail("Mold raw transformer tensor shape or byte count drifted")
    payload = TensorPayload(item["key"], shape, dtype, encoded)
    if not all(math.isfinite(float(value)) for value in payload.values()):
        fail("Mold raw transformer tensor contains non-finite values")
    return payload


def read_raw_output(
    encoded: bytes, task: str, device: str, source_revision: str
) -> tuple[TensorPayload, ...]:
    try:
        raw = json.loads(encoded)
    except (UnicodeError, json.JSONDecodeError):
        fail("Mold raw transformer output is malformed")
    expected_keys = {
        "schema_version", "claim_marker", "task", "component", "device",
        "source_revision", "authorization_source_sha256", "input_sha256",
        "authority_set_sha256", "tensors",
    }
    if set(raw) != expected_keys or raw["schema_version"] != RAW_SCHEMA or raw["claim_marker"] != CAPTURE_MARKER or raw["task"] != task or raw["component"] != ("transformer" if task == "fl2va" else "transformer_ref") or raw["device"] != device or raw["source_revision"] != source_revision or raw["authorization_source_sha256"] != AUTHORIZATION_SHA256 or raw["input_sha256"] != build_case(task)["input_sha256"] or raw["authority_set_sha256"] != AUTHORITY_SET_SHA256:
        fail("Mold raw transformer output is not bound to the requested authority")
    payloads = tuple(decode_raw_tensor(item) for item in raw["tensors"])
    if tuple(item.key for item in payloads) != RAW_KEYS:
        fail("Mold raw transformer tensor order drifted")
    return payloads


def capture_directory(fixture_root: pathlib.Path, task: str) -> pathlib.Path:
    directory = fixture_root / "captures" / "transformer" / build_case(task)["input_sha256"][:24]
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = directory.resolve(strict=True)
    metadata = directory.stat()
    if not BASE.is_relative_to(directory, fixture_root) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        fail("transformer capture directory must be external and owner-only")
    return directory


def capture_mold(fixture_root: pathlib.Path, authorization: pathlib.Path, model_root: pathlib.Path, manifest_path: pathlib.Path, task: str, device: str, source_revision: str, staged_mold: pathlib.Path) -> tuple[tuple[TensorPayload, ...], pathlib.Path]:
    directory = capture_directory(fixture_root, task)
    raw_path = directory / f"mold-{task}-raw.json"
    if raw_path.exists():
        fail("Mold raw transformer evidence already exists")
    target = fixture_root / "build" / "transformer" / source_revision
    QWEN.private_directory(target)
    command = [
        "cargo", "run", "--quiet", "--locked", "--release", "--target-dir", str(target),
        "-p", "mold-ai-inference", "--features", MOLD_FEATURES, "--bin", MOLD_BINARY, "--",
        "--model-root", str(model_root), "--manifest", str(manifest_path),
        "--authorization-record", str(authorization), "--task", task, "--device", device,
        "--source-revision", source_revision,
        "--raw-output", str(raw_path),
    ]
    result = subprocess.run(command, cwd=staged_mold, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raw_path.unlink(missing_ok=True)
        fail("Mold transformer capture adapter failed; inspect the protected runner log")
    try:
        snapshot, encoded = QWEN.SealedFileSnapshot.seal_and_read(
            raw_path, 64 * 1024 * 1024
        )
        payloads = read_raw_output(encoded, task, device, source_revision)
        snapshot.revalidate()
        return payloads, raw_path
    except BaseException:
        raw_path.unlink(missing_ok=True)
        raise


def metric_payload(key: str, values: Sequence[float], dtype: str = "float64") -> TensorPayload:
    encoding = "<d" if dtype == "float64" else "<q"
    return TensorPayload(key, (len(values),), dtype, b"".join(struct.pack(encoding, value) for value in values))


def complete_bf16(payload: TensorPayload, key: str) -> TensorPayload:
    if payload.dtype != "bfloat16":
        fail(f"{key} complete evidence source must be bfloat16")
    return TensorPayload(key, (math.prod(payload.shape),), "bfloat16", payload.encoded)


def leading_bf16(payload: TensorPayload, width: int) -> bytes:
    if payload.dtype != "bfloat16" or not payload.shape or width > payload.shape[-1]:
        fail("partial RoPE source tensor contract drifted")
    row_bytes = payload.shape[-1] * 2
    selected_bytes = width * 2
    rows = math.prod(payload.shape[:-1])
    return b"".join(
        payload.encoded[offset * row_bytes:offset * row_bytes + selected_bytes]
        for offset in range(rows)
    )


def measurement_payloads(raw: Sequence[TensorPayload], layer: str) -> tuple[TensorPayload, ...]:
    values = {item.key: item for item in raw}
    tolerance = metric_payload(
        "tolerance", (max(ABSOLUTE_TOLERANCE[layer].values()), RELATIVE_TOLERANCE)
    )
    if layer == "token-refiner":
        token = values["token-refiner"]
        numbers = [float(value) for value in token.values()]
        mean = sum(numbers) / len(numbers)
        std = math.sqrt(sum((value - mean) ** 2 for value in numbers) / len(numbers))
        return (
            metric_payload("shape", token.shape, "int64"),
            metric_payload("statistics", (min(numbers), max(numbers), mean, std)),
            complete_bf16(token, "sampled-values"), tolerance,
        )
    normalized = [values["normalized-q"], values["normalized-k"]]
    rms = [math.sqrt(sum(float(value) ** 2 for value in item.values()) / math.prod(item.shape)) for item in normalized]
    rotated_q, rotated_k = values["rotated-q"], values["rotated-k"]
    if rotated_q.shape != rotated_k.shape:
        fail("rotated Q/K tensor shapes drifted")
    rotary_width = 96
    partial_shape = (2, *rotated_q.shape[:-1], rotary_width)
    partial_rope = TensorPayload(
        "partial-mm-rope",
        partial_shape,
        "bfloat16",
        leading_bf16(rotated_q, rotary_width) + leading_bf16(rotated_k, rotary_width),
    )
    return (
        metric_payload("qk-rms", rms), partial_rope,
        TensorPayload("adaln", values["adaln"].shape, "bfloat16", values["adaln"].encoded),
        TensorPayload("video-head", values["video-head"].shape, "float32", values["video-head"].encoded),
        TensorPayload("audio-head", values["audio-head"].shape, "float32", values["audio-head"].encoded),
        tolerance,
    )


def sample_indexes(payload: TensorPayload) -> list[int]:
    return list(range(math.prod(payload.shape)))


def unravel(index: int, shape: Sequence[int]) -> list[int]:
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis], index = index % shape[axis], index // shape[axis]
    return result


def tensor_record(payload: TensorPayload) -> dict[str, Any]:
    values = payload.values()
    numeric = [float(value) for value in values]
    return {
        "key": payload.key, "shape": list(payload.shape), "dtype": payload.dtype,
        "content_sha256": sha256_bytes(payload.encoded),
        "statistics": BASE.finite_statistics(values),
        "samples": [{"index": unravel(index, payload.shape), "value": values[index]} for index in sample_indexes(payload)],
    }


def capture_command(role: str, device: str) -> str:
    return f"python3 scripts/capture-minimax-h3-transformer.py capture --role {role} --device {device}"


def build_layer_document(manifest: Mapping[str, Any], role: str, task: str, layer: str, source_revision: str, authorization_sha256: str, device: str, raw: Sequence[TensorPayload]) -> dict[str, Any]:
    contract = layer_contract(manifest, layer)
    case = build_case(task)
    authorities = component_authorities(manifest)
    outputs = [tensor_record(item) for item in measurement_payloads(raw, layer)]
    dtype = "bfloat16"
    adapter_sha256 = adapter_implementation_sha256(role)
    runtime_sha256 = sha256_bytes(
        BASE.canonical_json_bytes(manifest["numerical_authority"]["oracle_runtime"])
    )
    document: dict[str, Any] = {
        "schema_version": "mold.minimax-h3.layer-output.v1", "family": "minimax-h3",
        "case_id": f"{case['case_id']}-{layer}", "layer": layer, "authority_tier": "exact-full-bf16",
        "authorization_document_sha256": authorization_sha256,
        "input": {"id": case["case_id"], "sha256": case["input_sha256"], "component_index_sha256": AUTHORITY_SET_SHA256, "component_indexes": authorities},
        "producer": {"role": role, "implementation": f"{'diffusers' if role == 'oracle' else 'mold-candle'}-minimax-h3-{layer}-{task}-v1", "source_id": "diffusers" if role == "oracle" else "mold", "revision": source_revision},
        "adapter": {"schema_version": "mold.minimax-h3.layer-adapter.v1", "id": f"minimax-h3-{layer}-{role}-v1", "command": capture_command(role, device), "tensor_hash_encoding": "canonical-typed-le-v1", "implementation_sha256": adapter_sha256},
        "environment": {"device": device, "dtype": dtype, "attention_backend": "math", "acceleration_policy": {"enabled": ["cuda", "deterministic-cuda", "math-attention"], "disabled": list(manifest["numerical_authority"]["excluded_accelerations"])}, "forbidden_accelerations_disabled": True},
        "provenance": [], "outputs": outputs,
    }
    if role == "oracle":
        document["environment"]["oracle_runtime"] = dict(
            manifest["numerical_authority"]["oracle_runtime"]
        )
    provenance = {
        "source-revision": source_revision,
        "diffusers-revision": DIFFUSERS_REVISION,
        "adapter-implementation-sha256": adapter_sha256,
        "oracle-runtime-sha256": runtime_sha256,
        "component-index-hashes": ",".join(f"{item['id']}={item['sha256']}" for item in sorted(authorities, key=lambda item: item["id"])),
        "device": device, "dtype": dtype, "attention-backend": "math",
        "capture-command": capture_command(role, device),
    }
    document["provenance"] = [{"key": key, "value": provenance[key]} for key in contract["required_provenance"]]
    if role == "oracle":
        document["comparison"] = [
            {
                "key": item.key,
                "absolute": ABSOLUTE_TOLERANCE[layer].get(item.key, 0.0),
                "relative": (
                    RELATIVE_TOLERANCE
                    if item.key in RECORD_ONLY_MEASUREMENTS[layer]
                    else 0.0
                ),
                "metric": "elementwise-atol-plus-rtol",
                "hash_policy": (
                    "record-only"
                    if item.key in RECORD_ONLY_MEASUREMENTS[layer]
                    else "exact"
                ),
            }
            for item in measurement_payloads(raw, layer)
        ]
    return document


def build_bundle(manifest: Mapping[str, Any], role: str, source_revision: str, authorization_sha256: str, device: str, fixture_root: pathlib.Path, documents: Sequence[tuple[pathlib.Path, Mapping[str, Any], bytes]]) -> dict[str, Any]:
    fixtures = []
    for path, document, encoded in documents:
        output = document["outputs"][0]
        tolerance = 0 if output["dtype"] == "int64" else 0.015625
        fixtures.append({
            "id": f"{document['case_id']}-{role}", "layer": document["layer"], "authority_tier": document["authority_tier"],
            "relative_path": path.relative_to(fixture_root).as_posix(), "sha256": sha256_bytes(encoded),
            "component_index_sha256": document["input"]["component_index_sha256"], "component_indexes": document["input"]["component_indexes"],
            "tensor": {"shape": output["shape"], "dtype": output["dtype"], "min": output["statistics"]["min"], "max": output["statistics"]["max"], "mean": output["statistics"]["mean"], "std": output["statistics"]["std"], "sampled_values": [sample["value"] for sample in output["samples"]]},
            "tolerance": {"absolute": tolerance, "relative": tolerance, "metric": "elementwise-atol-plus-rtol"},
        })
    bundle = {"schema_version": "mold.minimax-h3.fixture-bundle.v1", "manifest_sha256": BASE.sha256_file(MANIFEST_PATH), "authorization_document_sha256": authorization_sha256, "capture_environment": {"framework": "diffusers" if role == "oracle" else "mold", "framework_revision": source_revision, "device": device, "dtype": "bfloat16", "attention_backend": "math", "acceleration_policy": {"enabled": ["cuda", "deterministic-cuda", "math-attention"], "disabled": list(manifest["numerical_authority"]["excluded_accelerations"])}, "command": f"transformer paired {role}", "forbidden_accelerations_disabled": True}, "fixtures": fixtures}
    if role == "oracle":
        bundle["capture_environment"]["oracle_runtime"] = dict(
            manifest["numerical_authority"]["oracle_runtime"]
        )
        bundle["capture_environment"]["oracle_adapters"] = [
            {
                "layer": document["layer"],
                "id": document["adapter"]["id"],
                "command": document["adapter"]["command"],
                "implementation_sha256": document["adapter"]["implementation_sha256"],
            }
            for _, document, _ in documents
        ]
    return bundle


def write_capture(fixture_root: pathlib.Path, authorization: pathlib.Path, manifest: Mapping[str, Any], role: str, source_revision: str, authorization_sha256: str, device: str, task: str, raw: Sequence[TensorPayload], conformance: Any, gpu: Any) -> tuple[list[pathlib.Path], pathlib.Path]:
    directory = capture_directory(fixture_root, task)
    documents = []
    for layer in LAYERS:
        document = build_layer_document(manifest, role, task, layer, source_revision, authorization_sha256, device, raw)
        conformance.validate_schema(document, conformance.LAYER_OUTPUT_SCHEMA_PATH)
        gpu.validate_manifest_layer_evidence(document, gpu.exact_layer_contracts(manifest)[layer], role)
        path = directory / f"{role}-{task}-{layer}.json"
        documents.append((path, document, document_bytes(document)))
    bundle_path = directory / f"{role}-{task}-bundle.json"
    bundle = build_bundle(manifest, role, source_revision, authorization_sha256, device, fixture_root, documents)
    conformance.validate_schema(bundle, conformance.BUNDLE_SCHEMA_PATH)
    created = []
    try:
        for path, _, encoded in documents:
            BASE.exclusive_write(path, encoded); created.append(path)
        BASE.exclusive_write(bundle_path, document_bytes(bundle)); created.append(bundle_path)
        conformance.validate_bundle(manifest, str(fixture_root), str(bundle_path), str(authorization))
    except BaseException:
        for path in created: path.unlink(missing_ok=True)
        raise
    return [item[0] for item in documents], bundle_path


def run_capture(
    environment: Mapping[str, str], role: str, device: str
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    BASE.reject_excluded_acceleration_environment(environment)
    values = required_environment(environment)
    conformance = load_module("minimax_h3_transformer_conformance", CONFORMANCE_PATH)
    gpu = load_module("minimax_h3_transformer_gpu", GPU_PATH)
    manifest = conformance.validate_manifest(); component_ids(manifest)
    fixture_root = BASE.external_directory("fixture root", values["MOLD_H3_FIXTURE_ROOT"])
    authorization = BASE.external_file("authorization record", values["MOLD_H3_AUTHORIZATION_RECORD"])
    auth = BASE.validate_reviewed_authorization(conformance, authorization)
    authorization_sha256 = reviewed_authorization_sha256(auth)
    model_root = BASE.external_directory("official model", values["MOLD_H3_OFFICIAL_MODEL"])
    diffusers = BASE.external_directory("Diffusers checkout", values["MOLD_H3_DIFFUSERS_CHECKOUT"])
    transformers = BASE.external_directory("Transformers checkout", values["MOLD_H3_TRANSFORMERS_CHECKOUT"])
    BASE.verify_git_checkout("Diffusers", diffusers, DIFFUSERS_REVISION, "https://github.com/huggingface/diffusers")
    BASE.verify_git_checkout("Transformers", transformers, TRANSFORMERS_REVISION, "https://github.com/huggingface/transformers")
    runtime = load_runtime(diffusers, transformers)
    runtime_identity = BASE.observed_oracle_runtime(runtime)
    if runtime_identity != manifest["numerical_authority"]["oracle_runtime"]: fail("oracle runtime differs from reviewed authority")
    paths: list[pathlib.Path] = []
    bundles: list[pathlib.Path] = []
    revision = (
        DIFFUSERS_REVISION
        if role == "oracle"
        else BASE.command_output(["git", "rev-parse", "HEAD"], REPO_ROOT)
    )
    created_artifacts: list[pathlib.Path] = []
    try:
        with (
            authenticate_transformer_snapshot(model_root, manifest) as transformer_snapshot,
            contextlib.ExitStack() as resources,
        ):
            if role == "oracle":
                staged_mold = None
            elif role == "mold":
                stage = resources.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix=".h3-transformer-source-", dir=fixture_root
                    )
                )
                staged_mold = pathlib.Path(stage) / "mold"
                QWEN.extract_mold_snapshot(REPO_ROOT, revision, staged_mold)
            else:
                fail("capture role must be oracle or mold")
            task_captures: list[tuple[str, tuple[TensorPayload, ...]]] = []
            for task in TASKS:
                if role == "oracle":
                    raw = capture_oracle(runtime, transformer_snapshot, task, device)
                else:
                    if staged_mold is None:
                        fail("Mold source snapshot is unavailable")
                    staged_manifest = (
                        staged_mold / "tests" / "fixtures" / "minimax_h3"
                        / "conformance-manifest.json"
                    )
                    raw, raw_path = capture_mold(
                        fixture_root, authorization, model_root, staged_manifest,
                        task, device, revision, staged_mold,
                    )
                    created_artifacts.append(raw_path)
                transformer_snapshot.revalidate()
                task_captures.append((task, raw))
                if role == "oracle":
                    runtime.torch.cuda.empty_cache()
            for task, raw in task_captures:
                task_paths, bundle = write_capture(
                    fixture_root, authorization, manifest, role, revision,
                    authorization_sha256, device, task, raw, conformance, gpu,
                )
                created_artifacts.extend((*task_paths, bundle))
                paths.extend(task_paths)
                bundles.append(bundle)
    except BaseException:
        for path in reversed(created_artifacts):
            path.unlink(missing_ok=True)
        raise
    return paths, bundles


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--role", choices=("oracle", "mold"), required=True)
    capture.add_argument("--device", default=os.environ.get("MOLD_H3_CAPTURE_DEVICE", "cuda:0"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        paths, bundles = run_capture(os.environ, args.role, args.device)
        print(
            f"MiniMax H3 transformer capture complete: role={args.role} "
            f"tasks={len(TASKS)} layers={len(paths)} bundles={len(bundles)}"
        )
        return 0
    except (CaptureFailure, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"MiniMax H3 transformer capture rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
