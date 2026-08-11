#!/usr/bin/env python3
"""Capture protected MiniMax H3 visual-VAE oracle and Mold evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pathlib
import platform
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
BASE_PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-conditioner.py"
QWEN_PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-qwen-layer50.py"
CONFORMANCE_TOOL_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
GPU_CONFORMANCE_TOOL_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"
MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "minimax_h3" / "conformance-manifest.json"
)
BASE_PRODUCER_SHA256 = (
    "006c761d6217430c78dcfc0b6c3a6a7d714924584a8fa79ea25407f82a876eda"
)
QWEN_PRODUCER_SHA256 = (
    "bc85f8497219e4d7159ccf4e9e478cab28ba02b66db245afc78e603149440410"
)

DIFFUSERS_REVISION = "9c6a68c32b3b2a64db91800b624d33cec6e25ab8"
TRANSFORMERS_REVISION = "42f189ded85d18d00b51161d694cafd325e32b91"
MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
VISUAL_INDEX_SHA256 = "15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca"
VISUAL_CONFIG_SHA256 = (
    "78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745"
)
VISUAL_PAYLOAD_BYTES = 10_415_475_936
VISUAL_SHARD_SIZES = {
    "diffusion_pytorch_model-00001-of-00003.safetensors": 5_061_033_024,
    "diffusion_pytorch_model-00002-of-00003.safetensors": 4_955_986_528,
    "diffusion_pytorch_model-00003-of-00003.safetensors": 398_539_336,
}
VISUAL_SHARD_AUTHORITIES = (
    (
        "official-video-vae-shard-00001-of-00003",
        "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
        "72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374",
    ),
    (
        "official-video-vae-shard-00002-of-00003",
        "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
        "2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198",
    ),
    (
        "official-video-vae-shard-00003-of-00003",
        "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
        "c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39",
    ),
)
REVIEWED_AUTHORIZATION_EVIDENCE_SHA256 = (
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d"
)
REQUEST_SCHEMA = "mold.minimax-h3.visual-vae-capture-request.v1"
RAW_OUTPUT_SCHEMA = "mold.minimax-h3.visual-vae-raw-output.v1"
CAPTURE_MARKER = "mold.minimax-h3.private-uat-visual-vae-f32-fp16-capture.v1"
INPUT_IDENTITY_DOMAIN = b"mold.minimax-h3.visual-vae-input.v1\0"
LAYER_ID = "visual-vae"
CASE_ID = "visual-vae-320x320x22-seed42-v1"
PATTERN_ID = "rgb-affine-uint8-v1"
CASE_FRAMES = 22
CASE_HEIGHT = 320
CASE_WIDTH = 320
LATENT_FRAMES = 7
LATENT_HEIGHT = 20
LATENT_WIDTH = 20
TILE_SIZE = 256
TILE_OVERLAP = 64
SEAMS = (64, 256)
SEAM_FRAMES = (0, 10, 21)
COMPONENT_IDS = (
    "official-video-vae-config",
    "official-video-vae-index",
    *(identifier for identifier, _, _ in VISUAL_SHARD_AUTHORITIES),
)
MEASUREMENTS = (
    "pixel-normalization",
    "posterior-moments",
    "posterior-seed-42",
    "posterior-sample",
    "posterior-fp16-roundtrip",
    "latent-normalization",
    "decoded-frames",
    "tile-seams",
)
MEASUREMENT_DTYPES = {
    "pixel-normalization": "float32",
    "posterior-moments": "float32",
    "posterior-seed-42": "float32",
    "posterior-sample": "float32",
    "posterior-fp16-roundtrip": "float16",
    "latent-normalization": "float32",
    "decoded-frames": "float32",
    "tile-seams": "float32",
}
TENSOR_HASH_ENCODING = "canonical-typed-le-v1"
MOLD_BINARY = "h3_visual_vae_capture"
MOLD_FEATURES = "dev-bins,h3-private-uat,cuda"
MAXIMUM_RAW_OUTPUT_BYTES = 128 * 1024 * 1024
MAXIMUM_REQUEST_BYTES = 64 * 1024
REQUIRED_ENVIRONMENT = (
    "MOLD_H3_FIXTURE_ROOT",
    "MOLD_H3_AUTHORIZATION_RECORD",
    "MOLD_H3_OFFICIAL_MODEL",
    "MOLD_H3_DIFFUSERS_CHECKOUT",
    "MOLD_H3_TRANSFORMERS_CHECKOUT",
)


class CaptureFailure(Exception):
    """A fail-closed protected visual-VAE capture violation."""


def fail(message: str) -> None:
    raise CaptureFailure(message)


def load_module(name: str, path: pathlib.Path) -> Any:
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


def load_bound_module(name: str, path: pathlib.Path, expected_sha256: str) -> Any:
    try:
        observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        fail(f"cannot read repository module {name}: {error}")
    if observed_sha256 != expected_sha256:
        fail(f"repository module {name} differs from the reviewed capture dependency")
    return load_module(name, path)


BASE = load_bound_module(
    "minimax_h3_visual_capture_base", BASE_PRODUCER_PATH, BASE_PRODUCER_SHA256
)
QWEN = load_bound_module(
    "minimax_h3_visual_capture_staging", QWEN_PRODUCER_PATH, QWEN_PRODUCER_SHA256
)


@dataclass(frozen=True)
class TensorPayload:
    key: str
    shape: tuple[int, ...]
    dtype: str
    encoded: bytes

    def values(self) -> list[float]:
        encoding = "<f" if self.dtype == "float32" else "<e"
        return [float(item[0]) for item in struct.iter_unpack(encoding, self.encoded)]


def canonical_json_bytes(value: Any) -> bytes:
    return BASE.canonical_json_bytes(value)


def document_bytes(value: Any) -> bytes:
    return BASE.document_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def required_environment(environment: Mapping[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in REQUIRED_ENVIRONMENT:
        value = environment.get(name, "").strip()
        if not value:
            fail(f"{name} is required")
        values[name] = value
    return values


def lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def layer_contract(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [layer for layer in manifest["fixture_layers"] if layer["id"] == LAYER_ID]
    if len(matches) != 1:
        fail("checked manifest does not contain one visual-VAE contract")
    layer = matches[0]
    if tuple(layer["required_component_indexes"]) != COMPONENT_IDS:
        fail("visual-VAE component authority drifted")
    if tuple(layer["required_measurements"]) != MEASUREMENTS:
        fail("visual-VAE measurement contract drifted")
    required = tuple(layer["required_provenance"])
    if required not in {
        (
            "source-revision",
            "component-index-hashes",
            "artifact-set-sha256",
            "device",
            "dtype",
            "capture-command",
        ),
        (
            "source-revision",
            "diffusers-revision",
            "adapter-implementation-sha256",
            "oracle-runtime-sha256",
            "component-index-hashes",
            "artifact-set-sha256",
            "device",
            "dtype",
            "capture-command",
        ),
    }:
        fail("visual-VAE provenance contract drifted")
    return layer


def adapter_implementation_sha256(role: str) -> str:
    script_sha = BASE.sha256_file(pathlib.Path(__file__).resolve())
    dependency_sha = BASE.sha256_file(QWEN_PRODUCER_PATH)
    if role == "oracle":
        return script_sha
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "mold.minimax-h3.visual-vae-mold-adapter.v1",
                "producer_sha256": script_sha,
                "staging_support_sha256": dependency_sha,
                "rust_adapter_sha256": BASE.sha256_file(
                    REPO_ROOT
                    / "crates"
                    / "mold-inference"
                    / "src"
                    / "bin"
                    / "h3_visual_vae_capture.rs"
                ),
                "candle_visual_condition_sha256": BASE.sha256_file(
                    REPO_ROOT
                    / "crates"
                    / "mold-candle"
                    / "src"
                    / "minimax_h3"
                    / "visual_condition.rs"
                ),
                "candle_visual_vae_sha256": BASE.sha256_file(
                    REPO_ROOT
                    / "crates"
                    / "mold-candle"
                    / "src"
                    / "minimax_h3"
                    / "visual_vae.rs"
                ),
                "cargo_manifest_sha256": BASE.sha256_file(
                    REPO_ROOT / "crates" / "mold-inference" / "Cargo.toml"
                ),
            }
        )
    )


def put_u64(destination: bytearray, value: int) -> None:
    if value < 0 or value > 0xFFFF_FFFF_FFFF_FFFF:
        fail("visual capture input length exceeds uint64")
    destination.extend(struct.pack("<Q", value))


def put_string(destination: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    put_u64(destination, len(encoded))
    destination.extend(encoded)


def build_case() -> dict[str, Any]:
    case: dict[str, Any] = {
        "case_id": CASE_ID,
        "pattern": PATTERN_ID,
        "frames": CASE_FRAMES,
        "height": CASE_HEIGHT,
        "width": CASE_WIDTH,
    }
    identity = bytearray(INPUT_IDENTITY_DOMAIN)
    put_string(identity, case["case_id"])
    put_string(identity, case["pattern"])
    for value in (
        case["frames"],
        case["height"],
        case["width"],
        TILE_SIZE,
        TILE_OVERLAP,
        *SEAMS,
        *SEAM_FRAMES,
    ):
        put_u64(identity, int(value))
    case["input_sha256"] = sha256_bytes(bytes(identity))
    return case


def metadata_lines(
    model_root: pathlib.Path, relative_path: str
) -> tuple[list[str], QWEN.ExternalFileIdentity]:
    return QWEN.model_metadata_lines(model_root, relative_path)


def verify_visual_snapshot(
    model_root: pathlib.Path,
) -> tuple[str, tuple[QWEN.ExternalFileIdentity, ...]]:
    reviewed = {
        "vae/config.json": VISUAL_CONFIG_SHA256,
        "vae/diffusion_pytorch_model.safetensors.index.json": VISUAL_INDEX_SHA256,
    }
    identities: list[QWEN.ExternalFileIdentity] = []
    component_hashes: dict[str, str] = {}
    for relative, expected in reviewed.items():
        path = (model_root / relative).resolve(strict=True)
        if not BASE.is_relative_to(path, model_root) or path.is_symlink():
            fail("official visual-VAE metadata escapes its model snapshot")
        before = QWEN.ExternalFileIdentity.capture(path)
        digest = BASE.sha256_file(path)
        after = QWEN.ExternalFileIdentity.capture(path)
        if before != after or digest != expected:
            fail("official visual-VAE config or index differs from its authority")
        lines, metadata_identity = metadata_lines(model_root, relative)
        if not lines or lines[0] != MODEL_REVISION:
            fail("official visual-VAE metadata revision drifted")
        identities.extend((after, metadata_identity))
        component_hashes[relative] = digest

    index_path = model_root / "vae/diffusion_pytorch_model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
        total_size = int(index["metadata"]["total_size"])
    except (
        OSError,
        UnicodeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        fail("official visual-VAE index is malformed")
    if set(shard_names) != set(VISUAL_SHARD_SIZES) or total_size != VISUAL_PAYLOAD_BYTES:
        fail("official visual-VAE shard geometry drifted")
    reviewed_shards = {
        pathlib.Path(relative).name: (relative, sha256)
        for _, relative, sha256 in VISUAL_SHARD_AUTHORITIES
    }
    if set(shard_names) != set(reviewed_shards):
        fail("official visual-VAE index differs from the reviewed shard set")
    for name in shard_names:
        if pathlib.Path(name).name != name or not name.endswith(".safetensors"):
            fail("official visual-VAE index contains a non-canonical shard path")
        relative, expected_sha256 = reviewed_shards[name]
        path = (model_root / relative).resolve(strict=True)
        if not BASE.is_relative_to(path, model_root) or path.is_symlink():
            fail("official visual-VAE shard escapes its model snapshot")
        before = QWEN.ExternalFileIdentity.capture(path)
        if before.size != VISUAL_SHARD_SIZES[name]:
            fail("official visual-VAE shard size differs from its reviewed artifact")
        lines, metadata_identity = metadata_lines(model_root, relative)
        if (
            len(lines) < 2
            or lines[0] != MODEL_REVISION
            or lines[1] != expected_sha256
        ):
            fail("official visual-VAE shard metadata differs from the reviewed pin")
        digest = BASE.sha256_file(path)
        after = QWEN.ExternalFileIdentity.capture(path)
        if before != after or digest != expected_sha256:
            fail("official visual-VAE shard differs from the reviewed pin")
        component_hashes[relative] = digest
        identities.extend((after, metadata_identity))
    artifact_set = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "mold.minimax-h3.visual-vae-artifact-set.v1",
                "model_revision": MODEL_REVISION,
                "files": component_hashes,
            }
        )
    )
    return artifact_set, tuple(identities)


def load_runtime(
    diffusers_checkout: pathlib.Path, transformers_checkout: pathlib.Path
) -> SimpleNamespace:
    runtime = BASE.load_capture_runtime(diffusers_checkout, transformers_checkout)
    try:
        autoencoder_module = importlib.import_module(
            "diffusers.models.autoencoders.autoencoder_kl_minimax_h3"
        )
    except (ImportError, RuntimeError) as error:
        fail(f"reviewed visual-VAE runtime cannot be imported: {error}")
    BASE.verify_module_origin(
        "Diffusers MiniMax H3 visual VAE",
        autoencoder_module.AutoencoderKLMiniMaxH3,
        diffusers_checkout,
    )
    runtime.vae_class = autoencoder_module.AutoencoderKLMiniMaxH3
    runtime.diffusers_checkout = diffusers_checkout
    return runtime


def observed_runtime(runtime: SimpleNamespace) -> dict[str, str]:
    if hasattr(BASE, "observed_oracle_runtime"):
        return BASE.observed_oracle_runtime(runtime)
    cuda = getattr(getattr(runtime.torch, "version", None), "cuda", None)
    values = {
        "python": platform.python_version(),
        "torch": str(getattr(runtime.torch, "__version__", "")),
        "numpy": str(getattr(runtime.numpy, "__version__", "")),
        "cuda": "" if cuda is None else str(cuda),
        "transformers_revision": TRANSFORMERS_REVISION,
    }
    if any(not value for value in values.values()):
        fail("visual-VAE capture ambient runtime identity is incomplete")
    return values


def rgb_pattern(numpy: Any) -> Any:
    return BASE.rgb_pattern(numpy, CASE_FRAMES, CASE_HEIGHT, CASE_WIDTH)


def tensor_payload(torch: Any, key: str, tensor: Any, dtype: str) -> TensorPayload:
    expected = torch.float32 if dtype == "float32" else torch.float16
    if tensor.dtype != expected:
        fail(f"{key} has {tensor.dtype} instead of {dtype}")
    copied = tensor.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(copied).all()):
        fail(f"{key} contains non-finite values")
    numpy_dtype = "<f4" if dtype == "float32" else "<f2"
    array = copied.numpy().astype(numpy_dtype, copy=False)
    return TensorPayload(
        key=key,
        shape=tuple(int(value) for value in copied.shape),
        dtype=dtype,
        encoded=array.tobytes(order="C"),
    )


def seam_probe(torch: Any, decoded: Any) -> Any:
    if tuple(decoded.shape) != (1, 3, CASE_FRAMES, CASE_HEIGHT, CASE_WIDTH):
        fail("decoded visual evidence has the wrong seam canvas")
    probes = (0, 63, 64, 255, 256, 319)
    values = []
    for frame in SEAM_FRAMES:
        for channel in range(3):
            for seam in SEAMS:
                for probe in probes:
                    values.extend(
                        (
                            decoded[0, channel, frame, seam - 1, probe],
                            decoded[0, channel, frame, seam, probe],
                            decoded[0, channel, frame, probe, seam - 1],
                            decoded[0, channel, frame, probe, seam],
                        )
                    )
    return torch.stack(values).to(device="cpu", dtype=torch.float32)


def reviewed_payload_shapes() -> dict[str, tuple[int, ...]]:
    latent = (1, 24, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    return {
        "pixel-normalization": (1, 3, CASE_FRAMES, CASE_HEIGHT, CASE_WIDTH),
        "posterior-moments": (1, 48, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
        "posterior-seed-42": latent,
        "posterior-sample": latent,
        "posterior-fp16-roundtrip": latent,
        "latent-normalization": latent,
        "decoded-frames": (1, 3, CASE_FRAMES, CASE_HEIGHT, CASE_WIDTH),
        "tile-seams": (len(SEAM_FRAMES) * 3 * len(SEAMS) * 6 * 4,),
    }


def validate_payload_shapes(payloads: Sequence[TensorPayload]) -> None:
    expected = reviewed_payload_shapes()
    if tuple(payload.key for payload in payloads) != MEASUREMENTS:
        fail("visual capture tensor order drifted")
    for payload in payloads:
        if payload.shape != expected[payload.key]:
            fail(f"{payload.key} shape differs from the reviewed visual case")


def capture_oracle(
    runtime: SimpleNamespace, model_root: pathlib.Path, device_label: str
) -> tuple[TensorPayload, ...]:
    torch = runtime.torch
    device = BASE.configure_torch(runtime, device_label)
    try:
        vae = runtime.vae_class.from_pretrained(
            model_root / "vae",
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        vae.eval().to(device)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        fail(f"pinned visual VAE could not be loaded: {error}")
    BASE.verify_module_origin(
        "loaded Diffusers MiniMax H3 visual VAE",
        vae.__class__,
        runtime.diffusers_checkout,
    )
    if any(parameter.dtype != torch.float32 for parameter in vae.parameters()):
        fail("Diffusers visual oracle downcast the official F32 checkpoint")
    if not vae.use_tiling or (
        vae.tile_sample_min_height,
        vae.tile_sample_min_width,
        vae.tile_sample_min_overlap_height,
        vae.tile_sample_min_overlap_width,
    ) != (TILE_SIZE, TILE_SIZE, TILE_OVERLAP, TILE_OVERLAP):
        fail("Diffusers visual oracle tiling policy drifted")

    pattern = rgb_pattern(runtime.numpy)
    pixels_u8 = torch.from_numpy(pattern).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1, 1)
    normalized_pixels = (pixels_u8.to(torch.float32).div(255.0) - mean) / std
    with torch.no_grad():
        posterior = vae.encode(normalized_pixels, return_dict=False)[0]
        moments = posterior.parameters
        generator = torch.Generator().manual_seed(42)
        noise = torch.randn(
            posterior.mean.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        sample = posterior.mean + posterior.std * noise.to(device)
        official_sample = posterior.sample(generator=torch.Generator().manual_seed(42))
        if not torch.equal(sample, official_sample):
            fail("captured seed-42 posterior differs from the Diffusers sampler")
        rounded_f16 = sample.to(torch.float16).cpu()
        roundtrip_f32 = rounded_f16.float()
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        normalized_latents = (roundtrip_f32 - latents_mean) / latents_std
        decoded_latents = normalized_latents.to(device) * latents_std.to(
            device
        ) + latents_mean.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            decoded = vae.decode(decoded_latents, return_dict=False)[0]
        decoded = (decoded.float() * std + mean).clamp(0, 1)
        seams = seam_probe(torch, decoded)
    payloads = (
        tensor_payload(torch, "pixel-normalization", normalized_pixels, "float32"),
        tensor_payload(torch, "posterior-moments", moments, "float32"),
        tensor_payload(torch, "posterior-seed-42", noise, "float32"),
        tensor_payload(torch, "posterior-sample", sample, "float32"),
        tensor_payload(torch, "posterior-fp16-roundtrip", rounded_f16, "float16"),
        tensor_payload(torch, "latent-normalization", normalized_latents, "float32"),
        tensor_payload(torch, "decoded-frames", decoded, "float32"),
        tensor_payload(torch, "tile-seams", seams, "float32"),
    )
    validate_payload_shapes(payloads)
    del vae
    torch.cuda.empty_cache()
    return payloads


def capture_directory(
    fixture_root: pathlib.Path, case: Mapping[str, Any]
) -> pathlib.Path:
    parent = fixture_root / "captures" / LAYER_ID
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    if not BASE.is_relative_to(parent, fixture_root):
        fail("visual capture destination escapes MOLD_H3_FIXTURE_ROOT")
    directory = parent / case["input_sha256"][:24]
    directory.mkdir(mode=0o700, exist_ok=True)
    directory = directory.resolve(strict=True)
    metadata = directory.stat()
    if (
        not BASE.is_relative_to(directory, fixture_root)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        fail("visual capture directory must be external and owner-only")
    return directory


def write_request(
    path: pathlib.Path,
    authorization_sha256: str,
    source_revision: str,
    case: Mapping[str, Any],
) -> QWEN.SealedFileSnapshot:
    request = {
        "schema_version": REQUEST_SCHEMA,
        "authorization_document_sha256": authorization_sha256,
        "source_revision": source_revision,
        "model_revision": MODEL_REVISION,
        "checkpoint_components": [
            {"id": identifier, "relative_path": relative, "sha256": sha256}
            for identifier, relative, sha256 in VISUAL_SHARD_AUTHORITIES
        ],
        "cases": [dict(case)],
    }
    encoded = document_bytes(request)
    if len(encoded) > MAXIMUM_REQUEST_BYTES:
        fail("Mold visual capture request exceeds its emitted-size bound")
    BASE.exclusive_write(path, encoded)
    return QWEN.SealedFileSnapshot.seal(path)


def decode_raw_tensor(item: Mapping[str, Any]) -> TensorPayload:
    if set(item) != {"key", "shape", "dtype", "little_endian_base64"}:
        fail("Mold raw visual tensor has an unknown field")
    key = item.get("key")
    dtype = item.get("dtype")
    if key not in MEASUREMENT_DTYPES or dtype != MEASUREMENT_DTYPES[key]:
        fail("Mold raw visual tensor key or dtype drifted")
    shape = tuple(int(value) for value in item.get("shape", ()))
    if not shape or any(value <= 0 for value in shape):
        fail("Mold raw visual tensor has an invalid shape")
    try:
        encoded = base64.b64decode(item["little_endian_base64"], validate=True)
    except (TypeError, ValueError):
        fail("Mold raw visual tensor is not canonical base64")
    width = 2 if dtype == "float16" else 4
    if len(encoded) != math.prod(shape) * width:
        fail("Mold raw visual tensor byte count differs from its shape")
    payload = TensorPayload(str(key), shape, str(dtype), encoded)
    if not all(math.isfinite(value) for value in payload.values()):
        fail("Mold raw visual tensor contains non-finite values")
    return payload


def read_raw_output_bytes(
    encoded: bytes,
    authorization_sha256: str,
    source_revision: str,
    device: str,
    case: Mapping[str, Any],
) -> tuple[TensorPayload, ...]:
    try:
        if len(encoded) > MAXIMUM_RAW_OUTPUT_BYTES:
            fail("Mold raw visual capture exceeds its bounded output size")
        raw = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        fail("Mold raw visual capture is unavailable or malformed")
    expected_keys = {
        "schema_version",
        "claim_marker",
        "authorization_document_sha256",
        "source_revision",
        "model_revision",
        "component_index_sha256",
        "component_fingerprint_sha256",
        "device",
        "stored_weight_dtype",
        "encoder_compute_dtype",
        "decoder_compute_dtype",
        "attention_backend",
        "cases",
    }
    if set(raw) != expected_keys or (
        raw["schema_version"] != RAW_OUTPUT_SCHEMA
        or raw["claim_marker"] != CAPTURE_MARKER
        or raw["authorization_document_sha256"] != authorization_sha256
        or raw["source_revision"] != source_revision
        or raw["model_revision"] != MODEL_REVISION
        or raw["component_index_sha256"] != VISUAL_INDEX_SHA256
        or not lower_hex(raw["component_fingerprint_sha256"], 64)
        or raw["device"] != device
        or raw["stored_weight_dtype"] != "float32"
        or raw["encoder_compute_dtype"] != "float32"
        or raw["decoder_compute_dtype"] != "float16"
        or raw["attention_backend"] != "math"
    ):
        fail("Mold raw visual capture is not bound to the requested authority")
    if len(raw["cases"]) != 1:
        fail("Mold raw visual capture must contain one reviewed case")
    raw_case = raw["cases"][0]
    if set(raw_case) != {"case_id", "input_sha256", "tensors"} or (
        raw_case["case_id"] != case["case_id"]
        or raw_case["input_sha256"] != case["input_sha256"]
    ):
        fail("Mold raw visual capture case differs from its request")
    payloads = tuple(decode_raw_tensor(item) for item in raw_case["tensors"])
    validate_payload_shapes(payloads)
    return payloads


def capture_mold(
    fixture_root: pathlib.Path,
    capture_dir: pathlib.Path,
    authorization_path: pathlib.Path,
    authorization_sha256: str,
    model_root: pathlib.Path,
    case: Mapping[str, Any],
    device: str,
    source_revision: str,
    staged_mold_root: pathlib.Path,
) -> tuple[TensorPayload, ...]:
    request_path = capture_dir / "mold-request.json"
    raw_output_path = capture_dir / "mold-raw.json"
    if raw_output_path.exists():
        fail("Mold raw visual capture already exists; refusing to overwrite evidence")
    request_snapshot = write_request(
        request_path, authorization_sha256, source_revision, case
    )
    target_dir = fixture_root / "build" / LAYER_ID / source_revision
    target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target_dir = target_dir.resolve(strict=True)
    if not BASE.is_relative_to(target_dir, fixture_root):
        fail("Mold visual capture build directory escapes MOLD_H3_FIXTURE_ROOT")
    command = [
        "cargo",
        "run",
        "--quiet",
        "--locked",
        "--release",
        "--target-dir",
        str(target_dir),
        "-p",
        "mold-ai-inference",
        "--features",
        MOLD_FEATURES,
        "--bin",
        MOLD_BINARY,
        "--",
        "--model-root",
        str(model_root),
        "--fixture-root",
        str(fixture_root),
        "--repository-root",
        str(staged_mold_root),
        "--authorization-record",
        str(authorization_path),
        "--request",
        str(request_path),
        "--raw-output",
        str(raw_output_path),
        "--source-revision",
        source_revision,
        "--device",
        device,
    ]
    result = subprocess.run(
        command,
        cwd=staged_mold_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raw_output_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        fail("Mold visual capture adapter failed; inspect the protected runner log")
    try:
        request_snapshot.revalidate()
        raw_snapshot, raw_bytes = QWEN.SealedFileSnapshot.seal_and_read(
            raw_output_path, MAXIMUM_RAW_OUTPUT_BYTES
        )
        payloads = read_raw_output_bytes(
            raw_bytes,
            authorization_sha256,
            source_revision,
            device,
            case,
        )
        raw_snapshot.revalidate()
        request_snapshot.revalidate()
    except BaseException:
        raw_output_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        raise
    return payloads


def sample_indexes(payload: TensorPayload) -> list[int]:
    length = math.prod(payload.shape)
    if payload.key == "tile-seams":
        return list(range(length))
    count = min(257, length)
    if count == 1:
        return [0]
    return [index * (length - 1) // (count - 1) for index in range(count)]


def unravel_index(index: int, shape: Sequence[int]) -> list[int]:
    coordinates = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        coordinates[axis] = index % int(shape[axis])
        index //= int(shape[axis])
    return coordinates


def tensor_record(payload: TensorPayload) -> dict[str, Any]:
    values = payload.values()
    samples = []
    for flat_index in sample_indexes(payload):
        samples.append(
            {
                "index": unravel_index(flat_index, payload.shape),
                "value": float(values[flat_index]),
            }
        )
    return {
        "key": payload.key,
        "shape": list(payload.shape),
        "dtype": payload.dtype,
        "content_sha256": sha256_bytes(payload.encoded),
        "statistics": BASE.finite_statistics(values),
        "samples": samples,
    }


def acceleration_policy(manifest: Mapping[str, Any], role: str) -> dict[str, list[str]]:
    disabled = list(manifest["numerical_authority"]["excluded_accelerations"])
    enabled = (
        [
            "cuda",
            "deterministic-cuda",
            "pytorch-float32-encode",
            "pytorch-float16-autocast",
            "math-sdpa",
        ]
        if role == "oracle"
        else [
            "cuda",
            "deterministic-cuda",
            "candle-float32-encode",
            "candle-float16-decode",
            "math-attention",
        ]
    )
    return {"enabled": enabled, "disabled": disabled}


def capture_command(role: str, device: str) -> str:
    return (
        "python3 scripts/capture-minimax-h3-visual-vae.py capture "
        f"--role {role} --device {device}"
    )


def component_authorities(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    components = BASE.manifest_component_map(manifest)
    expected = {
        "official-video-vae-config": VISUAL_CONFIG_SHA256,
        "official-video-vae-index": VISUAL_INDEX_SHA256,
        **{
            identifier: sha256
            for identifier, _, sha256 in VISUAL_SHARD_AUTHORITIES
        },
    }
    authorities = []
    for identifier in COMPONENT_IDS:
        component = components.get(identifier)
        if component is None or component["sha256"] != expected[identifier]:
            fail("checked manifest lacks the exact visual-VAE component authority")
        authorities.append({"id": identifier, "sha256": component["sha256"]})
    return authorities


def build_layer_document(
    manifest: Mapping[str, Any],
    role: str,
    source_revision: str,
    authorization_sha256: str,
    device: str,
    artifact_set_sha256: str,
    case: Mapping[str, Any],
    payloads: Sequence[TensorPayload],
    runtime_identity: Mapping[str, str],
) -> dict[str, Any]:
    contract = layer_contract(manifest)
    authorities = component_authorities(manifest)
    authority_set_sha256 = BASE.component_authority_set_sha256(
        COMPONENT_IDS, BASE.manifest_component_map(manifest)
    )
    command = capture_command(role, device)
    adapter_sha256 = adapter_implementation_sha256(role)
    reviewed_runtime = manifest["numerical_authority"].get("oracle_runtime")
    extended_authority = isinstance(reviewed_runtime, dict)
    runtime_sha256 = sha256_bytes(
        canonical_json_bytes(
            reviewed_runtime if extended_authority else runtime_identity
        )
    )
    outputs = [tensor_record(payload) for payload in payloads]
    document: dict[str, Any] = {
        "schema_version": "mold.minimax-h3.layer-output.v1",
        "family": "minimax-h3",
        "case_id": case["case_id"],
        "layer": LAYER_ID,
        "authority_tier": "exact-full-bf16",
        "authorization_document_sha256": authorization_sha256,
        "input": {
            "id": case["case_id"],
            "sha256": case["input_sha256"],
            "component_index_sha256": authority_set_sha256,
            "component_indexes": authorities,
        },
        "producer": {
            "role": role,
            "implementation": (
                "diffusers-minimax-h3-visual-vae-f32-encode-fp16-decode"
                if role == "oracle"
                else "mold-candle-minimax-h3-visual-vae-f32-encode-fp16-decode"
            ),
            "source_id": "diffusers" if role == "oracle" else "mold",
            "revision": source_revision,
        },
        "adapter": {
            "schema_version": "mold.minimax-h3.layer-adapter.v1",
            "id": f"minimax-h3-visual-vae-{role}-f32-fp16-v1",
            "command": command,
            "tensor_hash_encoding": TENSOR_HASH_ENCODING,
        },
        "environment": {
            "device": device,
            "dtype": "official-f32-encode-fp16-decode",
            "attention_backend": "math",
            "acceleration_policy": acceleration_policy(manifest, role),
            "forbidden_accelerations_disabled": True,
        },
        "provenance": [],
        "outputs": outputs,
    }
    if extended_authority:
        document["adapter"]["implementation_sha256"] = adapter_sha256
        if role == "oracle":
            document["environment"]["oracle_runtime"] = dict(runtime_identity)
    provenance_values = {
        "source-revision": source_revision,
        "diffusers-revision": DIFFUSERS_REVISION,
        "adapter-implementation-sha256": adapter_sha256,
        "oracle-runtime-sha256": runtime_sha256,
        "component-index-hashes": ",".join(
            f"{component['id']}={component['sha256']}"
            for component in sorted(authorities, key=lambda item: item["id"])
        ),
        "artifact-set-sha256": artifact_set_sha256,
        "device": device,
        "dtype": "official-f32-encode-fp16-decode",
        "capture-command": command,
    }
    document["provenance"] = [
        {"key": key, "value": provenance_values[key]}
        for key in contract["required_provenance"]
    ]
    if role == "oracle":
        document["comparison"] = []
        for key in MEASUREMENTS:
            exact = key == "posterior-seed-42"
            document["comparison"].append(
                {
                    "key": key,
                    "absolute": 0.0 if exact else 0.015625,
                    "relative": 0.0 if exact else 0.015625,
                    "metric": "elementwise-atol-plus-rtol",
                    "hash_policy": "exact" if exact else "record-only",
                }
            )
    return document


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
    role: str,
    source_revision: str,
    authorization_sha256: str,
    device: str,
    fixture_root: pathlib.Path,
    path: pathlib.Path,
    document: Mapping[str, Any],
    encoded: bytes,
) -> dict[str, Any]:
    first = document["outputs"][0]
    bundle: dict[str, Any] = {
        "schema_version": "mold.minimax-h3.fixture-bundle.v1",
        "manifest_sha256": BASE.sha256_file(MANIFEST_PATH),
        "authorization_document_sha256": authorization_sha256,
        "capture_environment": {
            "framework": "diffusers" if role == "oracle" else "mold",
            "framework_revision": source_revision,
            "device": device,
            "dtype": "official-f32-encode-fp16-decode",
            "attention_backend": "math",
            "acceleration_policy": acceleration_policy(manifest, role),
            "command": capture_command(role, device),
            "forbidden_accelerations_disabled": True,
        },
        "fixtures": [
            {
                "id": f"{CASE_ID}-{role}",
                "layer": LAYER_ID,
                "authority_tier": "exact-full-bf16",
                "relative_path": path.relative_to(fixture_root).as_posix(),
                "sha256": sha256_bytes(encoded),
                "component_index_sha256": document["input"]["component_index_sha256"],
                "component_indexes": document["input"]["component_indexes"],
                "tensor": expected_tensor_summary(first),
                "tolerance": {
                    "absolute": 0.015625,
                    "relative": 0.015625,
                    "metric": "elementwise-atol-plus-rtol",
                },
            }
        ],
    }
    if role == "oracle" and "oracle_runtime" in document["environment"]:
        bundle["capture_environment"]["oracle_runtime"] = document["environment"][
            "oracle_runtime"
        ]
        bundle["capture_environment"]["oracle_adapters"] = [
            {
                "layer": LAYER_ID,
                "id": document["adapter"]["id"],
                "command": document["adapter"]["command"],
                "implementation_sha256": document["adapter"]["implementation_sha256"],
            }
        ]
    return bundle


def write_capture(
    fixture_root: pathlib.Path,
    capture_dir: pathlib.Path,
    authorization_path: pathlib.Path,
    manifest: Mapping[str, Any],
    role: str,
    source_revision: str,
    authorization_sha256: str,
    device: str,
    artifact_set_sha256: str,
    case: Mapping[str, Any],
    payloads: Sequence[TensorPayload],
    runtime_identity: Mapping[str, str],
    conformance_tool: Any,
) -> tuple[pathlib.Path, pathlib.Path]:
    document = build_layer_document(
        manifest,
        role,
        source_revision,
        authorization_sha256,
        device,
        artifact_set_sha256,
        case,
        payloads,
        runtime_identity,
    )
    conformance_tool.validate_schema(
        document, conformance_tool.LAYER_OUTPUT_SCHEMA_PATH
    )
    path = capture_dir / f"{role}-{CASE_ID}.json"
    encoded = document_bytes(document)
    bundle_path = capture_dir / f"{role}-bundle.json"
    if path.exists() or bundle_path.exists():
        fail("visual-VAE capture already exists; refusing to overwrite evidence")
    bundle = build_bundle(
        manifest,
        role,
        source_revision,
        authorization_sha256,
        device,
        fixture_root,
        path,
        document,
        encoded,
    )
    conformance_tool.validate_schema(bundle, conformance_tool.BUNDLE_SCHEMA_PATH)
    created: list[pathlib.Path] = []
    try:
        BASE.exclusive_write(path, encoded)
        created.append(path)
        BASE.exclusive_write(bundle_path, document_bytes(bundle))
        created.append(bundle_path)
        conformance_tool.validate_bundle(
            manifest, str(fixture_root), str(bundle_path), str(authorization_path)
        )
    except BaseException:
        for candidate in created:
            candidate.unlink(missing_ok=True)
        raise
    return path, bundle_path


def run_capture(
    environment: Mapping[str, str], role: str, device: str
) -> tuple[pathlib.Path, pathlib.Path]:
    if role not in {"oracle", "mold"}:
        fail("capture role must be oracle or mold")
    BASE.reject_excluded_acceleration_environment(environment)
    values = required_environment(environment)
    conformance_tool = load_module(
        "minimax_h3_visual_conformance", CONFORMANCE_TOOL_PATH
    )
    load_module("minimax_h3_visual_gpu_conformance", GPU_CONFORMANCE_TOOL_PATH)
    manifest = conformance_tool.validate_manifest()
    layer_contract(manifest)
    if not isinstance(manifest["numerical_authority"].get("oracle_runtime"), dict):
        fail("visual-VAE capture requires the reviewed exact oracle runtime authority")
    fixture_root = BASE.external_directory(
        "fixture root", values["MOLD_H3_FIXTURE_ROOT"]
    )
    authorization_path = BASE.external_file(
        "authorization record", values["MOLD_H3_AUTHORIZATION_RECORD"]
    )
    authorization = BASE.validate_reviewed_authorization(
        conformance_tool, authorization_path
    )
    if (
        authorization["source_document_sha256"]
        != REVIEWED_AUTHORIZATION_EVIDENCE_SHA256
    ):
        fail("visual-VAE capture authorization differs from the reviewed evidence")
    model_root = BASE.external_directory(
        "official model snapshot", values["MOLD_H3_OFFICIAL_MODEL"]
    )
    diffusers_checkout = BASE.external_directory(
        "Diffusers checkout", values["MOLD_H3_DIFFUSERS_CHECKOUT"]
    )
    transformers_checkout = BASE.external_directory(
        "Transformers checkout", values["MOLD_H3_TRANSFORMERS_CHECKOUT"]
    )
    BASE.verify_git_checkout(
        "Diffusers",
        diffusers_checkout,
        DIFFUSERS_REVISION,
        "https://github.com/huggingface/diffusers",
    )
    BASE.verify_git_checkout(
        "Transformers",
        transformers_checkout,
        TRANSFORMERS_REVISION,
        "https://github.com/huggingface/transformers",
    )
    BASE.verify_model_components(model_root, manifest)
    artifact_set_sha256, checkpoint_inputs = verify_visual_snapshot(model_root)
    case = build_case()
    capture_dir = capture_directory(fixture_root, case)
    mold_revision = ""
    if role == "mold":
        mold_revision = BASE.command_output(["git", "rev-parse", "HEAD"], REPO_ROOT)
        if not lower_hex(mold_revision, 40):
            fail("Mold checkout does not have a canonical source revision")

    with tempfile.TemporaryDirectory(
        prefix=".h3-visual-vae-stage-", dir=fixture_root
    ) as staged_value:
        staged_root = pathlib.Path(staged_value).resolve(strict=True)
        os.chmod(staged_root, 0o700)
        source_root = QWEN.private_source_root(staged_root)
        staged_diffusers = source_root / "diffusers"
        staged_transformers = source_root / "transformers"
        staged_mold = staged_root / "mold-source"
        QWEN.extract_package_snapshot(
            "Diffusers",
            diffusers_checkout,
            DIFFUSERS_REVISION,
            "https://github.com/huggingface/diffusers",
            "diffusers",
            staged_diffusers,
        )
        QWEN.extract_package_snapshot(
            "Transformers",
            transformers_checkout,
            TRANSFORMERS_REVISION,
            "https://github.com/huggingface/transformers",
            "transformers",
            staged_transformers,
        )
        if role == "mold":
            QWEN.extract_mold_snapshot(REPO_ROOT, mold_revision, staged_mold)
        staged_snapshot = QWEN.seal_staging(staged_root)
        runtime = load_runtime(staged_diffusers, staged_transformers)
        runtime_identity = observed_runtime(runtime)
        if runtime_identity != manifest["numerical_authority"]["oracle_runtime"]:
            fail("visual-VAE capture runtime differs from the reviewed oracle pin")
        if role == "oracle":
            source_revision = DIFFUSERS_REVISION
            payloads = capture_oracle(runtime, model_root, device)
        else:
            BASE.configure_torch(runtime, device)
            source_revision = mold_revision
            payloads = capture_mold(
                fixture_root,
                capture_dir,
                authorization_path,
                authorization["source_document_sha256"],
                model_root,
                case,
                device,
                source_revision,
                staged_mold,
            )
        staged_snapshot.revalidate()
        for identity in checkpoint_inputs:
            identity.revalidate()
        return write_capture(
            fixture_root,
            capture_dir,
            authorization_path,
            manifest,
            role,
            source_revision,
            authorization["source_document_sha256"],
            device,
            artifact_set_sha256,
            case,
            payloads,
            runtime_identity,
            conformance_tool,
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture authorization-bound visual-VAE evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser(
        "capture", help="capture the reviewed tiled temporal case"
    )
    capture.add_argument("--role", choices=("oracle", "mold"), required=True)
    capture.add_argument(
        "--device",
        default=os.environ.get("MOLD_H3_CAPTURE_DEVICE", "cuda:0"),
        help="indexed CUDA device; defaults to MOLD_H3_CAPTURE_DEVICE or cuda:0",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command != "capture":
            fail("unsupported visual-VAE capture command")
        layer_path, bundle_path = run_capture(os.environ, args.role, args.device)
        print(
            "MiniMax H3 visual-VAE capture complete: "
            f"role={args.role} layer={layer_path.name} bundle={bundle_path.name}"
        )
        return 0
    except (
        CaptureFailure,
        BASE.CaptureFailure,
        QWEN.CaptureFailure,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"MiniMax H3 visual-VAE capture rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
