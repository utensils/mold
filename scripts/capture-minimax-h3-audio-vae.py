#!/usr/bin/env python3
"""Capture authorization-bound exact-FP32 MiniMax H3 AudioVAE evidence.

The oracle loads the pinned official Diffusers AudioVAE. The Mold role invokes
an isolated private binary over the reviewed Comfy folded FP32 checkpoint.
Weights, waveform payloads, raw tensors, and captured evidence remain under an
owner-only external fixture root and are never copied into the repository.
"""

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
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any


sys.dont_write_bytecode = True
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN_PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-qwen-layer50.py"
CONFORMANCE_TOOL_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
GPU_CONFORMANCE_TOOL_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"
MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "minimax_h3" / "conformance-manifest.json"
)
QWEN_PRODUCER_SHA256 = (
    "5df8f9291260319e3b41044109013a51795fb551201d6007b9e2e0848af38d3b"
)

DIFFUSERS_REVISION = "9c6a68c32b3b2a64db91800b624d33cec6e25ab8"
MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
COMFY_MODEL_REVISION = "eb8a16107c595128b3a578f82d2ce2f75920c355"
OFFICIAL_AUDIO_CONFIG_SHA256 = (
    "9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6"
)
OFFICIAL_AUDIO_CHECKPOINT_SHA256 = (
    "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb"
)
OFFICIAL_AUDIO_CHECKPOINT_BYTES = 605_429_340
COMFY_AUDIO_CHECKPOINT_SHA256 = (
    "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
)
COMFY_AUDIO_CHECKPOINT_BYTES = 605_254_808
REVIEWED_AUTHORIZATION_EVIDENCE_SHA256 = (
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d"
)
REQUEST_SCHEMA = "mold.minimax-h3.audio-vae-capture-request.v1"
RAW_OUTPUT_SCHEMA = "mold.minimax-h3.audio-vae-raw-output.v1"
INPUT_IDENTITY_DOMAIN = b"mold.minimax-h3.audio-vae-input.v1\0"
CAPTURE_MARKER = "mold.minimax-h3.private-uat-exact-fp32-audio-vae-capture.v1"
LAYER_ID = "audio-vae"
CASE_ID = "audio-vae-stereo-batch-v1"
COMPONENT_ID = "official-audio-vae-config"
AUTHORITY_TIER = "exact-full-fp32"
CANONICAL_DTYPE = "float32"
TENSOR_HASH_ENCODING = "canonical-typed-le-v1"
MOLD_BINARY = "h3_audio_vae_capture"
MOLD_FEATURES = "dev-bins,h3-private-uat,cuda"
SAMPLE_RATE = 32_000
SAMPLES_PER_LATENT = 800
LATENT_ROWS_PER_SECOND = 40
LATENT_CHANNELS = 32
BATCH = 2
CHANNELS = 2
INPUT_SAMPLES = 2_401
PADDED_SAMPLES = 3_200
LATENT_ROWS = 4
MAXIMUM_RAW_OUTPUT_BYTES = 64 * 1024 * 1024
MAXIMUM_REQUEST_BYTES = 4 * 1024 * 1024
OFFICIAL_CONFIG_RELATIVE = "audio_vae/config.json"
OFFICIAL_CHECKPOINT_RELATIVE = "audio_vae/diffusion_pytorch_model.safetensors"
COMFY_CHECKPOINT_RELATIVE = "vae/minimax_h3_audio_vae_fp32.safetensors"
REQUIRED_ENVIRONMENT = (
    "MOLD_H3_FIXTURE_ROOT",
    "MOLD_H3_AUTHORIZATION_RECORD",
    "MOLD_H3_OFFICIAL_MODEL",
    "MOLD_H3_DIFFUSERS_CHECKOUT",
    "MOLD_H3_COMFY_MODEL",
)
EXPECTED_ENCODE_PHASES = [
    "encode-preprocess",
    "encode-dac",
    "encode-projection",
    "encode-posterior",
]
EXPECTED_DECODE_PHASES = [
    "decode-projection",
    "decode-bigvgan",
    "decode-output",
]
MEASUREMENT_ORDER = (
    "stereo-packing",
    "latent-rate",
    "waveform-statistics",
    "phase-polarity",
    "decoded-pcm",
    "sampled-values",
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load repository module {name}")
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
        raise RuntimeError(f"cannot read repository module {name}: {error}") from error
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"repository module {name} differs from the reviewed capture dependency"
        )
    return load_module(name, path)


QWEN = load_bound_module(
    "minimax_h3_audio_capture_qwen_base",
    QWEN_PRODUCER_PATH,
    QWEN_PRODUCER_SHA256,
)
BASE = QWEN.BASE
CaptureFailure = QWEN.CaptureFailure
ExternalFileIdentity = QWEN.ExternalFileIdentity
SealedFileSnapshot = QWEN.SealedFileSnapshot


def fail(message: str) -> None:
    raise CaptureFailure(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        fail("AudioVAE capture contains non-canonical JSON evidence")


def document_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def required_environment(environment: Mapping[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in REQUIRED_ENVIRONMENT:
        value = environment.get(name, "").strip()
        if not value:
            fail(f"{name} is required")
        values[name] = value
    return values


def component_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return BASE.manifest_component_map(manifest)


def layer_contract(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [layer for layer in manifest["fixture_layers"] if layer["id"] == LAYER_ID]
    if len(matches) != 1:
        fail("checked manifest does not contain one AudioVAE contract")
    layer = matches[0]
    if layer["authority_tier"] != AUTHORITY_TIER:
        fail("AudioVAE authority is not exact FP32")
    if tuple(layer["required_component_indexes"]) != (COMPONENT_ID,):
        fail("AudioVAE component authority drifted")
    if tuple(layer["required_measurements"]) != MEASUREMENT_ORDER:
        fail("AudioVAE measurement contract drifted")
    if tuple(layer["required_provenance"]) != (
        "source-revision",
        "checkpoint-sha256",
        "adapter-implementation-sha256",
        "runtime-sha256",
        "component-index-hash",
        "device",
        "dtype",
        "capture-command",
    ):
        fail("AudioVAE provenance contract drifted")
    return layer


def private_regular_file(
    label: str,
    root: pathlib.Path,
    relative: str,
    expected_size: int,
    expected_sha256: str,
    revision: str,
) -> tuple[pathlib.Path, tuple[ExternalFileIdentity, ...]]:
    candidate = root / relative
    if candidate.is_symlink():
        fail(f"{label} may not be a symbolic link")
    try:
        path = candidate.resolve(strict=True)
    except OSError:
        fail(f"{label} is unavailable")
    if not path.is_file() or not BASE.is_relative_to(path, root):
        fail(f"{label} escapes its external model root")
    before = ExternalFileIdentity.capture(path)
    if before.size != expected_size or BASE.sha256_file(path) != expected_sha256:
        fail(f"{label} differs from the reviewed artifact")
    after = ExternalFileIdentity.capture(path)
    if before != after:
        fail(f"{label} changed while it was authenticated")
    metadata_relative = f".cache/huggingface/download/{relative}.metadata"
    metadata_path = root / metadata_relative
    if metadata_path.is_symlink():
        fail(f"{label} revision metadata may not be a symbolic link")
    try:
        metadata_path = metadata_path.resolve(strict=True)
        metadata = metadata_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        fail(f"{label} lacks revision-bound metadata")
    metadata_identity = ExternalFileIdentity.capture(metadata_path)
    if not metadata or metadata[0] != revision:
        fail(f"{label} revision metadata drifted")
    if (
        len(metadata) > 1
        and lower_hex(metadata[1], 64)
        and metadata[1] != expected_sha256
    ):
        fail(f"{label} metadata hash differs from its reviewed artifact")
    return path, (after, metadata_identity)


def verify_audio_inputs(
    official_model: pathlib.Path,
    comfy_model: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, tuple[ExternalFileIdentity, ...]]:
    config_path = official_model / OFFICIAL_CONFIG_RELATIVE
    if config_path.is_symlink():
        fail("official AudioVAE config may not be a symbolic link")
    try:
        config_path = config_path.resolve(strict=True)
    except OSError:
        fail("official AudioVAE config is unavailable")
    config_identity = ExternalFileIdentity.capture(config_path)
    if BASE.sha256_file(config_path) != OFFICIAL_AUDIO_CONFIG_SHA256:
        fail("official AudioVAE config differs from the manifest authority")
    if (
        BASE.huggingface_metadata_revision(official_model, OFFICIAL_CONFIG_RELATIVE)
        != MODEL_REVISION
    ):
        fail("official AudioVAE config revision metadata drifted")
    official_checkpoint, official_identities = private_regular_file(
        "official AudioVAE checkpoint",
        official_model,
        OFFICIAL_CHECKPOINT_RELATIVE,
        OFFICIAL_AUDIO_CHECKPOINT_BYTES,
        OFFICIAL_AUDIO_CHECKPOINT_SHA256,
        MODEL_REVISION,
    )
    comfy_checkpoint, comfy_identities = private_regular_file(
        "Comfy folded AudioVAE checkpoint",
        comfy_model,
        COMFY_CHECKPOINT_RELATIVE,
        COMFY_AUDIO_CHECKPOINT_BYTES,
        COMFY_AUDIO_CHECKPOINT_SHA256,
        COMFY_MODEL_REVISION,
    )
    return (
        config_path,
        official_checkpoint,
        comfy_checkpoint,
        (config_identity, *official_identities, *comfy_identities),
    )


def put_u64(destination: bytearray, value: int) -> None:
    if not 0 <= value <= (1 << 64) - 1:
        fail("AudioVAE input identity integer exceeds u64")
    destination.extend(struct.pack("<Q", value))


def put_string(destination: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    put_u64(destination, len(encoded))
    destination.extend(encoded)


def deterministic_waveform() -> tuple[list[int], bytes]:
    values: list[float] = []
    for batch in range(BATCH):
        for channel in range(CHANNELS):
            selector = batch * CHANNELS + channel
            for sample in range(INPUT_SAMPLES):
                if selector == 0:
                    integer = ((sample * 17 + 11) % 257) - 128
                    value = integer / 512.0
                    if sample == 137:
                        value += 0.5
                elif selector == 1:
                    integer = ((sample * 29 + 7) % 251) - 125
                    value = -integer / 512.0
                    if sample == 311:
                        value -= 0.5
                elif selector == 2:
                    integer = ((sample * 43 + 19) % 241) - 120
                    value = integer / 512.0
                    if sample == 521:
                        value += 0.375
                else:
                    integer = ((sample * 43 + 19) % 241) - 120
                    value = -integer / 512.0
                    if sample == 521:
                        value -= 0.375
                canonical = struct.unpack("<f", struct.pack("<f", value))[0]
                if not math.isfinite(canonical) or not -1.0 <= canonical <= 1.0:
                    fail("deterministic AudioVAE waveform left the FP32 unit domain")
                values.append(canonical)
    encoded = b"".join(struct.pack("<f", value) for value in values)
    if len(encoded) != BATCH * CHANNELS * INPUT_SAMPLES * 4:
        fail("deterministic AudioVAE waveform has the wrong shape")
    return [BATCH, CHANNELS, INPUT_SAMPLES], encoded


def case_identity(case: Mapping[str, Any]) -> str:
    try:
        waveform = base64.b64decode(case["waveform_f32_le_base64"], validate=True)
    except (TypeError, ValueError):
        fail("AudioVAE waveform payload is not canonical base64")
    destination = bytearray(INPUT_IDENTITY_DOMAIN)
    put_string(destination, str(case["case_id"]))
    for key in ("sample_rate", "batch", "channels", "samples"):
        put_u64(destination, int(case[key]))
    put_u64(destination, len(waveform))
    destination.extend(waveform)
    return sha256_bytes(bytes(destination))


def build_case() -> dict[str, Any]:
    shape, waveform = deterministic_waveform()
    case = {
        "case_id": CASE_ID,
        "sample_rate": SAMPLE_RATE,
        "batch": shape[0],
        "channels": shape[1],
        "samples": shape[2],
        "waveform_f32_le_base64": base64.b64encode(waveform).decode("ascii"),
    }
    case["input_sha256"] = case_identity(case)
    return case


def case_set_identity(case: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "schema_version": "mold.minimax-h3.audio-vae-case-set.v1",
            "case_id": case["case_id"],
            "input_sha256": case["input_sha256"],
            "model_revision": MODEL_REVISION,
            "official_checkpoint_sha256": OFFICIAL_AUDIO_CHECKPOINT_SHA256,
            "comfy_checkpoint_sha256": COMFY_AUDIO_CHECKPOINT_SHA256,
        }
    )


def capture_directory(
    fixture_root: pathlib.Path, case: Mapping[str, Any]
) -> pathlib.Path:
    parent = fixture_root / "captures" / LAYER_ID
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700, follow_symlinks=False)
    except OSError as error:
        fail(f"cannot secure external AudioVAE capture directory: {error}")
    parent = parent.resolve(strict=True)
    if not BASE.is_relative_to(parent, fixture_root):
        fail("AudioVAE capture destination escapes MOLD_H3_FIXTURE_ROOT")
    directory = parent / case_set_identity(case)[:24]
    directory.mkdir(mode=0o700, exist_ok=True)
    try:
        os.chmod(directory, 0o700, follow_symlinks=False)
    except OSError as error:
        fail(f"cannot secure external AudioVAE case directory: {error}")
    directory = directory.resolve(strict=True)
    metadata = directory.stat()
    if (
        not BASE.is_relative_to(directory, fixture_root)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        fail("AudioVAE capture directory is not owner-only and external")
    return directory


def load_oracle_runtime(staged_diffusers: pathlib.Path) -> SimpleNamespace:
    source = staged_diffusers / "src"
    if not source.is_dir():
        fail("staged Diffusers checkout lacks its source package")
    sys.path.insert(0, str(source))
    if "diffusers" in sys.modules:
        fail("Diffusers was imported before AudioVAE provenance was verified")
    try:
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
        diffusers = importlib.import_module("diffusers")
        audio_module = importlib.import_module(
            "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio"
        )
    except (ImportError, RuntimeError) as error:
        fail(f"reviewed AudioVAE oracle runtime cannot be imported: {error}")
    BASE.verify_module_origin("Diffusers", diffusers, staged_diffusers)
    BASE.verify_module_origin(
        "Diffusers MiniMax H3 AudioVAE",
        audio_module.AutoencoderKLMiniMaxH3Audio,
        staged_diffusers,
    )
    return SimpleNamespace(
        torch=torch,
        numpy=numpy,
        diffusers=diffusers,
        model_class=audio_module.AutoencoderKLMiniMaxH3Audio,
        checkout=staged_diffusers,
    )


def observed_oracle_runtime(runtime: SimpleNamespace) -> dict[str, str]:
    cuda = getattr(getattr(runtime.torch, "version", None), "cuda", None)
    values = {
        "python": platform.python_version(),
        "torch": str(getattr(runtime.torch, "__version__", "")),
        "numpy": str(getattr(runtime.numpy, "__version__", "")),
        "cuda": "" if cuda is None else str(cuda),
        "diffusers": str(getattr(runtime.diffusers, "__version__", "")),
        "diffusers_revision": DIFFUSERS_REVISION,
        "execution": "official-audio-vae-fp32",
    }
    if any(not value for value in values.values()):
        fail("AudioVAE oracle runtime identity is incomplete")
    return values


def mold_runtime_identity(source_revision: str) -> dict[str, str]:
    values = {
        "mold_revision": source_revision,
        "rustc": BASE.command_output(["rustc", "--version"], REPO_ROOT),
        "cargo": BASE.command_output(["cargo", "--version"], REPO_ROOT),
        "cargo_lock_sha256": BASE.sha256_file(REPO_ROOT / "Cargo.lock"),
        "execution": "mold-candle-comfy-folded-audio-vae-fp32",
    }
    if any(not value for value in values.values()):
        fail("AudioVAE Mold runtime identity is incomplete")
    return values


def decode_f32_payload(value: str, shape: Sequence[int], label: str) -> list[float]:
    try:
        encoded = base64.b64decode(value, validate=True)
    except (TypeError, ValueError):
        fail(f"{label} is not canonical FP32 base64")
    if len(encoded) != math.prod(shape) * 4:
        fail(f"{label} byte count differs from its shape")
    values = [item[0] for item in struct.iter_unpack("<f", encoded)]
    if any(not math.isfinite(item) for item in values):
        fail(f"{label} contains a non-finite value")
    return values


def capture_oracle(
    runtime: SimpleNamespace,
    official_model: pathlib.Path,
    case: Mapping[str, Any],
    device_label: str,
) -> dict[str, Any]:
    torch = runtime.torch
    numpy = runtime.numpy
    device = BASE.configure_torch(runtime, device_label)
    try:
        model = runtime.model_class.from_pretrained(
            official_model / "audio_vae",
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device=device, dtype=torch.float32)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        fail(f"pinned official FP32 AudioVAE could not be loaded: {error}")
    BASE.verify_module_origin(
        "loaded official AudioVAE", model.__class__, runtime.checkout
    )
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if parameter_dtypes != {torch.float32} or getattr(model, "is_quantized", False):
        fail("Diffusers oracle loaded a quantized or non-FP32 AudioVAE")
    if (
        int(model.config.sampling_rate) != SAMPLE_RATE
        or int(model.hop_length) != SAMPLES_PER_LATENT
        or int(model.config.latent_channels) != LATENT_CHANNELS
    ):
        fail("Diffusers oracle AudioVAE geometry drifted")

    waveform_bytes = base64.b64decode(case["waveform_f32_le_base64"], validate=True)
    waveform_array = numpy.frombuffer(waveform_bytes, dtype="<f4").copy()
    waveform = torch.from_numpy(waveform_array).reshape(BATCH, CHANNELS, INPUT_SAMPLES)
    packed = waveform.reshape(BATCH * CHANNELS, 1, INPUT_SAMPLES).to(device)
    with torch.no_grad():
        posterior = model.encode(packed, return_dict=False)[0]
        denormalized = posterior.mode().float().cpu()
        if list(denormalized.shape) != [BATCH * CHANNELS, LATENT_CHANNELS, LATENT_ROWS]:
            fail("Diffusers oracle emitted the wrong mono-batch latent geometry")
        latent_mean_cpu = torch.tensor(
            model.config.latents_mean, dtype=torch.float32
        ).view(1, LATENT_CHANNELS, 1)
        latent_std_cpu = torch.tensor(
            model.config.latents_std, dtype=torch.float32
        ).view(1, LATENT_CHANNELS, 1)
        normalized_mono = (denormalized - latent_mean_cpu) / latent_std_cpu
        normalized = (
            normalized_mono.reshape(BATCH, CHANNELS, LATENT_CHANNELS, LATENT_ROWS)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        latent_mean_cuda = latent_mean_cpu.to(device)
        latent_std_cuda = latent_std_cpu.to(device)
        decode_input = normalized_mono.to(device) * latent_std_cuda + latent_mean_cuda
        decoded_mono = model.decode(decode_input, return_dict=False)[0].float().cpu()
        decoded = decoded_mono.reshape(BATCH, CHANNELS, PADDED_SAMPLES).contiguous()
        torch.cuda.synchronize(device)

    normalized_values = normalized.reshape(-1).tolist()
    decoded_values = decoded.reshape(-1).tolist()
    del model
    torch.cuda.empty_cache()
    return {
        "case_id": case["case_id"],
        "input_sha256": case["input_sha256"],
        "input_shape": [BATCH, CHANNELS, INPUT_SAMPLES],
        "packed_mono_shape": [BATCH * CHANNELS, 1, INPUT_SAMPLES],
        "normalized_latent_shape": [BATCH, LATENT_CHANNELS, CHANNELS, LATENT_ROWS],
        "normalized_latents": [float(value) for value in normalized_values],
        "decoded_pcm_shape": [BATCH, CHANNELS, PADDED_SAMPLES],
        "decoded_pcm": [float(value) for value in decoded_values],
        "encode_phases": list(EXPECTED_ENCODE_PHASES),
        "decode_phases": list(EXPECTED_DECODE_PHASES),
    }


def write_request(
    path: pathlib.Path,
    authorization_sha256: str,
    source_revision: str,
    case: Mapping[str, Any],
) -> SealedFileSnapshot:
    request = {
        "schema_version": REQUEST_SCHEMA,
        "authorization_document_sha256": authorization_sha256,
        "source_revision": source_revision,
        "model_revision": MODEL_REVISION,
        "config_sha256": OFFICIAL_AUDIO_CONFIG_SHA256,
        "checkpoint_sha256": COMFY_AUDIO_CHECKPOINT_SHA256,
        "cases": [dict(case)],
    }
    encoded = document_bytes(request)
    if len(encoded) > MAXIMUM_REQUEST_BYTES:
        fail("Mold AudioVAE request exceeds its emitted-size bound")
    BASE.exclusive_write(path, encoded)
    return SealedFileSnapshot.seal(path)


def seal_and_read_raw_output(
    path: pathlib.Path,
) -> tuple[SealedFileSnapshot, bytes]:
    """Retain the authenticated bytes so pathname replacement cannot alter parsing."""
    seal_and_read = getattr(SealedFileSnapshot, "seal_and_read", None)
    if seal_and_read is not None:
        return seal_and_read(path, MAXIMUM_RAW_OUTPUT_BYTES)

    # Compatibility for capture branches based on the first Qwen producer. The
    # current producer exposes the equivalent descriptor-backed helper above.
    try:
        os.chmod(path, 0o400, follow_symlinks=False)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        fail(f"cannot seal Mold raw AudioVAE capture: {error}")
    try:
        before = os.fstat(descriptor)
        identity = ExternalFileIdentity.capture(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_size > MAXIMUM_RAW_OUTPUT_BYTES
            or (before.st_dev, before.st_ino, before.st_size)
            != (identity.device, identity.inode, identity.size)
        ):
            fail("Mold raw AudioVAE capture is not a bounded owner-only regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            encoded = source.read(MAXIMUM_RAW_OUTPUT_BYTES + 1)
        after = os.fstat(descriptor)
        if len(encoded) > MAXIMUM_RAW_OUTPUT_BYTES or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            fail("Mold raw AudioVAE capture changed while it was retained")
    finally:
        os.close(descriptor)
    snapshot = SealedFileSnapshot(identity, sha256_bytes(encoded))
    snapshot.revalidate()
    return snapshot, encoded


def read_raw_output_bytes(
    encoded: bytes,
    authorization_sha256: str,
    source_revision: str,
    device: str,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        raw = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        fail("Mold raw AudioVAE capture is unavailable or malformed")
    expected_keys = {
        "schema_version",
        "claim_marker",
        "authorization_document_sha256",
        "source_revision",
        "model_revision",
        "config_sha256",
        "checkpoint_sha256",
        "device",
        "dtype",
        "tensor_layout",
        "sample_rate",
        "samples_per_latent",
        "latent_rows_per_second",
        "encode_phases",
        "decode_phases",
        "cases",
    }
    if set(raw) != expected_keys or (
        raw["schema_version"] != RAW_OUTPUT_SCHEMA
        or raw["claim_marker"] != CAPTURE_MARKER
        or raw["authorization_document_sha256"] != authorization_sha256
        or raw["source_revision"] != source_revision
        or raw["model_revision"] != MODEL_REVISION
        or raw["config_sha256"] != OFFICIAL_AUDIO_CONFIG_SHA256
        or raw["checkpoint_sha256"] != COMFY_AUDIO_CHECKPOINT_SHA256
        or raw["device"] != device
        or raw["dtype"] != CANONICAL_DTYPE
        or raw["tensor_layout"] != "comfy-folded-weight-norm"
        or raw["sample_rate"] != SAMPLE_RATE
        or raw["samples_per_latent"] != SAMPLES_PER_LATENT
        or raw["latent_rows_per_second"] != LATENT_ROWS_PER_SECOND
        or raw["encode_phases"] != EXPECTED_ENCODE_PHASES
        or raw["decode_phases"] != EXPECTED_DECODE_PHASES
        or len(raw["cases"]) != 1
    ):
        fail("Mold raw AudioVAE output is not bound to the exact FP32 authority")
    item = raw["cases"][0]
    expected_case_keys = {
        "case_id",
        "input_sha256",
        "input_shape",
        "packed_mono_shape",
        "normalized_latent_shape",
        "normalized_latents_f32_le_base64",
        "decoded_pcm_shape",
        "decoded_pcm_f32_le_base64",
    }
    if set(item) != expected_case_keys or (
        item["case_id"] != case["case_id"]
        or item["input_sha256"] != case["input_sha256"]
        or item["input_shape"] != [BATCH, CHANNELS, INPUT_SAMPLES]
        or item["packed_mono_shape"] != [BATCH * CHANNELS, 1, INPUT_SAMPLES]
        or item["normalized_latent_shape"]
        != [BATCH, LATENT_CHANNELS, CHANNELS, LATENT_ROWS]
        or item["decoded_pcm_shape"] != [BATCH, CHANNELS, PADDED_SAMPLES]
    ):
        fail("Mold raw AudioVAE case differs from its reviewed input geometry")
    return {
        "case_id": item["case_id"],
        "input_sha256": item["input_sha256"],
        "input_shape": item["input_shape"],
        "packed_mono_shape": item["packed_mono_shape"],
        "normalized_latent_shape": item["normalized_latent_shape"],
        "normalized_latents": decode_f32_payload(
            item["normalized_latents_f32_le_base64"],
            item["normalized_latent_shape"],
            "Mold normalized AudioVAE latents",
        ),
        "decoded_pcm_shape": item["decoded_pcm_shape"],
        "decoded_pcm": decode_f32_payload(
            item["decoded_pcm_f32_le_base64"],
            item["decoded_pcm_shape"],
            "Mold decoded AudioVAE PCM",
        ),
        "encode_phases": raw["encode_phases"],
        "decode_phases": raw["decode_phases"],
    }


def capture_mold(
    fixture_root: pathlib.Path,
    capture_dir: pathlib.Path,
    authorization_path: pathlib.Path,
    authorization_sha256: str,
    config_path: pathlib.Path,
    comfy_checkpoint: pathlib.Path,
    case: Mapping[str, Any],
    device: str,
    source_revision: str,
    staged_mold_root: pathlib.Path,
) -> dict[str, Any]:
    if not lower_hex(source_revision, 40):
        fail("Mold checkout does not have a canonical source revision")
    request_path = capture_dir / "mold-audio-vae-request.json"
    raw_output_path = capture_dir / "mold-audio-vae-raw.json"
    if request_path.exists() or raw_output_path.exists():
        fail("Mold raw AudioVAE capture already exists; refusing to overwrite evidence")
    request_snapshot = write_request(
        request_path, authorization_sha256, source_revision, case
    )
    target_dir = fixture_root / "build" / LAYER_ID / source_revision
    target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(target_dir, 0o700, follow_symlinks=False)
    except OSError as error:
        fail(f"cannot secure Mold AudioVAE target directory: {error}")
    target_dir = target_dir.resolve(strict=True)
    if not BASE.is_relative_to(target_dir, fixture_root):
        fail("Mold AudioVAE target directory escapes MOLD_H3_FIXTURE_ROOT")
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
        "--config",
        str(config_path),
        "--weights",
        str(comfy_checkpoint),
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
    try:
        result = subprocess.run(
            command,
            cwd=staged_mold_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        request_path.unlink(missing_ok=True)
        fail("Mold exact-FP32 AudioVAE adapter could not be started")
    if result.returncode != 0:
        raw_output_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        fail(
            "Mold exact-FP32 AudioVAE adapter failed; inspect the protected runner log"
        )
    try:
        request_snapshot.revalidate()
        raw_snapshot, raw_output_bytes = seal_and_read_raw_output(raw_output_path)
        captured = read_raw_output_bytes(
            raw_output_bytes,
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
    return captured


def canonical_f32(value: float) -> float:
    converted = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    if not math.isfinite(converted):
        fail("AudioVAE evidence contains a non-finite FP32 value")
    return converted


def flat_coordinate(index: int, shape: Sequence[int]) -> list[int]:
    coordinate = [0] * len(shape)
    remainder = index
    for axis in range(len(shape) - 1, -1, -1):
        coordinate[axis] = remainder % int(shape[axis])
        remainder //= int(shape[axis])
    if remainder:
        fail("AudioVAE tensor sample index exceeds its shape")
    return coordinate


def float32_tensor_record(
    key: str,
    shape: Sequence[int],
    values: Sequence[float],
    sample_all: bool,
) -> dict[str, Any]:
    canonical = [canonical_f32(value) for value in values]
    shape = [int(value) for value in shape]
    if (
        not canonical
        or math.prod(shape) != len(canonical)
        or any(value <= 0 for value in shape)
    ):
        fail(f"{key} FP32 tensor shape differs from its payload")
    encoded = b"".join(struct.pack("<f", value) for value in canonical)
    indexes = (
        list(range(len(canonical)))
        if sample_all
        else BASE.sample_indexes(len(canonical))
    )
    return {
        "key": key,
        "shape": shape,
        "dtype": CANONICAL_DTYPE,
        "content_sha256": sha256_bytes(encoded),
        "statistics": BASE.finite_statistics(canonical),
        "samples": [
            {"index": flat_coordinate(index, shape), "value": canonical[index]}
            for index in indexes
        ],
    }


def int64_tensor_record(key: str, values: Sequence[int]) -> dict[str, Any]:
    converted = [int(value) for value in values]
    if not converted or any(not -(2**63) <= value <= 2**63 - 1 for value in converted):
        fail(f"{key} evidence is outside signed int64")
    digest = hashlib.sha256()
    for value in converted:
        digest.update(struct.pack("<q", value))
    return {
        "key": key,
        "shape": [len(converted)],
        "dtype": "int64",
        "content_sha256": digest.hexdigest(),
        "statistics": BASE.finite_statistics(converted),
        "samples": [
            {"index": [index], "value": value} for index, value in enumerate(converted)
        ],
    }


def float64_tensor_record(
    key: str, shape: Sequence[int], values: Sequence[float]
) -> dict[str, Any]:
    converted = [float(value) for value in values]
    shape = [int(value) for value in shape]
    if (
        not converted
        or math.prod(shape) != len(converted)
        or any(value <= 0 for value in shape)
        or any(not math.isfinite(value) for value in converted)
    ):
        fail(f"{key} evidence must contain finite shaped float64 values")
    encoded = b"".join(struct.pack("<d", value) for value in converted)
    return {
        "key": key,
        "shape": shape,
        "dtype": "float64",
        "content_sha256": sha256_bytes(encoded),
        "statistics": BASE.finite_statistics(converted),
        "samples": [
            {"index": flat_coordinate(index, shape), "value": converted[index]}
            for index in range(len(converted))
        ],
    }


def channel_values(values: Sequence[float], shape: Sequence[int]) -> list[list[float]]:
    if list(shape[:2]) != [BATCH, CHANNELS] or math.prod(shape) != len(values):
        fail("AudioVAE waveform evidence has invalid stereo geometry")
    samples = int(shape[2])
    return [
        [float(value) for value in values[index * samples : (index + 1) * samples]]
        for index in range(BATCH * CHANNELS)
    ]


def waveform_statistics(channels: Sequence[Sequence[float]]) -> list[float]:
    values: list[float] = []
    for channel in channels:
        if not channel:
            fail("AudioVAE waveform statistics received an empty channel")
        mean = math.fsum(channel) / len(channel)
        variance = math.fsum((value - mean) ** 2 for value in channel) / len(channel)
        rms = math.sqrt(math.fsum(value * value for value in channel) / len(channel))
        values.extend(
            [
                min(channel),
                max(channel),
                mean,
                math.sqrt(variance),
                rms,
                max(map(abs, channel)),
            ]
        )
    return values


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        fail("AudioVAE phase comparison received mismatched channels")
    numerator = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        fail("AudioVAE phase comparison received a silent channel")
    return numerator / (left_norm * right_norm)


def phase_polarity_metrics(
    input_channels: Sequence[Sequence[float]],
    decoded_channels: Sequence[Sequence[float]],
) -> list[float]:
    if (
        len(input_channels) != BATCH * CHANNELS
        or len(decoded_channels) != BATCH * CHANNELS
    ):
        fail("AudioVAE phase evidence lacks the complete stereo batch")
    values: list[float] = []
    max_lag = 128
    for source, decoded_full in zip(input_channels, decoded_channels, strict=True):
        decoded = decoded_full[: len(source)]
        zero = cosine_similarity(source, decoded)
        candidates: list[tuple[float, int, float]] = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                left = source[-lag:]
                right = decoded[: len(decoded) + lag]
            elif lag > 0:
                left = source[: len(source) - lag]
                right = decoded[lag:]
            else:
                left = source
                right = decoded
            correlation = cosine_similarity(left, right)
            candidates.append((abs(correlation), lag, correlation))
        _, best_lag, best = max(
            candidates, key=lambda value: (value[0], -abs(value[1]), -value[1])
        )
        polarity = 1.0 if best > 0 else -1.0 if best < 0 else 0.0
        values.extend([float(best_lag), best, zero, polarity])
    inverse_pair = cosine_similarity(decoded_channels[2], decoded_channels[3])
    values.extend(
        [inverse_pair, 1.0 if inverse_pair > 0 else -1.0 if inverse_pair < 0 else 0.0]
    )
    return values


def measurement_outputs(
    case: Mapping[str, Any], captured: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if (
        captured["input_shape"] != [BATCH, CHANNELS, INPUT_SAMPLES]
        or captured["packed_mono_shape"] != [BATCH * CHANNELS, 1, INPUT_SAMPLES]
        or captured["normalized_latent_shape"]
        != [BATCH, LATENT_CHANNELS, CHANNELS, LATENT_ROWS]
        or captured["decoded_pcm_shape"] != [BATCH, CHANNELS, PADDED_SAMPLES]
        or captured["encode_phases"] != EXPECTED_ENCODE_PHASES
        or captured["decode_phases"] != EXPECTED_DECODE_PHASES
    ):
        fail("AudioVAE capture geometry or phase trace drifted")
    input_values = decode_f32_payload(
        case["waveform_f32_le_base64"],
        [BATCH, CHANNELS, INPUT_SAMPLES],
        "prepared AudioVAE waveform",
    )
    decoded_channels = channel_values(
        captured["decoded_pcm"], captured["decoded_pcm_shape"]
    )
    input_channels = channel_values(input_values, [BATCH, CHANNELS, INPUT_SAMPLES])
    packing = [
        BATCH,
        CHANNELS,
        INPUT_SAMPLES,
        BATCH * CHANNELS,
        1,
        INPUT_SAMPLES,
    ]
    for batch in range(BATCH):
        for channel in range(CHANNELS):
            packing.extend([batch, channel, batch * CHANNELS + channel])
    waveform_stats = waveform_statistics(input_channels) + waveform_statistics(
        decoded_channels
    )
    return [
        int64_tensor_record("stereo-packing", packing),
        float64_tensor_record(
            "latent-rate",
            [7],
            [
                SAMPLE_RATE,
                SAMPLES_PER_LATENT,
                LATENT_ROWS_PER_SECOND,
                INPUT_SAMPLES,
                PADDED_SAMPLES,
                LATENT_ROWS,
                PADDED_SAMPLES,
            ],
        ),
        float64_tensor_record(
            "waveform-statistics", [2, BATCH, CHANNELS, 6], waveform_stats
        ),
        float64_tensor_record(
            "phase-polarity",
            [BATCH * CHANNELS * 4 + 2],
            phase_polarity_metrics(input_channels, decoded_channels),
        ),
        float32_tensor_record(
            "decoded-pcm",
            captured["decoded_pcm_shape"],
            captured["decoded_pcm"],
            sample_all=True,
        ),
        float32_tensor_record(
            "sampled-values",
            captured["normalized_latent_shape"],
            captured["normalized_latents"],
            sample_all=True,
        ),
    ]


def acceleration_policy(manifest: Mapping[str, Any], role: str) -> dict[str, list[str]]:
    disabled = list(manifest["numerical_authority"]["excluded_accelerations"])
    if len(disabled) != len(set(disabled)):
        fail("checked manifest repeats acceleration exclusions")
    enabled = (
        ["cuda", "deterministic-cuda", "pytorch-float32", "official-weight-norm"]
        if role == "oracle"
        else [
            "cuda",
            "deterministic-cuda",
            "candle-float32",
            "comfy-folded-weight-norm",
        ]
    )
    return {"enabled": enabled, "disabled": disabled}


def capture_command(role: str, device: str) -> str:
    return (
        "python3 scripts/capture-minimax-h3-audio-vae.py capture "
        f"--role {role} --device {device}"
    )


def adapter_implementation_sha256(role: str) -> str:
    values = {
        "schema_version": f"mold.minimax-h3.audio-vae-{role}-adapter.v1",
        "producer_sha256": BASE.sha256_file(pathlib.Path(__file__).resolve()),
        "qwen_capture_dependency_sha256": QWEN_PRODUCER_SHA256,
    }
    if role == "oracle":
        values["diffusers_audio_vae_sha256"] = BASE.sha256_file(
            pathlib.Path(os.environ["MOLD_H3_DIFFUSERS_CHECKOUT"])
            / "src"
            / "diffusers"
            / "models"
            / "autoencoders"
            / "autoencoder_kl_minimax_h3_audio.py"
        )
    else:
        values.update(
            {
                "rust_adapter_sha256": BASE.sha256_file(
                    REPO_ROOT
                    / "crates"
                    / "mold-inference"
                    / "src"
                    / "bin"
                    / "h3_audio_vae_capture.rs"
                ),
                "cargo_manifest_sha256": BASE.sha256_file(
                    REPO_ROOT / "crates" / "mold-inference" / "Cargo.toml"
                ),
                "cargo_lock_sha256": BASE.sha256_file(REPO_ROOT / "Cargo.lock"),
            }
        )
    return sha256_json(values)


def comparison_policy(key: str) -> dict[str, Any]:
    if key in {"stereo-packing", "latent-rate"}:
        absolute = 0.0
        relative = 0.0
        hash_policy = "exact"
    elif key in {"waveform-statistics", "phase-polarity", "sampled-values"}:
        absolute = 1.0 / 1024.0
        relative = 1.0 / 1024.0
        hash_policy = "record-only"
    elif key == "decoded-pcm":
        absolute = 1.0 / 512.0
        relative = 1.0 / 512.0
        hash_policy = "record-only"
    else:
        fail(f"AudioVAE comparison policy lacks measurement {key}")
    return {
        "key": key,
        "absolute": absolute,
        "relative": relative,
        "metric": "elementwise-atol-plus-rtol",
        "hash_policy": hash_policy,
    }


def build_layer_document(
    manifest: Mapping[str, Any],
    role: str,
    source_revision: str,
    authorization_sha256: str,
    device: str,
    case: Mapping[str, Any],
    captured: Mapping[str, Any],
    runtime_identity: Mapping[str, str],
    implementation_sha256: str | None = None,
) -> dict[str, Any]:
    contract = layer_contract(manifest)
    components = component_map(manifest)
    component = components.get(COMPONENT_ID)
    if component is None or component["sha256"] != OFFICIAL_AUDIO_CONFIG_SHA256:
        fail("checked manifest lacks the exact official AudioVAE config authority")
    command = capture_command(role, device)
    outputs = measurement_outputs(case, captured)
    checkpoint_sha256 = (
        OFFICIAL_AUDIO_CHECKPOINT_SHA256
        if role == "oracle"
        else COMFY_AUDIO_CHECKPOINT_SHA256
    )
    implementation_sha256 = implementation_sha256 or adapter_implementation_sha256(role)
    runtime_identity = dict(runtime_identity)
    runtime_sha256 = sha256_json(runtime_identity)
    provenance_values = {
        "source-revision": source_revision,
        "checkpoint-sha256": checkpoint_sha256,
        "adapter-implementation-sha256": implementation_sha256,
        "runtime-sha256": runtime_sha256,
        "component-index-hash": component["sha256"],
        "device": device,
        "dtype": CANONICAL_DTYPE,
        "capture-command": command,
    }
    document: dict[str, Any] = {
        "schema_version": "mold.minimax-h3.layer-output.v1",
        "family": "minimax-h3",
        "case_id": case["case_id"],
        "layer": LAYER_ID,
        "authority_tier": AUTHORITY_TIER,
        "authorization_document_sha256": authorization_sha256,
        "input": {
            "id": case["case_id"],
            "sha256": case["input_sha256"],
            "component_index_sha256": component["sha256"],
            "component_indexes": [{"id": COMPONENT_ID, "sha256": component["sha256"]}],
            "checkpoint_sha256": checkpoint_sha256,
        },
        "producer": {
            "role": role,
            "implementation": (
                "diffusers-minimax-h3-audio-vae-official-fp32"
                if role == "oracle"
                else "mold-candle-minimax-h3-audio-vae-comfy-folded-fp32"
            ),
            "source_id": "diffusers" if role == "oracle" else "mold",
            "revision": source_revision,
        },
        "adapter": {
            "schema_version": "mold.minimax-h3.layer-adapter.v1",
            "id": f"minimax-h3-audio-vae-{role}-exact-fp32-v1",
            "command": command,
            "tensor_hash_encoding": TENSOR_HASH_ENCODING,
            "implementation_sha256": implementation_sha256,
        },
        "environment": {
            "device": device,
            "dtype": CANONICAL_DTYPE,
            "attention_backend": "math",
            "acceleration_policy": acceleration_policy(manifest, role),
            "runtime": runtime_identity,
            "forbidden_accelerations_disabled": True,
        },
        "provenance": [
            {"key": key, "value": provenance_values[key]}
            for key in contract["required_provenance"]
        ],
        "outputs": outputs,
    }
    if role == "oracle":
        document["comparison"] = [comparison_policy(key) for key in MEASUREMENT_ORDER]
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
    document_path: pathlib.Path,
    document: Mapping[str, Any],
    encoded: bytes,
) -> dict[str, Any]:
    first = document["outputs"][0]
    first_policy = comparison_policy(first["key"])
    return {
        "schema_version": "mold.minimax-h3.fixture-bundle.v1",
        "manifest_sha256": BASE.sha256_file(MANIFEST_PATH),
        "authorization_document_sha256": authorization_sha256,
        "capture_environment": {
            "framework": "diffusers" if role == "oracle" else "mold",
            "framework_revision": source_revision,
            "device": device,
            "dtype": CANONICAL_DTYPE,
            "attention_backend": "math",
            "acceleration_policy": acceleration_policy(manifest, role),
            "command": capture_command(role, device),
            "forbidden_accelerations_disabled": True,
        },
        "fixtures": [
            {
                "id": f"{CASE_ID}-{role}",
                "layer": LAYER_ID,
                "authority_tier": AUTHORITY_TIER,
                "relative_path": document_path.relative_to(fixture_root).as_posix(),
                "sha256": sha256_bytes(encoded),
                "component_index_sha256": OFFICIAL_AUDIO_CONFIG_SHA256,
                "component_indexes": document["input"]["component_indexes"],
                "tensor": expected_tensor_summary(first),
                "tolerance": {
                    "absolute": first_policy["absolute"],
                    "relative": first_policy["relative"],
                    "metric": first_policy["metric"],
                },
            }
        ],
    }


def write_capture(
    fixture_root: pathlib.Path,
    capture_dir: pathlib.Path,
    authorization_path: pathlib.Path,
    manifest: Mapping[str, Any],
    role: str,
    source_revision: str,
    authorization_sha256: str,
    device: str,
    case: Mapping[str, Any],
    captured: Mapping[str, Any],
    runtime_identity: Mapping[str, str],
    implementation_sha256: str,
    conformance_tool: Any,
    gpu_tool: Any,
) -> tuple[pathlib.Path, pathlib.Path]:
    contract = gpu_tool.exact_layer_contracts(manifest)[LAYER_ID]
    document = build_layer_document(
        manifest,
        role,
        source_revision,
        authorization_sha256,
        device,
        case,
        captured,
        runtime_identity,
        implementation_sha256,
    )
    gpu_tool.validate_manifest_layer_evidence(document, contract, role)
    conformance_tool.validate_schema(
        document, conformance_tool.LAYER_OUTPUT_SCHEMA_PATH
    )
    document_path = capture_dir / f"{role}-{CASE_ID}.json"
    bundle_path = capture_dir / f"{role}-bundle.json"
    if document_path.exists() or bundle_path.exists():
        fail("AudioVAE layer capture already exists; refusing to overwrite evidence")
    encoded = document_bytes(document)
    bundle = build_bundle(
        manifest,
        role,
        source_revision,
        authorization_sha256,
        device,
        fixture_root,
        document_path,
        document,
        encoded,
    )
    conformance_tool.validate_schema(bundle, conformance_tool.BUNDLE_SCHEMA_PATH)
    created: list[pathlib.Path] = []
    try:
        BASE.exclusive_write(document_path, encoded)
        created.append(document_path)
        BASE.exclusive_write(bundle_path, document_bytes(bundle))
        created.append(bundle_path)
        conformance_tool.validate_bundle(
            manifest,
            str(fixture_root),
            str(bundle_path),
            str(authorization_path),
        )
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return document_path, bundle_path


def run_capture(
    environment: Mapping[str, str], role: str, device: str
) -> tuple[pathlib.Path, pathlib.Path]:
    if role not in {"oracle", "mold"}:
        fail("capture role must be oracle or mold")
    BASE.reject_excluded_acceleration_environment(environment)
    values = required_environment(environment)
    conformance_tool = load_module(
        "minimax_h3_audio_conformance", CONFORMANCE_TOOL_PATH
    )
    gpu_tool = load_module(
        "minimax_h3_audio_gpu_conformance", GPU_CONFORMANCE_TOOL_PATH
    )
    manifest = conformance_tool.validate_manifest()
    layer_contract(manifest)
    if conformance_tool.EXPECTED_REVISIONS["diffusers"] != DIFFUSERS_REVISION:
        fail("AudioVAE capture Diffusers revision differs from the checked pin")
    if conformance_tool.EXPECTED_REVISIONS["minimax-official-model"] != MODEL_REVISION:
        fail("AudioVAE capture model revision differs from the checked pin")
    if conformance_tool.EXPECTED_REVISIONS["comfy-checkpoints"] != COMFY_MODEL_REVISION:
        fail("AudioVAE capture Comfy revision differs from the checked pin")

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
        fail("AudioVAE capture authorization differs from the reviewed evidence")
    authorization_source = BASE.external_file(
        "authorization source document", authorization["source_document_path"]
    )
    official_model = BASE.external_directory(
        "official model snapshot", values["MOLD_H3_OFFICIAL_MODEL"]
    )
    comfy_model = BASE.external_directory(
        "Comfy model snapshot", values["MOLD_H3_COMFY_MODEL"]
    )
    diffusers_checkout = BASE.external_directory(
        "Diffusers checkout", values["MOLD_H3_DIFFUSERS_CHECKOUT"]
    )
    BASE.verify_git_checkout(
        "Diffusers",
        diffusers_checkout,
        DIFFUSERS_REVISION,
        "https://github.com/huggingface/diffusers",
    )
    BASE.verify_model_components(official_model, manifest)
    config_path, _official_checkpoint, comfy_checkpoint, model_identities = (
        verify_audio_inputs(official_model, comfy_model)
    )
    protected_inputs = [
        ExternalFileIdentity.capture(MANIFEST_PATH),
        ExternalFileIdentity.capture(pathlib.Path(__file__).resolve()),
        ExternalFileIdentity.capture(authorization_path),
        ExternalFileIdentity.capture(authorization_source),
        *model_identities,
    ]
    protected_hashes = {
        identity.path: BASE.sha256_file(identity.path) for identity in protected_inputs
    }

    mold_revision = ""
    if role == "mold":
        mold_revision = BASE.command_output(["git", "rev-parse", "HEAD"], REPO_ROOT)
        if not lower_hex(mold_revision, 40):
            fail("Mold checkout does not have a canonical source revision")
        BASE.verify_git_checkout(
            "Mold", REPO_ROOT, mold_revision, "https://github.com/utensils/mold"
        )
        for path in (
            REPO_ROOT
            / "crates"
            / "mold-inference"
            / "src"
            / "bin"
            / "h3_audio_vae_capture.rs",
            REPO_ROOT / "crates" / "mold-inference" / "Cargo.toml",
            REPO_ROOT / "Cargo.lock",
        ):
            identity = ExternalFileIdentity.capture(path)
            protected_inputs.append(identity)
            protected_hashes[path] = BASE.sha256_file(path)

    with tempfile.TemporaryDirectory(
        prefix=".h3-audio-vae-stage-", dir=fixture_root
    ) as staged_value:
        staged_root = pathlib.Path(staged_value).resolve(strict=True)
        os.chmod(staged_root, 0o700)
        source_root = QWEN.private_source_root(staged_root)
        staged_diffusers = source_root / "diffusers"
        staged_mold = staged_root / "mold-source"
        QWEN.extract_package_snapshot(
            "Diffusers",
            diffusers_checkout,
            DIFFUSERS_REVISION,
            "https://github.com/huggingface/diffusers",
            "diffusers",
            staged_diffusers,
        )
        if role == "mold":
            QWEN.extract_mold_snapshot(REPO_ROOT, mold_revision, staged_mold)
        staged_snapshot = QWEN.seal_staging(staged_root)
        case = build_case()
        capture_dir = capture_directory(fixture_root, case)
        implementation_sha256 = adapter_implementation_sha256(role)
        if role == "oracle":
            runtime = load_oracle_runtime(staged_diffusers)
            runtime_identity = observed_oracle_runtime(runtime)
            captured = capture_oracle(runtime, official_model, case, device)
            source_revision = DIFFUSERS_REVISION
        else:
            runtime_identity = mold_runtime_identity(mold_revision)
            captured = capture_mold(
                fixture_root,
                capture_dir,
                authorization_path,
                authorization["source_document_sha256"],
                config_path,
                comfy_checkpoint,
                case,
                device,
                mold_revision,
                staged_mold,
            )
            source_revision = mold_revision

        staged_snapshot.revalidate()
        for identity in protected_inputs:
            identity.revalidate()
            if BASE.sha256_file(identity.path) != protected_hashes[identity.path]:
                fail("protected AudioVAE capture input changed during execution")
        return write_capture(
            fixture_root,
            capture_dir,
            authorization_path,
            manifest,
            role,
            source_revision,
            authorization["source_document_sha256"],
            device,
            case,
            captured,
            runtime_identity,
            implementation_sha256,
            conformance_tool,
            gpu_tool,
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture authorization-bound exact-FP32 MiniMax H3 AudioVAE evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser(
        "capture", help="capture deterministic stereo-batch encode/decode evidence"
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
            fail("unsupported AudioVAE capture command")
        layer_path, bundle_path = run_capture(os.environ, args.role, args.device)
        print(
            "MiniMax H3 AudioVAE capture complete: "
            f"role={args.role} layer={layer_path.name} bundle={bundle_path.name}"
        )
        return 0
    except (CaptureFailure, OSError, subprocess.SubprocessError) as error:
        print(f"MiniMax H3 AudioVAE capture rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
