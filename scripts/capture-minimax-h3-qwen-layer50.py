#!/usr/bin/env python3
"""Capture paired exact-BF16 MiniMax H3 Qwen hidden_states[50] evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
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
import tarfile
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_PRODUCER_PATH = REPO_ROOT / "scripts" / "capture-minimax-h3-conditioner.py"
CONFORMANCE_TOOL_PATH = REPO_ROOT / "scripts" / "minimax-h3-conformance.py"
GPU_CONFORMANCE_TOOL_PATH = REPO_ROOT / "scripts" / "run-minimax-h3-gpu-conformance.py"
MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "minimax_h3" / "conformance-manifest.json"
)

DIFFUSERS_REVISION = "9c6a68c32b3b2a64db91800b624d33cec6e25ab8"
TRANSFORMERS_REVISION = "42f189ded85d18d00b51161d694cafd325e32b91"
MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
TEXT_ENCODER_INDEX_SHA256 = (
    "06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a"
)
TEXT_ENCODER_CONFIG_SHA256 = (
    "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427"
)
TEXT_ENCODER_PAYLOAD_BYTES = 66_714_780_128
REVIEWED_AUTHORIZATION_EVIDENCE_SHA256 = (
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d"
)
REQUEST_SCHEMA = "mold.minimax-h3.qwen-layer50-capture-request.v1"
RAW_OUTPUT_SCHEMA = "mold.minimax-h3.qwen-layer50-raw-output.v1"
INPUT_IDENTITY_DOMAIN = b"mold.minimax-h3.qwen-layer50-input.v1\0"
CAPTURE_MARKER = "mold.minimax-h3.private-uat-exact-bf16-qwen-layer50-capture.v1"
LAYER_ID = "qwen-layer-50"
TEXT_CASE_ID = "qwen-layer-50-text-v1"
MULTIMODAL_CASE_ID = "qwen-layer-50-multimodal-v1"
CASE_IDS = (TEXT_CASE_ID, MULTIMODAL_CASE_ID)
COMPONENT_ID = "official-text-encoder-index"
TEXT_ENCODER_SHARD_AUTHORITIES = (
    (
        "official-text-encoder-shard-00001-of-00014",
        "text_encoder/model-00001-of-00014.safetensors",
        "6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb",
    ),
    (
        "official-text-encoder-shard-00002-of-00014",
        "text_encoder/model-00002-of-00014.safetensors",
        "d8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa",
    ),
    (
        "official-text-encoder-shard-00003-of-00014",
        "text_encoder/model-00003-of-00014.safetensors",
        "54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940",
    ),
    (
        "official-text-encoder-shard-00004-of-00014",
        "text_encoder/model-00004-of-00014.safetensors",
        "ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f",
    ),
    (
        "official-text-encoder-shard-00005-of-00014",
        "text_encoder/model-00005-of-00014.safetensors",
        "fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2",
    ),
    (
        "official-text-encoder-shard-00006-of-00014",
        "text_encoder/model-00006-of-00014.safetensors",
        "82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974",
    ),
    (
        "official-text-encoder-shard-00007-of-00014",
        "text_encoder/model-00007-of-00014.safetensors",
        "fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934",
    ),
    (
        "official-text-encoder-shard-00008-of-00014",
        "text_encoder/model-00008-of-00014.safetensors",
        "431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f",
    ),
    (
        "official-text-encoder-shard-00009-of-00014",
        "text_encoder/model-00009-of-00014.safetensors",
        "3825e3f4302f4d2f7d76aa7430d2ce0864fde6b9e540a5806bf0d8e38e4d9f47",
    ),
    (
        "official-text-encoder-shard-00010-of-00014",
        "text_encoder/model-00010-of-00014.safetensors",
        "aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f",
    ),
    (
        "official-text-encoder-shard-00011-of-00014",
        "text_encoder/model-00011-of-00014.safetensors",
        "3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f",
    ),
    (
        "official-text-encoder-shard-00012-of-00014",
        "text_encoder/model-00012-of-00014.safetensors",
        "05ad2d08ce71963121c9b03f1d9ec5d7641052f4b23c6c12b80d71065eb8e98e",
    ),
    (
        "official-text-encoder-shard-00013-of-00014",
        "text_encoder/model-00013-of-00014.safetensors",
        "b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c",
    ),
    (
        "official-text-encoder-shard-00014-of-00014",
        "text_encoder/model-00014-of-00014.safetensors",
        "e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b",
    ),
)
TEXT_ENCODER_COMPONENT_IDS = (COMPONENT_ID,) + tuple(
    identifier for identifier, _, _ in TEXT_ENCODER_SHARD_AUTHORITIES
)
CANONICAL_DTYPE = "bfloat16"
TENSOR_HASH_ENCODING = "canonical-typed-le-v1"
MOLD_BINARY = "h3_qwen_layer50_capture"
MOLD_FEATURES = "dev-bins,h3-private-uat,cuda"
MAXIMUM_RAW_OUTPUT_BYTES = 128 * 1024 * 1024
MAXIMUM_REQUEST_BYTES = 32 * 1024 * 1024
SAMPLED_ACTIVATION_VALUES = 257
REQUIRED_ENVIRONMENT = (
    "MOLD_H3_FIXTURE_ROOT",
    "MOLD_H3_AUTHORIZATION_RECORD",
    "MOLD_H3_OFFICIAL_MODEL",
    "MOLD_H3_DIFFUSERS_CHECKOUT",
    "MOLD_H3_TRANSFORMERS_CHECKOUT",
)


class CaptureFailure(Exception):
    """A fail-closed exact-BF16 layer capture violation."""


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


BASE = load_module("minimax_h3_conditioner_capture_base", BASE_PRODUCER_PATH)


@dataclass(frozen=True)
class SnapshotFile:
    relative_path: str
    kind: str
    size: int
    sha256: str
    device: int
    inode: int
    user: int
    mode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class StagedSnapshot:
    root: pathlib.Path
    files: tuple[SnapshotFile, ...]

    def revalidate(self) -> None:
        if snapshot_files(self.root) != self.files:
            fail("private Qwen capture staging changed during execution")


@dataclass(frozen=True)
class ExternalFileIdentity:
    path: pathlib.Path
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_metadata(
        cls, path: pathlib.Path, metadata: os.stat_result
    ) -> "ExternalFileIdentity":
        if not stat.S_ISREG(metadata.st_mode):
            fail("protected Qwen capture input is not a regular file")
        return cls(
            path=path,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
        )

    @classmethod
    def capture(cls, path: pathlib.Path) -> "ExternalFileIdentity":
        try:
            metadata = path.lstat()
        except OSError:
            fail("protected Qwen capture input became unavailable")
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            fail("protected Qwen capture input is not a direct regular file")
        return cls.from_metadata(path, metadata)

    def revalidate(self) -> None:
        if ExternalFileIdentity.capture(self.path) != self:
            fail("protected Qwen capture input changed during execution")


@dataclass(frozen=True)
class SealedFileSnapshot:
    identity: ExternalFileIdentity
    sha256: str

    @classmethod
    def seal_and_read(
        cls, path: pathlib.Path, maximum_bytes: int
    ) -> tuple["SealedFileSnapshot", bytes]:
        try:
            os.chmod(path, 0o400, follow_symlinks=False)
        except OSError as error:
            fail(f"cannot seal staged Qwen capture input: {error}")
        metadata = path.lstat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o400:
            fail("staged Qwen capture input is not immutable and owner-only")
        identity, digest, encoded = authenticate_regular_file(
            path,
            "staged Qwen capture input",
            maximum_bytes=maximum_bytes,
            retain_bytes=True,
        )
        return cls(identity, digest), encoded

    @classmethod
    def seal(cls, path: pathlib.Path) -> "SealedFileSnapshot":
        snapshot, _ = cls.seal_and_read(path, MAXIMUM_REQUEST_BYTES)
        return snapshot

    def revalidate(self) -> None:
        identity, digest, _ = authenticate_regular_file(
            self.identity.path,
            "staged Qwen capture input",
            maximum_bytes=MAXIMUM_RAW_OUTPUT_BYTES,
            retain_bytes=False,
        )
        metadata = self.identity.path.lstat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o400:
            fail("staged Qwen capture input permissions changed during execution")
        if identity != self.identity or digest != self.sha256:
            fail("staged Qwen capture input bytes changed during execution")


def authenticate_regular_file(
    path: pathlib.Path,
    label: str,
    *,
    maximum_bytes: int | None = None,
    retain_bytes: bool = False,
) -> tuple[ExternalFileIdentity, str, bytes | None]:
    before = ExternalFileIdentity.capture(path)
    if maximum_bytes is not None and before.size > maximum_bytes:
        fail(f"{label} exceeds its bounded size")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError:
        fail(f"{label} is unavailable or unsafe")
    try:
        opened = ExternalFileIdentity.from_metadata(path, os.fstat(descriptor))
        if opened != before:
            fail(f"{label} changed while it was opened")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            observed += len(chunk)
            if retain_bytes:
                chunks.append(chunk)
        after_descriptor = ExternalFileIdentity.from_metadata(path, os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = ExternalFileIdentity.capture(path)
    if (
        opened != after_descriptor
        or opened != after_path
        or observed != opened.size
    ):
        fail(f"{label} changed while it was authenticated")
    encoded = b"".join(chunks) if retain_bytes else None
    return after_path, digest.hexdigest(), encoded


def private_directory(path: pathlib.Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700, follow_symlinks=False)
    except OSError as error:
        fail(f"cannot secure private Qwen capture staging: {error}")


def write_private_file(path: pathlib.Path, encoded: bytes) -> None:
    private_directory(path.parent)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        fail(f"cannot create private Qwen capture staging file: {error}")
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    except OSError as error:
        fail(f"cannot write private Qwen capture staging file: {error}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def safe_relative_path(value: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        fail("private Qwen capture archive contains an unsafe path")
    return path


def read_regular_file_once(root: pathlib.Path, relative_value: str) -> bytes:
    relative = safe_relative_path(relative_value)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    try:
        directory = os.open(root, directory_flags)
    except OSError:
        fail("required Qwen capture input root is unavailable")
    try:
        for part in relative.parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=directory)
            except OSError:
                fail("required Qwen capture input path is unavailable or unsafe")
            os.close(directory)
            directory = child
        try:
            descriptor = os.open(
                relative.parts[-1], os.O_RDONLY | nofollow, dir_fd=directory
            )
        except OSError:
            fail("required Qwen capture input is unavailable or not a regular file")
    finally:
        os.close(directory)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail("required Qwen capture input is not a regular file")
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            observed += len(chunk)
        if observed != metadata.st_size:
            fail("required Qwen capture input changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def extract_archive(encoded: bytes, destination: pathlib.Path) -> None:
    private_directory(destination)
    try:
        archive = tarfile.open(fileobj=io.BytesIO(encoded), mode="r:")
    except tarfile.TarError:
        fail("private Qwen capture source snapshot is not a valid Git archive")
    with archive:
        members = archive.getmembers()
        if not members:
            fail("private Qwen capture source snapshot is empty")
        # Mold tracks `AGENTS.md -> CLAUDE.md`, so a whole-repository archive
        # always carries a symbolic link. Materialize it as a regular copy of an
        # in-snapshot target instead of rejecting it, which would make the
        # `--role mold` staging path unreachable. The staged tree therefore
        # holds regular files only, and `snapshot_files` keeps rejecting links.
        links: list[tuple[pathlib.PurePosixPath, str]] = []
        for member in members:
            relative = safe_relative_path(member.name)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                private_directory(target)
                continue
            if member.issym():
                links.append((relative, member.linkname))
                continue
            if not member.isfile():
                fail("private Qwen capture source snapshot contains a non-regular file")
            source = archive.extractfile(member)
            if source is None:
                fail("private Qwen capture source snapshot cannot be read")
            contents = source.read()
            if len(contents) != member.size:
                fail("private Qwen capture source snapshot changed while it was read")
            write_private_file(target, contents)
        for relative, linkname in links:
            materialize_snapshot_link(destination, relative, linkname)


def materialize_snapshot_link(
    destination: pathlib.Path,
    relative: pathlib.PurePosixPath,
    linkname: str,
) -> None:
    if not linkname or pathlib.PurePosixPath(linkname).is_absolute():
        fail("private Qwen capture source snapshot contains an unsafe link")
    resolved = pathlib.PurePosixPath(
        os.path.normpath(str(relative.parent / linkname))
    )
    if resolved.is_absolute() or not resolved.parts or ".." in resolved.parts:
        fail("private Qwen capture source snapshot link escapes the snapshot")
    source = destination.joinpath(*resolved.parts)
    if source.is_symlink() or not source.is_file():
        fail("private Qwen capture source snapshot link has no in-snapshot target")
    try:
        contents = source.read_bytes()
    except OSError as error:
        fail(f"cannot read private Qwen capture snapshot link target: {error}")
    write_private_file(destination.joinpath(*relative.parts), contents)


def safetensors_payload_length(path: pathlib.Path, file_size: int) -> int:
    """Tensor payload bytes of one safetensors shard.

    A Hugging Face shard index reports the summed tensor payload, while the
    shard on disk also carries an eight-byte little-endian header length and the
    JSON header itself. Comparing whole-file sizes to the index total therefore
    never balances; derive the payload from the shard's own header instead.
    """
    if file_size < 8:
        fail("official text encoder shard is too short to be safetensors")
    try:
        with open(path, "rb") as shard:
            prefix = shard.read(8)
    except OSError as error:
        fail(f"cannot read official text encoder shard header: {error}")
    if len(prefix) != 8:
        fail("official text encoder shard header is truncated")
    header_length = int.from_bytes(prefix, "little")
    if header_length == 0 or header_length > file_size - 8:
        fail("official text encoder shard header length is out of range")
    return file_size - 8 - header_length


def git_archive(
    checkout: pathlib.Path, revision: str, path: str | None = None
) -> bytes:
    arguments = ["git", "archive", "--format=tar", revision]
    if path is not None:
        arguments.append(path)
    try:
        result = subprocess.run(
            arguments, cwd=checkout, check=False, capture_output=True
        )
    except OSError:
        fail("private Qwen capture source snapshot could not be created")
    if result.returncode != 0:
        fail("private Qwen capture source snapshot could not be created")
    return result.stdout


def extract_package_snapshot(
    label: str,
    checkout: pathlib.Path,
    revision: str,
    repository: str,
    package: str,
    destination: pathlib.Path,
) -> None:
    if hasattr(BASE, "extract_git_source_snapshot"):
        BASE.extract_git_source_snapshot(
            label, checkout, revision, repository, package, destination
        )
        return
    BASE.verify_git_checkout(label, checkout, revision, repository)
    extract_archive(git_archive(checkout, revision, f"src/{package}"), destination)


def extract_mold_snapshot(
    checkout: pathlib.Path, revision: str, destination: pathlib.Path
) -> None:
    BASE.verify_git_checkout(
        "Mold", checkout, revision, "https://github.com/utensils/mold"
    )
    extract_archive(git_archive(checkout, revision), destination)


def snapshot_files(root: pathlib.Path) -> tuple[SnapshotFile, ...]:
    files: list[SnapshotFile] = []
    try:
        root_metadata = root.lstat()
    except OSError as error:
        fail(f"cannot inspect private Qwen capture staging root: {error}")
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
    ):
        fail("private Qwen capture staging root is not owner-only")
    try:
        entries = sorted(root.rglob("*"))
    except OSError as error:
        fail(f"cannot enumerate private Qwen capture staging: {error}")
    for path in entries:
        try:
            metadata = path.lstat()
        except OSError as error:
            fail(f"cannot inspect private Qwen capture staging: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            fail("private Qwen capture staging contains a symbolic link")
        if stat.S_ISDIR(metadata.st_mode):
            if (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                fail("private Qwen capture staging directory is not owner-only")
            files.append(
                SnapshotFile(
                    relative_path=path.relative_to(root).as_posix(),
                    kind="directory",
                    size=0,
                    sha256="",
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                    user=metadata.st_uid,
                    mode=stat.S_IMODE(metadata.st_mode),
                    modified_ns=metadata.st_mtime_ns,
                    changed_ns=metadata.st_ctime_ns,
                )
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            fail("private Qwen capture staging contains a non-regular file")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            fail("private Qwen capture staging file is not owner-only")
        files.append(
            SnapshotFile(
                relative_path=path.relative_to(root).as_posix(),
                kind="file",
                size=metadata.st_size,
                sha256=BASE.sha256_file(path),
                device=metadata.st_dev,
                inode=metadata.st_ino,
                user=metadata.st_uid,
                mode=stat.S_IMODE(metadata.st_mode),
                modified_ns=metadata.st_mtime_ns,
                changed_ns=metadata.st_ctime_ns,
            )
        )
    if not any(item.kind == "file" for item in files):
        fail("private Qwen capture staging is empty")
    return tuple(files)


def seal_staging(root: pathlib.Path) -> StagedSnapshot:
    files = snapshot_files(root)
    for item in files:
        if item.kind != "file":
            continue
        try:
            os.chmod(root / item.relative_path, 0o400, follow_symlinks=False)
        except OSError as error:
            fail(f"cannot make private Qwen capture staging immutable: {error}")
    sealed = snapshot_files(root)
    if any(item.kind == "file" and item.mode != 0o400 for item in sealed):
        fail("private Qwen capture staging file did not become immutable")
    return StagedSnapshot(root, sealed)


def stage_model_components(
    model_root: pathlib.Path,
    destination: pathlib.Path,
    manifest: Mapping[str, Any],
) -> None:
    if hasattr(BASE, "stage_model_components"):
        BASE.stage_model_components(model_root, destination, manifest)
        return
    components = component_map(manifest)
    private_directory(destination)
    for identifier in BASE.REQUIRED_COMPONENT_IDS:
        component = components[identifier]
        relative = safe_relative_path(component["relative_path"])
        encoded = read_regular_file_once(model_root, component["relative_path"])
        if sha256_bytes(encoded) != component["sha256"]:
            fail(f"official model component hash drifted: {identifier}")
        if (
            BASE.huggingface_metadata_revision(model_root, component["relative_path"])
            != MODEL_REVISION
        ):
            fail(f"official model component revision drifted: {identifier}")
        write_private_file(destination.joinpath(*relative.parts), encoded)


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
        fail("Qwen capture ambient runtime identity is incomplete")
    return values


def canonical_json_bytes(value: Any) -> bytes:
    return BASE.canonical_json_bytes(value)


def document_bytes(value: Any) -> bytes:
    return BASE.document_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def reviewed_text_encoder_shards(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    components = component_map(manifest)
    reviewed: list[dict[str, str]] = []
    for identifier, relative_path, sha256 in TEXT_ENCODER_SHARD_AUTHORITIES:
        component = components.get(identifier)
        if component != {
            "id": identifier,
            "relative_path": relative_path,
            "sha256": sha256,
        }:
            fail("checked manifest lacks an exact official text encoder shard pin")
        reviewed.append(component)
    return tuple(reviewed)


def layer_contract(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [layer for layer in manifest["fixture_layers"] if layer["id"] == LAYER_ID]
    if len(matches) != 1:
        fail("checked manifest does not contain one Qwen layer-50 contract")
    layer = matches[0]
    if tuple(layer["required_component_indexes"]) != TEXT_ENCODER_COMPONENT_IDS:
        fail("Qwen layer-50 component authority drifted")
    if tuple(layer["required_measurements"]) != (
        "shape",
        "dtype",
        "statistics",
        "sampled-values",
        "tolerance",
    ):
        fail("Qwen layer-50 measurement contract drifted")
    legacy_provenance = (
        "source-revision",
        "component-index-hashes",
        "device",
        "dtype",
        "capture-command",
    )
    exact_adapter_provenance = (
        "source-revision",
        "transformers-revision",
        "adapter-implementation-sha256",
        "oracle-runtime-sha256",
        "component-index-hashes",
        "device",
        "dtype",
        "capture-command",
    )
    if tuple(layer["required_provenance"]) not in {
        legacy_provenance,
        exact_adapter_provenance,
    }:
        fail("Qwen layer-50 provenance contract drifted")
    adapter = layer.get("oracle_adapter")
    if adapter is not None:
        expected = {
            "id": "minimax-h3-qwen-layer50-oracle-exact-bf16-v1",
            "command_prefix": (
                "python3 scripts/capture-minimax-h3-qwen-layer50.py capture --role oracle"
            ),
            "implementation_path": "scripts/capture-minimax-h3-qwen-layer50.py",
            "implementation_sha256": BASE.sha256_file(pathlib.Path(__file__).resolve()),
        }
        if adapter != expected:
            fail("Qwen layer-50 oracle adapter authority drifted")
    return layer


def adapter_implementation_sha256(role: str) -> str:
    script_sha = BASE.sha256_file(pathlib.Path(__file__).resolve())
    if role == "oracle":
        return script_sha
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "mold.minimax-h3.qwen-layer50-mold-adapter.v1",
                "producer_sha256": script_sha,
                "rust_adapter_sha256": BASE.sha256_file(
                    REPO_ROOT
                    / "crates"
                    / "mold-inference"
                    / "src"
                    / "bin"
                    / "h3_qwen_layer50_capture.rs"
                ),
                "cargo_manifest_sha256": BASE.sha256_file(
                    REPO_ROOT / "crates" / "mold-inference" / "Cargo.toml"
                ),
            }
        )
    )


def model_metadata_lines(
    model_root: pathlib.Path, relative_path: str
) -> tuple[list[str], ExternalFileIdentity]:
    metadata_relative = f".cache/huggingface/download/{relative_path}.metadata"
    metadata_path = model_root / metadata_relative
    if metadata_path.is_symlink():
        fail("official model component metadata may not be a symbolic link")
    try:
        metadata_path = metadata_path.resolve(strict=True)
    except OSError:
        fail("official model component lacks revision-bound metadata")
    if not metadata_path.is_file() or not BASE.is_relative_to(
        metadata_path, model_root
    ):
        fail("official model component metadata escapes its snapshot")
    identity, _, encoded = authenticate_regular_file(
        metadata_path,
        "official model component metadata",
        maximum_bytes=4096,
        retain_bytes=True,
    )
    if encoded is None:
        fail("official model component metadata was not retained")
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeError:
        fail("official model component metadata is not UTF-8")
    return lines, identity


def authenticate_text_encoder_shard(
    model_root: pathlib.Path, component: Mapping[str, str]
) -> tuple[ExternalFileIdentity, ExternalFileIdentity]:
    relative = component["relative_path"]
    relative_path = pathlib.PurePosixPath(relative)
    if (
        relative_path.parent != pathlib.PurePosixPath("text_encoder")
        or relative_path.name != relative.removeprefix("text_encoder/")
        or not relative_path.name.endswith(".safetensors")
    ):
        fail("reviewed text encoder shard authority has a non-canonical path")
    candidate = model_root / relative
    if candidate.is_symlink():
        fail("official text encoder shard may not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        fail("official text encoder shard is unavailable")
    if not resolved.is_file() or not BASE.is_relative_to(resolved, model_root):
        fail("official text encoder shard escapes its snapshot")
    lines, metadata_identity = model_metadata_lines(model_root, relative)
    if (
        len(lines) < 2
        or lines[0] != MODEL_REVISION
        or lines[1] != component["sha256"]
    ):
        fail("official text encoder shard metadata differs from the reviewed pin")
    identity, sha256, _ = authenticate_regular_file(
        resolved, "official text encoder shard"
    )
    if sha256 != component["sha256"]:
        fail("official text encoder shard bytes differ from the reviewed pin")
    return identity, metadata_identity


def verify_text_encoder_snapshot(
    model_root: pathlib.Path, manifest: Mapping[str, Any]
) -> tuple[ExternalFileIdentity, ...]:
    relative_index = "text_encoder/model.safetensors.index.json"
    index_path = model_root / relative_index
    if index_path.is_symlink():
        fail("official text encoder index may not be a symbolic link")
    try:
        index_path = index_path.resolve(strict=True)
    except OSError:
        fail("official text encoder index is unavailable")
    if not index_path.is_file() or not BASE.is_relative_to(index_path, model_root):
        fail("official text encoder index escapes its snapshot")
    index_after, index_sha256, index_bytes = authenticate_regular_file(
        index_path,
        "official text encoder index",
        maximum_bytes=16 * 1024 * 1024,
        retain_bytes=True,
    )
    if index_sha256 != TEXT_ENCODER_INDEX_SHA256:
        fail("official text encoder index differs from the reviewed manifest")
    if index_bytes is None:
        fail("official text encoder index was not retained")
    index_metadata, index_metadata_identity = model_metadata_lines(
        model_root, relative_index
    )
    if not index_metadata or index_metadata[0] != MODEL_REVISION:
        fail("official text encoder index revision drifted")
    try:
        index = json.loads(index_bytes.decode("utf-8"))
        weight_map = index["weight_map"]
        total_size = int(index["metadata"]["total_size"])
    except (
        UnicodeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        fail("official text encoder index is malformed")
    reviewed_shards = reviewed_text_encoder_shards(manifest)
    shard_names = sorted(set(weight_map.values()))
    expected_names = [
        pathlib.PurePosixPath(component["relative_path"]).name
        for component in reviewed_shards
    ]
    if shard_names != expected_names or total_size != TEXT_ENCODER_PAYLOAD_BYTES:
        fail("official text encoder index payload geometry drifted")
    observed_payload = 0
    identities = [index_after, index_metadata_identity]
    for component in reviewed_shards:
        identity, metadata_identity = authenticate_text_encoder_shard(
            model_root, component
        )
        observed_payload += safetensors_payload_length(
            model_root / component["relative_path"], identity.size
        )
        identities.extend((identity, metadata_identity))
    if observed_payload != TEXT_ENCODER_PAYLOAD_BYTES:
        fail("official text encoder shard payloads differ from the reviewed index")
    config_relative = "text_encoder/config.json"
    config_path = (model_root / config_relative).resolve(strict=True)
    config_after, config_sha256, _ = authenticate_regular_file(
        config_path,
        "official text encoder config",
        maximum_bytes=16 * 1024 * 1024,
    )
    if config_sha256 != TEXT_ENCODER_CONFIG_SHA256:
        fail("official text encoder config differs from the reviewed authority")
    config_metadata, config_metadata_identity = model_metadata_lines(
        model_root, config_relative
    )
    if not config_metadata or config_metadata[0] != MODEL_REVISION:
        fail("official text encoder config metadata differs from its authority")
    identities.extend((config_after, config_metadata_identity))
    return tuple(identities)


def load_runtime(
    diffusers_checkout: pathlib.Path, transformers_checkout: pathlib.Path
) -> SimpleNamespace:
    runtime = BASE.load_capture_runtime(diffusers_checkout, transformers_checkout)
    try:
        transformers = importlib.import_module("transformers")
        encoder_module = importlib.import_module(
            "diffusers.modular_pipelines.minimax_h3.encoders"
        )
    except (ImportError, RuntimeError) as error:
        fail(f"reviewed Qwen layer runtime cannot be imported: {error}")
    BASE.verify_module_origin(
        "Qwen3-VL exact model",
        transformers.Qwen3VLForConditionalGeneration,
        transformers_checkout,
    )
    BASE.verify_module_origin(
        "Diffusers Qwen layer adapter",
        encoder_module.get_qwen3vl_prompt_embeds,
        diffusers_checkout,
    )
    runtime.model_class = transformers.Qwen3VLForConditionalGeneration
    runtime.encoder_function = encoder_module.get_qwen3vl_prompt_embeds
    return runtime


def processor_instances(
    runtime: SimpleNamespace, model_root: pathlib.Path
) -> tuple[Any, Any]:
    try:
        tokenizer = runtime.tokenizer_class.from_pretrained(
            model_root / "tokenizer", local_files_only=True
        )
        processor = runtime.processor_class.from_pretrained(
            model_root / "processor", local_files_only=True
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        fail(f"pinned tokenizer/processor could not be loaded: {error}")
    BASE.verify_module_origin(
        "Qwen tokenizer", tokenizer.__class__, runtime.transformers_checkout
    )
    BASE.verify_module_origin(
        "Qwen processor tokenizer",
        processor.tokenizer.__class__,
        runtime.transformers_checkout,
    )
    BASE.verify_module_origin(
        "Qwen image processor",
        processor.image_processor.__class__,
        runtime.transformers_checkout,
    )
    BASE.verify_module_origin(
        "Qwen video processor",
        processor.video_processor.__class__,
        runtime.transformers_checkout,
    )
    return tokenizer, processor


def tensor_bfloat16_payload(torch: Any, tensor: Any) -> tuple[str, list[int]]:
    copied = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    values = [
        int(value) & 0xFFFF for value in copied.view(torch.int16).reshape(-1).tolist()
    ]
    encoded = b"".join(struct.pack("<H", value) for value in values)
    return base64.b64encode(encoded).decode("ascii"), values


def put_length(destination: bytearray, value: int) -> None:
    if value < 0 or value > 2**64 - 1:
        fail("capture input length is outside u64")
    destination.extend(struct.pack("<Q", value))


def put_string(destination: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    put_length(destination, len(encoded))
    destination.extend(encoded)


def case_identity(case: Mapping[str, Any]) -> str:
    encoded = bytearray(INPUT_IDENTITY_DOMAIN)
    put_string(encoded, case["case_id"])
    put_string(encoded, case["kind"])
    token_ids = case["token_ids"]
    put_length(encoded, len(token_ids))
    for value in token_ids:
        encoded.extend(struct.pack("<I", int(value)))
    for axis in case["position_ids"]:
        put_length(encoded, len(axis))
        for value in axis:
            encoded.extend(struct.pack("<I", int(value)))
    for name in ("image", "video"):
        payload = case[name]
        if payload is None:
            encoded.append(0)
            continue
        encoded.append(1)
        for value in payload["shape"]:
            put_length(encoded, int(value))
        for value in payload["grid_thw"]:
            put_length(encoded, int(value))
        try:
            raw = base64.b64decode(payload["bfloat16_le_base64"], validate=True)
        except (ValueError, TypeError) as error:
            fail(f"capture input vision payload is not canonical base64: {error}")
        put_length(encoded, len(raw))
        encoded.extend(raw)
    return sha256_bytes(bytes(encoded))


def qwen_positions(
    mm_types: Sequence[int],
    image_grids: Sequence[Sequence[int]],
    video_grids: Sequence[Sequence[int]],
) -> list[list[int]]:
    merge = 2
    expanded_video: list[list[int]] = []
    for temporal, height, width in video_grids:
        expanded_video.extend([[1, int(height), int(width)]] * int(temporal))
    grid_iters = {1: iter(image_grids), 2: iter(expanded_video)}
    positions = [[], [], []]
    current = 0
    start = 0
    while start < len(mm_types):
        kind = int(mm_types[start])
        end = start + 1
        while end < len(mm_types) and int(mm_types[end]) == kind:
            end += 1
        length = end - start
        if kind == 0:
            for offset in range(length):
                for axis in positions:
                    axis.append(current + offset)
            current += length
        elif kind in (1, 2):
            try:
                temporal, height, width = next(grid_iters[kind])
            except StopIteration:
                fail("Qwen modality runs exceed their processor grids")
            temporal, height, width = int(temporal), int(height), int(width)
            if height % merge or width % merge:
                fail("Qwen processor grid is not divisible by spatial merge")
            merged_height, merged_width = height // merge, width // merge
            expected = temporal * merged_height * merged_width
            if expected != length:
                fail("Qwen modality run differs from its processor grid")
            for time in range(temporal):
                for row in range(merged_height):
                    for column in range(merged_width):
                        positions[0].append(current + time)
                        positions[1].append(current + row)
                        positions[2].append(current + column)
            current += max(temporal, merged_height, merged_width)
        else:
            fail("Qwen processor emitted an unsupported modality type")
        start = end
    for iterator in grid_iters.values():
        try:
            next(iterator)
        except StopIteration:
            continue
        fail("Qwen processor returned more grids than the presentation uses")
    if any(len(axis) != len(mm_types) for axis in positions):
        fail("Qwen position construction changed sequence length")
    return positions


def vision_payload(torch: Any, pixels: Any, grid: Any) -> dict[str, Any]:
    encoded, _ = tensor_bfloat16_payload(torch, pixels)
    shape = [int(value) for value in pixels.shape]
    grids = grid.detach().to(device="cpu").tolist()
    if len(grids) != 1 or len(grids[0]) != 3:
        fail("capture case requires exactly one vision grid per modality")
    return {
        "shape": shape,
        "grid_thw": [int(value) for value in grids[0]],
        "bfloat16_le_base64": encoded,
    }


def build_cases(
    runtime: SimpleNamespace, staged_model_root: pathlib.Path
) -> list[dict[str, Any]]:
    torch = runtime.torch
    numpy = runtime.numpy
    tokenizer, processor = processor_instances(runtime, staged_model_root)

    text_prompt = "Qwen layer 50 text-only parity: coral robot, café, exact BF16."
    text_ids = [
        int(value)
        for value in tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
    ]
    text_types = [
        int(value) for value in processor.create_mm_token_type_ids([text_ids])[0]
    ]
    text_case: dict[str, Any] = {
        "case_id": TEXT_CASE_ID,
        "kind": "text-only",
        "token_ids": text_ids,
        "position_ids": qwen_positions(text_types, [], []),
        "image": None,
        "video": None,
    }
    text_case["input_sha256"] = case_identity(text_case)

    prompt = "Representative image and video presentation: coral robot, café."
    image = BASE.rgb_pattern(numpy, 1, 256, 256)[0]
    video = BASE.rgb_pattern(numpy, 37, 64, 64)
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
    if tuple(round(float(value) * 1000) for value in timestamps) != (250, 1250):
        fail("pinned Diffusers timestamp authority drifted")
    try:
        image_features = processor.image_processor(images=[image], return_tensors="pt")
        video_features = processor.video_processor(
            videos=[numpy.stack(sampled_frames)],
            do_sample_frames=False,
            return_tensors="pt",
        )
    except (RuntimeError, TypeError, ValueError) as error:
        fail(f"pinned Qwen processor execution failed: {error}")
    image_grid = image_features["image_grid_thw"]
    video_grid = video_features["video_grid_thw"]
    image_tokens = int(image_grid[0].prod().item()) // 4
    video_tokens = int(video_grid[0][1].item()) * int(video_grid[0][2].item()) // 4
    if int(video_grid[0][0].item()) != len(timestamps):
        fail("pinned Qwen video processor changed temporal geometry")
    references = [
        SimpleNamespace(kind="image", has_audio=False),
        SimpleNamespace(kind="video", has_audio=False),
    ]
    token_ids, _ = runtime.encoder_step._build_presentation(
        tokenizer,
        prompt,
        references,
        [image_tokens],
        [video_tokens],
        [list(timestamps)],
    )
    token_ids = [int(value) for value in token_ids]
    mm_types = [
        int(value) for value in processor.create_mm_token_type_ids([token_ids])[0]
    ]
    image_payload = vision_payload(torch, image_features["pixel_values"], image_grid)
    video_payload = vision_payload(
        torch, video_features["pixel_values_videos"], video_grid
    )
    multimodal_case: dict[str, Any] = {
        "case_id": MULTIMODAL_CASE_ID,
        "kind": "multimodal",
        "token_ids": token_ids,
        "position_ids": qwen_positions(
            mm_types, [image_payload["grid_thw"]], [video_payload["grid_thw"]]
        ),
        "image": image_payload,
        "video": video_payload,
    }
    multimodal_case["input_sha256"] = case_identity(multimodal_case)
    return [text_case, multimodal_case]


def case_set_identity(cases: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "mold.minimax-h3.qwen-layer50-case-set.v1",
                "cases": [
                    {"case_id": case["case_id"], "input_sha256": case["input_sha256"]}
                    for case in cases
                ],
            }
        )
    )


def capture_directory(
    fixture_root: pathlib.Path, cases: Sequence[Mapping[str, Any]]
) -> pathlib.Path:
    parent = fixture_root / "captures" / LAYER_ID
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    if not BASE.is_relative_to(parent, fixture_root):
        fail("Qwen capture destination escapes MOLD_H3_FIXTURE_ROOT")
    directory = parent / case_set_identity(cases)[:24]
    try:
        directory.mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        fail(f"cannot create external Qwen capture directory: {error}")
    directory = directory.resolve(strict=True)
    if not BASE.is_relative_to(directory, fixture_root):
        fail("Qwen capture directory escapes MOLD_H3_FIXTURE_ROOT")
    metadata = directory.stat()
    if stat.S_IMODE(metadata.st_mode) & 0o077 or metadata.st_uid != os.geteuid():
        fail("Qwen capture directory must be owner-only")
    return directory


def payload_tensor(runtime: SimpleNamespace, payload: Mapping[str, Any]) -> Any:
    try:
        raw = base64.b64decode(payload["bfloat16_le_base64"], validate=True)
    except (TypeError, ValueError) as error:
        fail(f"vision payload is not canonical base64: {error}")
    if len(raw) != math.prod(payload["shape"]) * 2:
        fail("vision payload byte count differs from its shape")
    numpy = runtime.numpy
    bits = numpy.frombuffer(raw, dtype="<u2").astype("<u4")
    bits <<= 16
    values = bits.view("<f4").copy().reshape(payload["shape"])
    return runtime.torch.from_numpy(values).to(dtype=runtime.torch.bfloat16)


def torch_bits(torch: Any, tensor: Any, device: Any) -> tuple[list[int], list[int]]:
    copied = tensor.detach().to(device=device, dtype=torch.bfloat16).contiguous()
    torch.cuda.synchronize(device)
    shape = [int(value) for value in copied.shape]
    signed = copied.to(device="cpu").view(torch.int16).reshape(-1).tolist()
    return shape, [int(value) & 0xFFFF for value in signed]


def capture_oracle(
    runtime: SimpleNamespace,
    weight_model_root: pathlib.Path,
    staged_model_root: pathlib.Path,
    cases: Sequence[Mapping[str, Any]],
    device_label: str,
) -> dict[str, tuple[list[int], list[int]]]:
    torch = runtime.torch
    device = BASE.configure_torch(runtime, device_label)
    try:
        model = runtime.model_class.from_pretrained(
            weight_model_root / "text_encoder",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        fail(f"pinned exact-BF16 Qwen model could not be loaded: {error}")
    BASE.verify_module_origin(
        "loaded Qwen3-VL exact model", model.__class__, runtime.transformers_checkout
    )
    if getattr(model, "is_quantized", False) or model.dtype != torch.bfloat16:
        fail("Diffusers oracle loaded a quantized or non-BF16 conditioner")
    if int(model.config.text_config.num_hidden_layers) != 64:
        fail("Diffusers oracle did not load the full official language stack")

    _, processor = processor_instances(runtime, staged_model_root)
    results: dict[str, tuple[list[int], list[int]]] = {}
    with torch.no_grad():
        for case in cases:
            input_ids = torch.tensor(
                [case["token_ids"]], dtype=torch.long, device=device
            )
            mm_types = torch.tensor(
                processor.create_mm_token_type_ids([case["token_ids"]]),
                dtype=torch.long,
                device=device,
            )
            vision_inputs: dict[str, Any] = {}
            rope_arguments: dict[str, Any] = {}
            if case["image"] is not None:
                pixels = payload_tensor(runtime, case["image"])
                grid = torch.tensor([case["image"]["grid_thw"]], dtype=torch.long)
                vision_inputs.update(pixel_values=pixels, image_grid_thw=grid)
                rope_arguments["image_grid_thw"] = grid.to(device)
            if case["video"] is not None:
                pixels = payload_tensor(runtime, case["video"])
                grid = torch.tensor([case["video"]["grid_thw"]], dtype=torch.long)
                vision_inputs.update(pixel_values_videos=pixels, video_grid_thw=grid)
                rope_arguments["video_grid_thw"] = grid.to(device)
            observed_positions, _ = model.model.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_types,
                attention_mask=torch.ones_like(input_ids),
                **rope_arguments,
            )
            expected_positions = torch.tensor(
                case["position_ids"], dtype=torch.long, device=device
            ).unsqueeze(1)
            if not torch.equal(observed_positions, expected_positions):
                fail(
                    "shared capture positions differ from the pinned Transformers authority"
                )
            activation = runtime.encoder_function(
                model,
                processor,
                case["token_ids"],
                vision_inputs,
                text_encoder_layer=50,
                device=device,
                dtype=torch.bfloat16,
            )
            shape, bits = torch_bits(torch, activation, device)
            if shape != [1, len(case["token_ids"]), 5120]:
                fail("Diffusers oracle emitted the wrong unnormalized layer-50 shape")
            results[case["case_id"]] = (shape, bits)
    del model
    torch.cuda.empty_cache()
    return results


def write_request(
    path: pathlib.Path,
    authorization_sha256: str,
    source_revision: str,
    cases: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> SealedFileSnapshot:
    request = {
        "schema_version": REQUEST_SCHEMA,
        "authorization_document_sha256": authorization_sha256,
        "source_revision": source_revision,
        "model_revision": MODEL_REVISION,
        "checkpoint_components": list(reviewed_text_encoder_shards(manifest)),
        "cases": list(cases),
    }
    encoded = document_bytes(request)
    if len(encoded) > MAXIMUM_REQUEST_BYTES:
        fail("Mold staged capture request exceeds its emitted-size bound")
    BASE.exclusive_write(path, encoded)
    return SealedFileSnapshot.seal(path)


def read_raw_output_bytes(
    encoded: bytes,
    authorization_sha256: str,
    source_revision: str,
    device: str,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[list[int], list[int]]]:
    try:
        if len(encoded) > MAXIMUM_RAW_OUTPUT_BYTES:
            fail("Mold raw layer capture exceeds its bounded output size")
        raw = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        fail("Mold raw layer capture is unavailable or malformed")
    expected_keys = {
        "schema_version",
        "claim_marker",
        "authorization_document_sha256",
        "source_revision",
        "model_revision",
        "device",
        "dtype",
        "resident_language_layers",
        "cases",
    }
    if set(raw) != expected_keys or (
        raw["schema_version"] != RAW_OUTPUT_SCHEMA
        or raw["claim_marker"] != CAPTURE_MARKER
        or raw["authorization_document_sha256"] != authorization_sha256
        or raw["source_revision"] != source_revision
        or raw["model_revision"] != MODEL_REVISION
        or raw["device"] != device
        or raw["dtype"] != CANONICAL_DTYPE
        or raw["resident_language_layers"] != 50
    ):
        fail("Mold raw layer capture is not bound to the requested exact authority")
    expected = {case["case_id"]: case for case in cases}
    results: dict[str, tuple[list[int], list[int]]] = {}
    for item in raw["cases"]:
        if set(item) != {"case_id", "input_sha256", "shape", "bfloat16_le_base64"}:
            fail("Mold raw layer capture case has an unknown field")
        case = expected.get(item["case_id"])
        if case is None or item["input_sha256"] != case["input_sha256"]:
            fail("Mold raw layer capture case differs from its prepared input")
        shape = [int(value) for value in item["shape"]]
        if shape != [1, len(case["token_ids"]), 5120]:
            fail("Mold raw layer capture emitted the wrong hidden-state shape")
        try:
            encoded = base64.b64decode(item["bfloat16_le_base64"], validate=True)
        except (TypeError, ValueError):
            fail("Mold raw layer capture is not canonical BF16 base64")
        if len(encoded) != math.prod(shape) * 2:
            fail("Mold raw layer capture BF16 bytes differ from its shape")
        bits = [
            int.from_bytes(encoded[index : index + 2], "little")
            for index in range(0, len(encoded), 2)
        ]
        if item["case_id"] in results:
            fail("Mold raw layer capture repeats a case")
        results[item["case_id"]] = (shape, bits)
    if set(results) != set(expected):
        fail("Mold raw layer capture lacks one prepared case")
    return results


def capture_mold(
    fixture_root: pathlib.Path,
    capture_dir: pathlib.Path,
    authorization_path: pathlib.Path,
    authorization_sha256: str,
    model_root: pathlib.Path,
    cases: Sequence[Mapping[str, Any]],
    device: str,
    source_revision: str,
    staged_mold_root: pathlib.Path,
    manifest: Mapping[str, Any],
) -> tuple[str, dict[str, tuple[list[int], list[int]]]]:
    if not lower_hex(source_revision, 40):
        fail("Mold checkout does not have a canonical source revision")
    request_path = capture_dir / "mold-request.json"
    raw_output_path = capture_dir / "mold-raw.json"
    if raw_output_path.exists():
        fail("Mold raw layer capture already exists; refusing to overwrite evidence")
    request_snapshot = write_request(
        request_path, authorization_sha256, source_revision, cases, manifest
    )
    target_dir = fixture_root / "build" / LAYER_ID / source_revision
    target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target_dir = target_dir.resolve(strict=True)
    if not BASE.is_relative_to(target_dir, fixture_root):
        fail("Mold capture build directory escapes MOLD_H3_FIXTURE_ROOT")
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
        fail("Mold exact-BF16 capture adapter could not be started")
    if result.returncode != 0:
        raw_output_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        fail("Mold exact-BF16 capture adapter failed; inspect the protected runner log")
    try:
        request_snapshot.revalidate()
        raw_output_snapshot, raw_output_bytes = SealedFileSnapshot.seal_and_read(
            raw_output_path, MAXIMUM_RAW_OUTPUT_BYTES
        )
        captured = read_raw_output_bytes(
            raw_output_bytes,
            authorization_sha256,
            source_revision,
            device,
            cases,
        )
        raw_output_snapshot.revalidate()
        request_snapshot.revalidate()
    except BaseException:
        raw_output_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        raise
    return source_revision, captured


def float64_tensor_record(key: str, values: Sequence[float]) -> dict[str, Any]:
    encoded_values = [float(value) for value in values]
    if not encoded_values or not all(math.isfinite(value) for value in encoded_values):
        fail(f"{key} evidence must contain finite float64 values")
    digest = hashlib.sha256()
    for value in encoded_values:
        digest.update(struct.pack("<d", value))
    return {
        "key": key,
        "shape": [len(encoded_values)],
        "dtype": "float64",
        "content_sha256": digest.hexdigest(),
        "statistics": BASE.finite_statistics(encoded_values),
        "samples": [
            {"index": [index], "value": encoded_values[index]}
            for index in BASE.sample_indexes(len(encoded_values))
        ],
    }


def activation_sample_indexes(length: int) -> list[int]:
    if length <= 0:
        fail("Qwen layer-50 activation is empty")
    count = min(SAMPLED_ACTIVATION_VALUES, length)
    if count == 1:
        return [0]
    indexes = [index * (length - 1) // (count - 1) for index in range(count)]
    if len(indexes) != len(set(indexes)):
        fail("Qwen activation sampling geometry produced duplicate indexes")
    return indexes


def activation_tensor_record(
    shape: Sequence[int], bits: Sequence[int], values: Sequence[float]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for value in bits:
        if not 0 <= int(value) <= 0xFFFF:
            fail("Qwen activation contains an invalid BF16 encoding")
        digest.update(struct.pack("<H", int(value)))
    sequence, hidden = int(shape[1]), int(shape[2])
    samples = []
    for flat_index in activation_sample_indexes(len(bits)):
        batch = flat_index // (sequence * hidden)
        remainder = flat_index % (sequence * hidden)
        token = remainder // hidden
        channel = remainder % hidden
        samples.append(
            {
                "index": [batch, token, channel],
                "value": float(values[flat_index]),
            }
        )
    return {
        "key": "sampled-values",
        "shape": [int(value) for value in shape],
        "dtype": CANONICAL_DTYPE,
        "content_sha256": digest.hexdigest(),
        "statistics": BASE.finite_statistics(values),
        "samples": samples,
    }


def measurement_outputs(
    shape: Sequence[int], bits: Sequence[int]
) -> list[dict[str, Any]]:
    if (
        len(shape) != 3
        or shape[0] != 1
        or shape[2] != 5120
        or math.prod(shape) != len(bits)
    ):
        fail("Qwen layer-50 activation shape differs from its BF16 payload")
    values = [BASE.bfloat16_bits_to_float(int(value)) for value in bits]
    statistics = BASE.finite_statistics(values)
    return [
        BASE.integer_tensor_record("shape", [int(value) for value in shape]),
        BASE.integer_tensor_record("dtype", [1]),
        float64_tensor_record(
            "statistics",
            [
                float(statistics["min"]),
                float(statistics["max"]),
                float(statistics["mean"]),
                float(statistics["std"]),
            ],
        ),
        activation_tensor_record(shape, bits, values),
        float64_tensor_record("tolerance", [0.0, 0.0]),
    ]


def acceleration_policy(manifest: Mapping[str, Any], role: str) -> dict[str, list[str]]:
    disabled = list(manifest["numerical_authority"]["excluded_accelerations"])
    if len(disabled) != len(set(disabled)):
        fail("checked manifest repeats acceleration exclusions")
    enabled = (
        ["cuda", "deterministic-cuda", "pytorch-bfloat16", "math-sdpa"]
        if role == "oracle"
        else [
            "cuda",
            "deterministic-cuda",
            "candle-bfloat16",
            "dense-official-conditioner",
        ]
    )
    return {"enabled": enabled, "disabled": disabled}


def capture_command(role: str, device: str) -> str:
    return (
        "python3 scripts/capture-minimax-h3-qwen-layer50.py capture "
        f"--role {role} --device {device}"
    )


def build_layer_document(
    manifest: Mapping[str, Any],
    role: str,
    source_revision: str,
    authorization_sha256: str,
    device: str,
    case: Mapping[str, Any],
    activation: tuple[list[int], list[int]],
    runtime_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    contract = layer_contract(manifest)
    components = component_map(manifest)
    component = components.get(COMPONENT_ID)
    if component is None or component["sha256"] != TEXT_ENCODER_INDEX_SHA256:
        fail("checked manifest lacks the exact official text encoder authority")
    component_authorities = [component, *reviewed_text_encoder_shards(manifest)]
    component_summary_sha256 = BASE.component_authority_set_sha256(
        [authority["id"] for authority in component_authorities], components
    )
    component_hashes = ",".join(
        f"{authority['id']}={authority['sha256']}"
        for authority in sorted(component_authorities, key=lambda item: item["id"])
    )
    command = capture_command(role, device)
    outputs = measurement_outputs(*activation)
    adapter_sha256 = adapter_implementation_sha256(role)
    reviewed_runtime = manifest["numerical_authority"].get("oracle_runtime")
    extended_authority = isinstance(reviewed_runtime, dict)
    runtime_identity = dict(
        runtime_identity
        or (
            reviewed_runtime if extended_authority else {"unavailable": "contract-test"}
        )
    )
    runtime_sha256 = sha256_bytes(
        canonical_json_bytes(
            reviewed_runtime if extended_authority else runtime_identity
        )
    )
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
            "component_index_sha256": component_summary_sha256,
            "component_indexes": [
                {"id": authority["id"], "sha256": authority["sha256"]}
                for authority in component_authorities
            ],
        },
        "producer": {
            "role": role,
            "implementation": (
                "diffusers-minimax-h3-qwen3-vl-hidden-states-50-exact-bf16"
                if role == "oracle"
                else "mold-candle-dense-official-qwen-layer50-exact-bf16"
            ),
            "source_id": "diffusers" if role == "oracle" else "mold",
            "revision": source_revision,
        },
        "adapter": {
            "schema_version": "mold.minimax-h3.layer-adapter.v1",
            "id": f"minimax-h3-qwen-layer50-{role}-exact-bf16-v1",
            "command": command,
            "tensor_hash_encoding": TENSOR_HASH_ENCODING,
        },
        "environment": {
            "device": device,
            "dtype": CANONICAL_DTYPE,
            "attention_backend": "math",
            "acceleration_policy": acceleration_policy(manifest, role),
            "forbidden_accelerations_disabled": True,
        },
        "provenance": [
            {"key": "source-revision", "value": source_revision},
            {"key": "component-index-hashes", "value": component_hashes},
            {"key": "device", "value": device},
            {"key": "dtype", "value": CANONICAL_DTYPE},
            {"key": "capture-command", "value": command},
        ],
        "outputs": outputs,
    }
    if extended_authority:
        document["adapter"]["implementation_sha256"] = adapter_sha256
        if role == "oracle":
            document["environment"]["oracle_runtime"] = runtime_identity
    if "adapter-implementation-sha256" in contract["required_provenance"]:
        provenance_values = {
            "source-revision": source_revision,
            "transformers-revision": TRANSFORMERS_REVISION,
            "adapter-implementation-sha256": adapter_sha256,
            "oracle-runtime-sha256": runtime_sha256,
            "component-index-hashes": component_hashes,
            "device": device,
            "dtype": CANONICAL_DTYPE,
            "capture-command": command,
        }
        document["provenance"] = [
            {"key": key, "value": provenance_values[key]}
            for key in contract["required_provenance"]
        ]
    if role == "oracle":
        document["comparison"] = [
            {
                "key": key,
                "absolute": 0,
                "relative": 0,
                "metric": "elementwise-atol-plus-rtol",
                "hash_policy": "exact",
            }
            for key in contract["required_measurements"]
        ]
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
    documents: Sequence[tuple[pathlib.Path, Mapping[str, Any], bytes]],
) -> dict[str, Any]:
    fixtures: list[dict[str, Any]] = []
    for path, document, encoded in documents:
        first = document["outputs"][0]
        fixtures.append(
            {
                "id": f"{document['case_id']}-{role}",
                "layer": LAYER_ID,
                "authority_tier": "exact-full-bf16",
                "relative_path": path.relative_to(fixture_root).as_posix(),
                "sha256": sha256_bytes(encoded),
                "component_index_sha256": TEXT_ENCODER_INDEX_SHA256,
                "component_indexes": document["input"]["component_indexes"],
                "tensor": expected_tensor_summary(first),
                "tolerance": {
                    "absolute": 0,
                    "relative": 0,
                    "metric": "elementwise-atol-plus-rtol",
                },
            }
        )
    bundle = {
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
        "fixtures": fixtures,
    }
    first_document = documents[0][1]
    if role == "oracle" and "oracle_runtime" in first_document["environment"]:
        bundle["capture_environment"]["oracle_runtime"] = first_document["environment"][
            "oracle_runtime"
        ]
        if layer_contract(manifest).get("oracle_adapter") is not None:
            bundle["capture_environment"]["oracle_adapters"] = [
                {
                    "layer": first_document["layer"],
                    "id": first_document["adapter"]["id"],
                    "command": first_document["adapter"]["command"],
                    "implementation_sha256": first_document["adapter"][
                        "implementation_sha256"
                    ],
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
    cases: Sequence[Mapping[str, Any]],
    activations: Mapping[str, tuple[list[int], list[int]]],
    runtime_identity: Mapping[str, str],
    conformance_tool: Any,
    gpu_tool: Any,
) -> tuple[list[pathlib.Path], pathlib.Path]:
    contract = gpu_tool.exact_layer_contracts(manifest)[LAYER_ID]
    documents: list[tuple[pathlib.Path, Mapping[str, Any], bytes]] = []
    for case in cases:
        activation = activations.get(case["case_id"])
        if activation is None:
            fail("capture adapter omitted a prepared Qwen case")
        document = build_layer_document(
            manifest,
            role,
            source_revision,
            authorization_sha256,
            device,
            case,
            activation,
            runtime_identity,
        )
        gpu_tool.validate_manifest_layer_evidence(document, contract, role)
        conformance_tool.validate_schema(
            document, conformance_tool.LAYER_OUTPUT_SCHEMA_PATH
        )
        path = capture_dir / f"{role}-{case['case_id']}.json"
        documents.append((path, document, document_bytes(document)))
    bundle_path = capture_dir / f"{role}-bundle.json"
    if bundle_path.exists() or any(path.exists() for path, _, _ in documents):
        fail("Qwen layer capture already exists; refusing to overwrite evidence")
    bundle = build_bundle(
        manifest,
        role,
        source_revision,
        authorization_sha256,
        device,
        fixture_root,
        documents,
    )
    conformance_tool.validate_schema(bundle, conformance_tool.BUNDLE_SCHEMA_PATH)
    created: list[pathlib.Path] = []
    try:
        for path, _, encoded in documents:
            BASE.exclusive_write(path, encoded)
            created.append(path)
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
    return [path for path, _, _ in documents], bundle_path


def run_capture(
    environment: Mapping[str, str], role: str, device: str
) -> tuple[list[pathlib.Path], pathlib.Path]:
    if role not in {"oracle", "mold"}:
        fail("capture role must be oracle or mold")
    BASE.reject_excluded_acceleration_environment(environment)
    values = required_environment(environment)
    conformance_tool = load_module("minimax_h3_qwen_conformance", CONFORMANCE_TOOL_PATH)
    gpu_tool = load_module("minimax_h3_qwen_gpu_conformance", GPU_CONFORMANCE_TOOL_PATH)
    manifest = conformance_tool.validate_manifest()
    layer_contract(manifest)
    if not isinstance(manifest["numerical_authority"].get("oracle_runtime"), dict):
        fail("Qwen capture requires the reviewed exact oracle runtime authority")
    if conformance_tool.EXPECTED_REVISIONS["diffusers"] != DIFFUSERS_REVISION:
        fail("Qwen capture Diffusers revision differs from the checked pin")
    if conformance_tool.EXPECTED_REVISIONS["minimax-official-model"] != MODEL_REVISION:
        fail("Qwen capture model revision differs from the checked pin")

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
        fail("Qwen capture authorization differs from the reviewed evidence")
    authorization_source_path = BASE.external_file(
        "authorization source document", authorization["source_document_path"]
    )
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
    manifest_identity = ExternalFileIdentity.capture(MANIFEST_PATH)
    producer_identity = ExternalFileIdentity.capture(pathlib.Path(__file__).resolve())
    authorization_identity = ExternalFileIdentity.capture(authorization_path)
    authorization_source_identity = ExternalFileIdentity.capture(
        authorization_source_path
    )
    protected_inputs = [
        manifest_identity,
        producer_identity,
        authorization_identity,
        authorization_source_identity,
    ]
    manifest_sha256 = BASE.sha256_file(MANIFEST_PATH)
    producer_sha256 = BASE.sha256_file(pathlib.Path(__file__).resolve())
    authorization_sha256 = BASE.sha256_file(authorization_path)
    authorization_source_sha256 = BASE.sha256_file(authorization_source_path)

    mold_revision = ""
    if role == "mold":
        mold_revision = BASE.command_output(["git", "rev-parse", "HEAD"], REPO_ROOT)
        if not lower_hex(mold_revision, 40):
            fail("Mold checkout does not have a canonical source revision")
        BASE.verify_git_checkout(
            "Mold",
            REPO_ROOT,
            mold_revision,
            "https://github.com/utensils/mold",
        )
        for path in (
            REPO_ROOT
            / "crates"
            / "mold-inference"
            / "src"
            / "bin"
            / "h3_qwen_layer50_capture.rs",
            REPO_ROOT / "crates" / "mold-inference" / "Cargo.toml",
            REPO_ROOT / "Cargo.lock",
        ):
            protected_inputs.append(ExternalFileIdentity.capture(path))

    with tempfile.TemporaryDirectory(
        prefix=".h3-qwen-layer50-stage-", dir=fixture_root
    ) as staged_value:
        staged_root = pathlib.Path(staged_value).resolve(strict=True)
        os.chmod(staged_root, 0o700)
        staged_diffusers = staged_root / "sources" / "diffusers"
        staged_transformers = staged_root / "sources" / "transformers"
        staged_model = staged_root / "model"
        staged_mold = staged_root / "mold-source"
        extract_package_snapshot(
            "Diffusers",
            diffusers_checkout,
            DIFFUSERS_REVISION,
            "https://github.com/huggingface/diffusers",
            "diffusers",
            staged_diffusers,
        )
        extract_package_snapshot(
            "Transformers",
            transformers_checkout,
            TRANSFORMERS_REVISION,
            "https://github.com/huggingface/transformers",
            "transformers",
            staged_transformers,
        )
        stage_model_components(model_root, staged_model, manifest)
        if role == "mold":
            extract_mold_snapshot(REPO_ROOT, mold_revision, staged_mold)
        staged_snapshot = seal_staging(staged_root)

        runtime = load_runtime(staged_diffusers, staged_transformers)
        runtime_identity = observed_runtime(runtime)
        expected_runtime = manifest["numerical_authority"]["oracle_runtime"]
        if runtime_identity != expected_runtime:
            fail("Qwen capture runtime identity differs from the reviewed oracle pin")
        cases = build_cases(runtime, staged_model)
        if tuple(case["case_id"] for case in cases) != CASE_IDS:
            fail("Qwen capture cases are not in their reviewed order")
        checkpoint_inputs = verify_text_encoder_snapshot(model_root, manifest)
        capture_dir = capture_directory(fixture_root, cases)
        if role == "oracle":
            source_revision = DIFFUSERS_REVISION
            activations = capture_oracle(
                runtime, model_root, staged_model, cases, device
            )
        else:
            BASE.configure_torch(runtime, device)
            source_revision, activations = capture_mold(
                fixture_root,
                capture_dir,
                authorization_path,
                authorization["source_document_sha256"],
                model_root,
                cases,
                device,
                mold_revision,
                staged_mold,
                manifest,
            )

        staged_snapshot.revalidate()
        for identity in (*protected_inputs, *checkpoint_inputs):
            identity.revalidate()
        if BASE.sha256_file(MANIFEST_PATH) != manifest_sha256:
            fail("checked conformance manifest changed during Qwen capture")
        if BASE.sha256_file(pathlib.Path(__file__).resolve()) != producer_sha256:
            fail("Qwen capture producer changed during execution")
        if BASE.sha256_file(authorization_path) != authorization_sha256:
            fail("Qwen capture authorization record changed during execution")
        if BASE.sha256_file(authorization_source_path) != authorization_source_sha256:
            fail("Qwen capture authorization source changed during execution")
        return write_capture(
            fixture_root,
            capture_dir,
            authorization_path,
            manifest,
            role,
            source_revision,
            authorization["source_document_sha256"],
            device,
            cases,
            activations,
            runtime_identity,
            conformance_tool,
            gpu_tool,
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture authorization-bound exact-BF16 Qwen hidden_states[50] evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser(
        "capture", help="capture text-only and representative multimodal layer-50 cases"
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
            fail("unsupported Qwen capture command")
        layer_paths, bundle_path = run_capture(os.environ, args.role, args.device)
        print(
            "MiniMax H3 Qwen layer-50 capture complete: "
            f"role={args.role} layers={len(layer_paths)} bundle={bundle_path.name}"
        )
        return 0
    except (CaptureFailure, OSError, subprocess.SubprocessError) as error:
        print(f"MiniMax H3 Qwen layer-50 capture rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
