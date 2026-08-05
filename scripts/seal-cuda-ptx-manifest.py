#!/usr/bin/env python3
"""Seal a CUDA executable with a manifest for its generated Candle PTX.

The manifest records exact byte ranges in the executable's ELF `.rodata`
section which also match PTX emitted by the current candle-kernels build.
Release verification reads only those ranges; unrelated printable bytes in the
ELF cannot become qualification evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MANIFEST_SECTION = ".mold_cuda_ptx"
SCHEMA = "mold.cuda.ptx-manifest.v1"
PTX_INCLUDE_RE = re.compile(r'"/([^"/]+\.ptx)"')


def fail(message: str) -> None:
    print(f"seal CUDA PTX manifest: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_section(binary: Path, section: str, output: Path) -> bytes:
    objcopy = os.environ.get("OBJCOPY", "objcopy")
    input_sha256 = sha256_file(binary)
    rewritten_elf = output.with_name(f"{output.name}.elf")
    result = run(
        objcopy,
        "--dump-section",
        f"{section}={output}",
        str(binary),
        str(rewritten_elf),
    )
    if result.returncode != 0 or not output.is_file():
        fail(f"could not extract ELF {section}: {result.stderr.strip()}")
    if sha256_file(binary) != input_sha256:
        fail("ELF section inventory mutated the unsealed input artifact")
    return output.read_bytes()


def current_kernel_output_dirs(build_root: Path, compute_cap: str) -> list[Path]:
    matches: list[Path] = []
    for ptx_rs in build_root.glob("candle-kernels-*/out/ptx.rs"):
        output_dir = ptx_rs.parent
        cache = output_dir / ".cudaforge_cache.json"
        if cache.is_file():
            if cudaforge_cache_matches(output_dir, compute_cap):
                matches.append(output_dir)
            continue

        output = output_dir.parent / "output"
        marker = f"cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap}"
        if output.is_file() and marker in output.read_text(
            encoding="utf-8", errors="replace"
        ):
            matches.append(output_dir)
    return matches


def cudaforge_cache_matches(output_dir: Path, compute_cap: str) -> bool:
    cache_path = output_dir / ".cudaforge_cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    if not isinstance(cache, dict):
        return False
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        return False

    expected_arch = f"sm_{compute_cap}"
    referenced_names = {path.name for path in referenced_ptx_files(output_dir)}
    observed: dict[str, set[str]] = {}
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        object_path = entry.get("object_path")
        gpu_arch = entry.get("gpu_arch")
        if not isinstance(object_path, str) or not isinstance(gpu_arch, str):
            continue
        name = Path(object_path).name
        if name in referenced_names:
            observed.setdefault(name, set()).add(gpu_arch)

    return bool(referenced_names) and all(
        observed.get(name) == {expected_arch} for name in referenced_names
    )


def referenced_ptx_files(output_dir: Path) -> list[Path]:
    ptx_rs = output_dir / "ptx.rs"
    names = PTX_INCLUDE_RE.findall(
        ptx_rs.read_text(encoding="utf-8", errors="strict")
    )
    if not names:
        fail(f"{ptx_rs} does not reference generated PTX")
    return [output_dir / name for name in names]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("compute_cap")
    parser.add_argument(
        "build_root",
        type=Path,
        help="Cargo profile build directory, for example target/release/build",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9]+", args.compute_cap):
        fail("compute_cap must be numeric")
    if not args.binary.is_file():
        fail(f"binary does not exist: {args.binary}")
    output_dirs = current_kernel_output_dirs(args.build_root, args.compute_cap)
    if not output_dirs:
        fail(
            f"no candle-kernels output for CUDA_COMPUTE_CAP={args.compute_cap} "
            f"under {args.build_root}"
        )

    readelf = run("readelf", "-SW", str(args.binary))
    if readelf.returncode != 0:
        fail(f"could not inspect ELF sections: {readelf.stderr.strip()}")
    if MANIFEST_SECTION in readelf.stdout:
        fail(f"{args.binary} already contains {MANIFEST_SECTION}")

    temp_parent = os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(
        prefix="mold-cuda-ptx-seal-", dir=temp_parent
    ) as temp:
        temp_dir = Path(temp)
        rodata = dump_section(args.binary, ".rodata", temp_dir / "rodata.bin")
        modules: dict[str, dict[str, object]] = {}
        for output_dir in output_dirs:
            for ptx_path in referenced_ptx_files(output_dir):
                if not ptx_path.is_file():
                    fail(f"generated PTX is missing: {ptx_path}")
                ptx = ptx_path.read_bytes()
                offset = rodata.find(ptx)
                if offset < 0:
                    # Dead code can remove an otherwise generated module.
                    continue
                if rodata.find(ptx, offset + 1) >= 0:
                    fail(f"generated PTX has ambiguous duplicate ranges: {ptx_path}")
                digest = hashlib.sha256(ptx).hexdigest()
                modules[digest] = {
                    "source_name": ptx_path.name,
                    "rodata_offset": offset,
                    "byte_length": len(ptx),
                    "sha256": digest,
                }
        if not modules:
            fail("no generated candle-kernels PTX is present in ELF .rodata")

        manifest = {
            "schema_version": SCHEMA,
            "cuda_target": f"sm_{args.compute_cap}",
            "modules": sorted(
                modules.values(),
                key=lambda module: (
                    int(module["rodata_offset"]),
                    str(module["source_name"]),
                ),
            ),
        }
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        objcopy = os.environ.get("OBJCOPY", "objcopy")
        result = run(
            objcopy,
            "--add-section",
            f"{MANIFEST_SECTION}={manifest_path}",
            "--set-section-flags",
            f"{MANIFEST_SECTION}=readonly,contents",
            str(args.binary),
        )
        if result.returncode != 0:
            fail(f"could not add {MANIFEST_SECTION}: {result.stderr.strip()}")

    print(
        f"sealed {args.binary} with {len(modules)} generated "
        f"sm_{args.compute_cap} PTX modules"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
