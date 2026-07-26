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


def dump_section(binary: Path, section: str, output: Path) -> bytes:
    objcopy = os.environ.get("OBJCOPY", "objcopy")
    result = run(objcopy, "--dump-section", f"{section}={output}", str(binary))
    if result.returncode != 0 or not output.is_file():
        fail(f"could not extract ELF {section}: {result.stderr.strip()}")
    return output.read_bytes()


def current_kernel_output_dirs(build_root: Path, compute_cap: str) -> list[Path]:
    matches: list[Path] = []
    for ptx_rs in build_root.glob("candle-kernels-*/out/ptx.rs"):
        build_dir = ptx_rs.parent.parent
        output = build_dir / "output"
        if not output.is_file():
            continue
        marker = f"cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap}"
        if marker in output.read_text(encoding="utf-8", errors="replace"):
            matches.append(ptx_rs.parent)
    return matches


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
