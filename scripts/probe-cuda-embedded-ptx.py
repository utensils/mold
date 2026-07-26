#!/usr/bin/env python3
"""Load PTX extracted from an exact Mold executable through the CUDA driver.

CUDA_FORCE_PTX_JIT affects NVIDIA runtime libraries as well as Mold kernels and
can fail during cuBLAS initialization before Mold PTX is reached. This probe
instead extracts complete entry modules from the named executable and submits
those exact bytes to cuModuleLoadData on the CUDA_VISIBLE_DEVICES device 0.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import re
import sys
from pathlib import Path

CUDA_SUCCESS = 0
CUDA_ERROR_NO_BINARY_FOR_GPU = 209
CUDA_ERROR_INVALID_PTX = 218
INCOMPATIBLE_RESULTS = {
    CUDA_ERROR_NO_BINARY_FOR_GPU,
    CUDA_ERROR_INVALID_PTX,
}
VERSION_RE = re.compile(rb"\.version[ \t]+[0-9]+(?:\.[0-9]+)?")


def fail(message: str) -> None:
    print(f"embedded PTX probe: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_entry_modules(data: bytes, compute_cap: str) -> list[dict[str, object]]:
    """Extract self-contained first-entry PTX candidates for an exact target.

    Rust stores PTX as string data without a guaranteed NUL terminator in the
    final ELF. Starting at each `.version`, retain the header and first complete
    `.entry` body. Most Candle modules are self-contained at that boundary; the
    driver tries candidates in order because an entry may reference a helper
    function that appears later in its original module.
    """

    target_re = re.compile(
        rb"\.target[ \t]+sm_" + compute_cap.encode("ascii") + rb"(?=[ \t\r\n,])"
    )
    candidates: list[dict[str, object]] = []
    seen_hashes: set[str] = set()

    for version_match in VERSION_RE.finditer(data):
        start = version_match.start()
        search_end = min(len(data), start + 64 * 1024)
        entry = data.find(b".entry", start, search_end)
        if entry < 0 or target_re.search(data, start, entry) is None:
            continue
        opening_brace = data.find(b"{", entry, min(search_end, entry + 16 * 1024))
        if opening_brace < 0:
            continue

        depth = 0
        end = None
        body_limit = min(len(data), opening_brace + 2 * 1024 * 1024)
        for index in range(opening_brace, body_limit):
            byte = data[index]
            if byte == ord("{"):
                depth += 1
            elif byte == ord("}"):
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            continue

        ptx = data[start:end] + b"\n"
        ptx_sha256 = sha256_bytes(ptx)
        if ptx_sha256 in seen_hashes:
            continue
        seen_hashes.add(ptx_sha256)
        candidates.append(
            {
                "offset": start,
                "length": len(ptx),
                "ptx_sha256": ptx_sha256,
                "_ptx": ptx,
            }
        )

    return candidates


class CudaDriver:
    def __init__(self) -> None:
        library = ctypes.util.find_library("cuda") or "libcuda.so.1"
        try:
            self.lib = ctypes.CDLL(library)
        except OSError as error:
            fail(f"cannot load CUDA driver library: {error}")

        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGet.restype = ctypes.c_int
        self.lib.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.lib.cuCtxCreate_v2.restype = ctypes.c_int
        self.lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cuCtxDestroy_v2.restype = ctypes.c_int
        self.lib.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.lib.cuModuleLoadData.restype = ctypes.c_int
        self.lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.lib.cuModuleUnload.restype = ctypes.c_int
        self.lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuGetErrorName.restype = ctypes.c_int

        result = self.lib.cuInit(0)
        if result != CUDA_SUCCESS:
            fail(f"cuInit failed: {self.error_name(result)} ({result})")
        device = ctypes.c_int()
        result = self.lib.cuDeviceGet(ctypes.byref(device), 0)
        if result != CUDA_SUCCESS:
            fail(f"cuDeviceGet(0) failed: {self.error_name(result)} ({result})")
        self.context = ctypes.c_void_p()
        result = self.lib.cuCtxCreate_v2(
            ctypes.byref(self.context),
            0,
            device,
        )
        if result != CUDA_SUCCESS:
            fail(f"cuCtxCreate_v2 failed: {self.error_name(result)} ({result})")

    def error_name(self, result: int) -> str:
        name = ctypes.c_char_p()
        if self.lib.cuGetErrorName(result, ctypes.byref(name)) == CUDA_SUCCESS and name.value:
            return name.value.decode("ascii", "replace")
        return f"CUDA_ERROR_{result}"

    def load(self, ptx: bytes) -> tuple[int, str]:
        source = ctypes.create_string_buffer(ptx + b"\0")
        module = ctypes.c_void_p()
        result = self.lib.cuModuleLoadData(
            ctypes.byref(module),
            ctypes.cast(source, ctypes.c_void_p),
        )
        if result == CUDA_SUCCESS:
            self.lib.cuModuleUnload(module)
        return result, self.error_name(result)

    def close(self) -> None:
        if self.context:
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = ctypes.c_void_p()


def public_candidate(candidate: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in candidate.items() if key != "_ptx"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("compute_cap", help="numeric compute capability, for example 86")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="validate and describe embedded PTX without opening CUDA",
    )
    parser.add_argument(
        "--expect-incompatible",
        action="store_true",
        help="pass only when every candidate is rejected as incompatible",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9]+", args.compute_cap):
        fail("compute_cap must be numeric, for example 86")
    if not args.binary.is_file():
        fail(f"binary does not exist: {args.binary}")
    if args.extract_only and args.expect_incompatible:
        fail("--extract-only and --expect-incompatible cannot be combined")

    data = args.binary.read_bytes()
    candidates = extract_entry_modules(data, args.compute_cap)
    if not candidates:
        fail(f"no embedded sm_{args.compute_cap} PTX entry modules found")

    result: dict[str, object] = {
        "artifact_path": str(args.binary.resolve()),
        "artifact_sha256": sha256_file(args.binary),
        "expected_target": f"sm_{args.compute_cap}",
        "candidate_count": len(candidates),
        "candidates": [public_candidate(candidate) for candidate in candidates],
        "attempts": [],
        "loaded": False,
        "expect_incompatible": args.expect_incompatible,
    }
    if args.extract_only:
        print(json.dumps(result, sort_keys=True))
        return 0

    driver = CudaDriver()
    try:
        attempts: list[dict[str, object]] = []
        for candidate in candidates:
            cuda_result, cuda_error_name = driver.load(candidate["_ptx"])  # type: ignore[arg-type]
            attempt = {
                **public_candidate(candidate),
                "cuda_result": cuda_result,
                "cuda_error_name": cuda_error_name,
                "loaded": cuda_result == CUDA_SUCCESS,
            }
            attempts.append(attempt)
            if cuda_result == CUDA_SUCCESS:
                result["loaded"] = True
                if not args.expect_incompatible:
                    break
        result["attempts"] = attempts
    finally:
        driver.close()

    print(json.dumps(result, sort_keys=True))
    if args.expect_incompatible:
        attempts = result["attempts"]
        return int(
            bool(result["loaded"])
            or not attempts
            or any(
                attempt["cuda_result"] not in INCOMPATIBLE_RESULTS
                for attempt in attempts  # type: ignore[union-attr]
            )
        )
    return 0 if result["loaded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
