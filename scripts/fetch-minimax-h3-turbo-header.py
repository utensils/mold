#!/usr/bin/env python3
"""Fetch and verify one MiniMax H3 Turbo LoRA safetensors header from HF.

Purpose
-------
Captures a golden `.header` fixture: the exact eight-byte little-endian
safetensors header-length prefix followed by the exact published JSON
header bytes, with no tensor payload attached. These fixtures back
`published_tier_pins_are_recomputed_from_the_checked_in_headers` in
`crates/mold-candle/src/minimax_h3/turbo_lora.rs`, which recomputes every
Turbo tier's `header_len`, `header_identity_sha256`, payload byte count,
file size, tensor count, and training metadata from the checked-in blob
rather than restating them as literals. This script is how those blobs
are produced, and it is safe (and expected) to re-run to reproduce or
re-verify a fixture already checked in.

Provenance
----------
Two ranged HTTP GETs against
`https://huggingface.co/{repo}/resolve/{revision}/{path}`: the first eight
bytes give the little-endian header length `N`, then the next `N` bytes
are the header JSON itself. Each request MUST answer HTTP 206 (Partial
Content) — a 200 would mean the range request was silently upgraded to a
full-file transfer (a misbehaving mirror or proxy sitting in front of the
canonical host), which for a multi-gigabyte checkpoint must fail loudly
before it can stream even one byte of payload, never be read quietly to
completion. The script then cross-checks the header-derived file size
against the HF repository tree API's own recorded `size` and `lfs.oid`
for that exact path at that exact revision, so the header pin and the
published artifact identity are corroborated independently of one
another. That cross-check runs BEFORE the output file is opened, so a
mismatched fixture never reaches disk.

Companion of `scripts/capture-minimax-h3-*.py`; unlike those, this tool
never opens a checkpoint payload and needs no fixture root or
authorization record — it captures only structural header bytes plus HF's
own published size and digest, so it is always safe to run and its output
is always safe to check in.

Usage
-----
    python3 scripts/fetch-minimax-h3-turbo-header.py \\
        --repo lightx2v/Minimax-h3-Turbo \\
        --revision 05ef678438e84933c406131b59abbf86919b3aac \\
        --path minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors \\
        --out crates/mold-candle/testdata/minimax_h3/turbo/fl2v-4step-768p-v1.1.header

Prints one JSON object summarizing what it found (repo, revision, path,
header_len, header_identity_sha256, payload_bytes, file_bytes, lfs_sha256,
tensor_count, metadata) so the caller can diff it against the pins
recorded in code and in `crates/mold-candle/testdata/minimax_h3/turbo/README.md`
before committing the fixture. Honours `HF_TOKEN` if set (not required for
the public apache-2.0 repositories this script has been used against).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import struct
import sys
import urllib.error
import urllib.request
from typing import Any

HF_HOST = "https://huggingface.co"

# __metadata__ keys worth echoing in the summary when a published header
# carries them. Not every tier has every key: rank-uniform lightx2v/Comfy
# adapters carry training_rank/training_alpha/training_scale; drbaph's
# dynamic-rank resized adapters carry baked_scale/resized_from instead.
METADATA_ECHO_KEYS = (
    "training_rank",
    "training_alpha",
    "training_scale",
    "baked_scale",
    "resized_from",
)


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _ranged_get(url: str, start: int, end: int) -> bytes:
    """GET an inclusive byte range and REQUIRE HTTP 206.

    The status is checked BEFORE any bytes are read off the response body,
    so a mirror that ignores Range and answers 200 is refused before this
    function reads even one byte of what could be a multi-gigabyte file.
    """
    length = end - start + 1
    headers = {"Range": f"bytes={start}-{end}", **_auth_headers()}
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"refusing: HTTP {error.code} for {url} (wanted bytes {start}-{end}): {error.reason}"
        )
    try:
        status = response.status
        if status != 206:
            raise SystemExit(
                f"refusing: expected HTTP 206 Partial Content, got {status} for "
                f"{url} (wanted bytes {start}-{end}); a mirror that streams the "
                "whole file on a Range request must never be read silently"
            )
        body = response.read(length)
    finally:
        response.close()
    if len(body) != length:
        raise SystemExit(
            f"expected {length} bytes from {url} (bytes {start}-{end}), got {len(body)}"
        )
    return body


def fetch_header(repo: str, revision: str, path: str) -> tuple[bytes, bytes]:
    """Return `(length_prefix, json_bytes)` for one safetensors file."""
    url = f"{HF_HOST}/{repo}/resolve/{revision}/{path}"
    prefix = _ranged_get(url, 0, 7)
    (header_len,) = struct.unpack("<Q", prefix)
    if header_len == 0:
        raise SystemExit(f"safetensors header length is zero for {url}")
    json_bytes = _ranged_get(url, 8, 8 + header_len - 1)
    return prefix, json_bytes


def _next_link(link_header: str | None) -> str | None:
    """Parse a GitHub-style `Link: <url>; rel="next"` header, if present."""
    if not link_header:
        return None
    for part in link_header.split(","):
        segment = part.strip()
        if not segment.endswith('rel="next"'):
            continue
        start, end = segment.find("<"), segment.find(">")
        if start != -1 and end != -1:
            return segment[start + 1 : end]
    return None


def _tree_entries(repo: str, revision: str, directory: str) -> list[dict[str, Any]]:
    """Every entry of one tree-API directory, following pagination."""
    base = f"{HF_HOST}/api/models/{repo}/tree/{revision}"
    url = f"{base}/{directory}" if directory else base
    entries: list[dict[str, Any]] = []
    while url:
        request = urllib.request.Request(url, headers=_auth_headers())
        with urllib.request.urlopen(request, timeout=60) as response:
            entries.extend(json.loads(response.read().decode("utf-8")))
            url = _next_link(response.headers.get("Link"))
    return entries


def lookup_tree_entry(repo: str, revision: str, path: str) -> dict[str, Any]:
    """The tree-API entry for `path`, found by following its directory."""
    directory = posixpath.dirname(path)
    for entry in _tree_entries(repo, revision, directory):
        if entry.get("path") == path:
            return entry
    raise SystemExit(f"{path!r} not found in {repo}@{revision} tree (dir {directory!r})")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and pin a MiniMax H3 Turbo LoRA safetensors header from HF.",
    )
    parser.add_argument("--repo", required=True, help="owner/name, e.g. lightx2v/Minimax-h3-Turbo")
    parser.add_argument("--revision", required=True, help="exact pinned commit SHA")
    parser.add_argument("--path", required=True, help="repo-relative file path")
    parser.add_argument("--out", required=True, help="fixture path to write prefix||json to")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    prefix, json_bytes = fetch_header(args.repo, args.revision, args.path)

    try:
        header = json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"header is not strict JSON: {error}")
    if not isinstance(header, dict):
        raise SystemExit("header JSON root must be an object")

    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise SystemExit("__metadata__ must be an object")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise SystemExit("header declares no tensors")

    payload_bytes = 0
    for name, tensor in tensors.items():
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if not (isinstance(offsets, list) and len(offsets) == 2):
            raise SystemExit(f"tensor {name!r} has no valid data_offsets")
        payload_bytes = max(payload_bytes, int(offsets[1]))

    header_len = len(json_bytes)
    file_bytes = 8 + header_len + payload_bytes
    header_identity_sha256 = hashlib.sha256(prefix + json_bytes).hexdigest()

    # Cross-check BEFORE writing anything. The size the header implies must
    # equal the size the repository records for this exact path at this exact
    # revision; a mirror that ignored `Range`, or a fixture pulled from the
    # wrong revision, fails here with no bytes on disk to clean up.
    entry = lookup_tree_entry(args.repo, args.revision, args.path)
    tree_size = entry.get("size")
    if tree_size != file_bytes:
        raise SystemExit(
            "file size mismatch -- STOP, do not adjust expectations: "
            f"header-derived {file_bytes} bytes, HF tree API reports {tree_size} "
            f"bytes for {args.path!r} ({args.repo}@{args.revision})"
        )
    lfs_sha256 = (entry.get("lfs") or {}).get("oid")

    out_path = args.out
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(prefix)
        handle.write(json_bytes)

    summary = {
        "repo": args.repo,
        "revision": args.revision,
        "path": args.path,
        "header_len": header_len,
        "header_identity_sha256": header_identity_sha256,
        "payload_bytes": payload_bytes,
        "file_bytes": file_bytes,
        "lfs_sha256": lfs_sha256,
        "tensor_count": len(tensors),
        "metadata": {key: metadata[key] for key in METADATA_ECHO_KEYS if key in metadata},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
