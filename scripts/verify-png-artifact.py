#!/usr/bin/env python3
"""Strict, dependency-free PNG decode check for hardware qualification."""

import binascii
import json
import struct
import sys
import zlib


def fail(message: str) -> None:
    print(f"invalid PNG: {message}", file=sys.stderr)
    raise SystemExit(1)


if len(sys.argv) != 4:
    fail("usage: verify-png-artifact.py <file> <width> <height>")

path, expected_width, expected_height = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
data = open(path, "rb").read()
if data[:8] != b"\x89PNG\r\n\x1a\n":
    fail("bad signature")

offset = 8
ihdr = None
idat = bytearray()
saw_iend = False
while offset < len(data):
    if offset + 12 > len(data):
        fail("truncated chunk")
    length = struct.unpack(">I", data[offset : offset + 4])[0]
    kind = data[offset + 4 : offset + 8]
    end = offset + 12 + length
    if end > len(data):
        fail("chunk extends beyond file")
    payload = data[offset + 8 : offset + 8 + length]
    expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
    actual_crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        fail(f"CRC mismatch in {kind!r}")
    if kind == b"IHDR":
        if ihdr is not None or length != 13:
            fail("invalid IHDR")
        ihdr = struct.unpack(">IIBBBBB", payload)
    elif kind == b"IDAT":
        idat.extend(payload)
    elif kind == b"IEND":
        if length != 0:
            fail("invalid IEND")
        saw_iend = True
        if end != len(data):
            fail("trailing bytes after IEND")
        break
    offset = end

if ihdr is None or not idat or not saw_iend:
    fail("missing IHDR, IDAT, or IEND")
width, height, depth, color_type, compression, filtering, interlace = ihdr
if (width, height) != (expected_width, expected_height):
    fail(f"dimensions are {width}x{height}, expected {expected_width}x{expected_height}")
if depth != 8 or compression != 0 or filtering != 0 or interlace != 0:
    fail("qualification output must be non-interlaced 8-bit PNG")
channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
if channels is None:
    fail(f"unsupported color type {color_type}")
try:
    decoded = zlib.decompress(bytes(idat))
except zlib.error as error:
    fail(f"IDAT cannot be decompressed: {error}")
expected_decoded = height * (1 + width * channels)
if len(decoded) != expected_decoded:
    fail(f"decoded byte count is {len(decoded)}, expected {expected_decoded}")
for row in range(height):
    filter_byte = decoded[row * (1 + width * channels)]
    if filter_byte > 4:
        fail(f"invalid row filter {filter_byte}")

print(json.dumps({"decoded": True, "width": width, "height": height}))
