#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MOBILE_DIST="$ROOT/desktop/dist-mobile"
SOURCE_ICON="$ROOT/desktop/icon-master.png"
APP_ICON="$ROOT/apps/mobile/src-tauri/app-icon.png"
SOURCE_CHECKSUM="$ROOT/apps/mobile/src-tauri/app-icon-source.sha256"
APP_ICON_SET="$ROOT/apps/mobile/src-tauri/gen/apple/Assets.xcassets/AppIcon.appiconset"
APP_STORE_ICON="$APP_ICON_SET/AppIcon-512@2x.png"
CATALOG_CHECKSUM="$APP_ICON_SET/SHA256SUMS"

fail() {
  echo "ios-release-assets: $*" >&2
  exit 1
}

(
  cd "$ROOT/desktop"
  bun run build:mobile >/dev/null
)

test -f "$MOBILE_DIST/index.html" ||
  fail "mobile frontend must emit index.html for Tauri"

test ! -e "$MOBILE_DIST/index.mobile.html" ||
  fail "mobile frontend emitted index.mobile.html instead of index.html"

grep -Fq 'mold-mobile-entry-v1' "$MOBILE_DIST/index.html" ||
  fail "mobile frontend is missing its archive smoke-test marker"

test -f "$APP_ICON" ||
  fail "authoritative iOS app icon is missing"

test -f "$APP_STORE_ICON" ||
  fail "generated App Store icon is missing"

cmp -s "$APP_ICON" "$APP_STORE_ICON" ||
  fail "generated App Store icon does not match the Mold iOS icon"

python3 - \
  "$SOURCE_ICON" \
  "$APP_ICON" \
  "$SOURCE_CHECKSUM" \
  "$APP_ICON_SET/Contents.json" \
  "$CATALOG_CHECKSUM" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
path = sys.argv[2]
source_checksum_path = Path(sys.argv[3])
contents_path = Path(sys.argv[4])
catalog_checksum_path = Path(sys.argv[5])


def png_header(icon_path: Path) -> tuple[int, int, int, int]:
    with icon_path.open("rb") as icon:
        signature = icon.read(8)
        length = struct.unpack(">I", icon.read(4))[0]
        chunk = icon.read(4)
        data = icon.read(length)
    if signature != b"\x89PNG\r\n\x1a\n" or chunk != b"IHDR":
        raise SystemExit(f"ios-release-assets: {icon_path.name} is not a PNG")
    return struct.unpack(">IIBB", data[:10])


with contents_path.open(encoding="utf-8") as contents_file:
    images = json.load(contents_file)["images"]

if not source_checksum_path.is_file():
    raise SystemExit("ios-release-assets: iOS icon source checksum is missing")
expected_source_hash = source_checksum_path.read_text(encoding="utf-8").strip()
actual_source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
if actual_source_hash != expected_source_hash:
    raise SystemExit("ios-release-assets: regenerate iOS icons after changing icon-master.png")

if not catalog_checksum_path.is_file():
    raise SystemExit("ios-release-assets: iOS icon catalog checksum is missing")
catalog_hashes = {}
for line in catalog_checksum_path.read_text(encoding="utf-8").splitlines():
    digest, filename = line.split(maxsplit=1)
    catalog_hashes[filename.strip()] = digest

catalog_filenames = {image["filename"] for image in images if image.get("filename")}
if set(catalog_hashes) != catalog_filenames:
    raise SystemExit("ios-release-assets: iOS icon catalog checksum is incomplete")

for image in images:
    filename = image.get("filename")
    if not filename:
        continue
    icon_path = contents_path.parent / filename
    if not icon_path.is_file():
        raise SystemExit(f"ios-release-assets: missing catalog icon {filename}")
    width, height, bit_depth, color_type = png_header(icon_path)
    points = float(image["size"].split("x", 1)[0])
    scale = int(image["scale"].removesuffix("x"))
    expected = round(points * scale)
    if (width, height) != (expected, expected):
        raise SystemExit(
            f"ios-release-assets: {filename} must be {expected}x{expected}, "
            f"got {width}x{height}"
        )
    if bit_depth != 8 or color_type != 2:
        raise SystemExit(
            f"ios-release-assets: {filename} must be opaque 8-bit RGB without alpha"
        )
    if hashlib.sha256(icon_path.read_bytes()).hexdigest() != catalog_hashes[filename]:
        raise SystemExit(f"ios-release-assets: {filename} does not match the generated catalog")

source = Path(path)
if png_header(source) != (1024, 1024, 8, 2):
    raise SystemExit("ios-release-assets: authoritative iOS icon must be opaque 1024px RGB")

stock_tauri_sha256 = "00badc78c2ecb7460bbe02aeeca1e8848724ee133391bb3025df561007187eaf"
if hashlib.sha256(source.read_bytes()).hexdigest() == stock_tauri_sha256:
    raise SystemExit("ios-release-assets: stock Tauri icon is not Mold branding")
PY

echo "ios-release-assets: ok"
