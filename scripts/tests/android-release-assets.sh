#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_ICON="$ROOT/desktop/icon-master.png"
MOBILE="$ROOT/apps/mobile/src-tauri"
SOURCE_CHECKSUM="$MOBILE/android-icon-source.sha256"
ANDROID_RES="$MOBILE/gen/android/app/src/main/res"
CATALOG_CHECKSUM="$ANDROID_RES/ANDROID_ICON_SHA256SUMS"
MANIFEST="$MOBILE/gen/android/app/src/main/AndroidManifest.xml"

fail() {
  echo "android-release-assets: $*" >&2
  exit 1
}

test -f "$SOURCE_CHECKSUM" || fail "Android icon source checksum is missing"
test -f "$CATALOG_CHECKSUM" || fail "Android icon catalog checksum is missing"

expected_source_hash=$(tr -d '[:space:]' < "$SOURCE_CHECKSUM")
actual_source_hash=$(shasum -a 256 "$SOURCE_ICON" | awk '{print $1}')
[[ "$actual_source_hash" == "$expected_source_hash" ]] ||
  fail "regenerate Android icons after changing icon-master.png"

grep -Fq 'android:icon="@mipmap/ic_launcher"' "$MANIFEST" ||
  fail "AndroidManifest.xml must use the generated launcher icon"
grep -Fq 'android:roundIcon="@mipmap/ic_launcher_round"' "$MANIFEST" ||
  fail "AndroidManifest.xml must use the generated round launcher icon"

(
  cd "$ANDROID_RES"
  shasum -a 256 --check ANDROID_ICON_SHA256SUMS >/dev/null
) || fail "generated Android launcher icons do not match their catalog"

python3 - "$ANDROID_RES" "$CATALOG_CHECKSUM" <<'PY'
import hashlib
import struct
import sys
from pathlib import Path

res = Path(sys.argv[1])
catalog_path = Path(sys.argv[2])

expected_sizes = {
    "mipmap-mdpi": (48, 108),
    # tauri-cli 2.11.4 emits a 49px legacy HDPI asset; keep this exact so the
    # committed catalog cannot silently drift from the pinned generator.
    "mipmap-hdpi": (49, 162),
    "mipmap-xhdpi": (96, 216),
    "mipmap-xxhdpi": (144, 324),
    "mipmap-xxxhdpi": (192, 432),
}


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as icon:
        if icon.read(8) != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"android-release-assets: {path} is not a PNG")
        length = struct.unpack(">I", icon.read(4))[0]
        if icon.read(4) != b"IHDR":
            raise SystemExit(f"android-release-assets: {path} has no PNG header")
        width, height = struct.unpack(">II", icon.read(length)[:8])
    return width, height


catalog_files = {
    line.split(maxsplit=1)[1].strip()
    for line in catalog_path.read_text(encoding="utf-8").splitlines()
}
required_files = {"mipmap-anydpi-v26/ic_launcher.xml", "values/ic_launcher_background.xml"}

for density, (legacy_size, foreground_size) in expected_sizes.items():
    for name, size in (
        ("ic_launcher.png", legacy_size),
        ("ic_launcher_round.png", legacy_size),
        ("ic_launcher_foreground.png", foreground_size),
    ):
        relative = f"{density}/{name}"
        path = res / relative
        if not path.is_file():
            raise SystemExit(f"android-release-assets: missing {relative}")
        actual_size = png_size(path)
        if actual_size != (size, size):
            raise SystemExit(
                f"android-release-assets: {relative} must be {size}x{size}, got {actual_size}"
            )
        required_files.add(relative)

if catalog_files != required_files:
    missing = sorted(required_files - catalog_files)
    extra = sorted(catalog_files - required_files)
    raise SystemExit(
        f"android-release-assets: icon checksum catalog mismatch; missing={missing}, extra={extra}"
    )

stock_tauri_hashes = {
    "dae1ff05b101efea50e4b622fe6a3af8ba8f761162fa7c4fd864adc7cb39eeac",
    "27cf0cdbc78bec8b9a14eaedb084c541a3c191fe5db89766e831fbfd21ce955d",
}
for relative in required_files:
    path = res / relative
    if path.suffix == ".png" and hashlib.sha256(path.read_bytes()).hexdigest() in stock_tauri_hashes:
        raise SystemExit(f"android-release-assets: {relative} is the stock Tauri icon")

adaptive = (res / "mipmap-anydpi-v26/ic_launcher.xml").read_text(encoding="utf-8")
if "@mipmap/ic_launcher_foreground" not in adaptive or "@color/ic_launcher_background" not in adaptive:
    raise SystemExit("android-release-assets: adaptive launcher icon layers are incomplete")
PY

echo "android-release-assets: ok"
