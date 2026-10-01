#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
activity=apps/ios/Sources/Widgets/GenerationLiveActivity.swift
if grep -E 'UIImage|checkmark|wand.and.sparkles|exclamationmark.triangle.fill' "$activity"; then exit 1; fi
python3 - <<'CHECK'
from pathlib import Path
source = Path("apps/ios/Sources/Widgets/GenerationLiveActivity.swift").read_text()
leading = source.split("} compactLeading: {", 1)[1].split("} compactTrailing:", 1)[0]
assert "ActivityBrandIcon()" in leading
minimal = source.split("} minimal: {", 1)[1].split(".widgetURL", 1)[0]
assert "ActivityRing(state: context.state)" in minimal
preview = source.split("struct ActivityPreview:", 1)[1].split("private struct ActivityRing:", 1)[0]
assert "ActivityBrandIcon()" in preview and "context.state.preview" not in preview
ring = source.split("private struct ActivityRing:", 1)[1]
for phase, end in [("running", "finished"), ("finished", "failed"), ("failed", None)]:
    body = ring.split(f"case .{phase}:", 1)[1]
    if end:
        body = body.split(f"case .{end}:", 1)[0]
    assert "ActivityBrandIcon()" in body, phase
CHECK
asset=apps/ios/Sources/Widgets/Resources/Assets.xcassets/MoldLogo.imageset/MoldLogo.png
# WidgetKit rejects oversized image archives: keep the logo within 128 pixels.
python3 - "$asset" <<'PNG'
import struct, sys
with open(sys.argv[1], "rb") as image:
    header = image.read(24)
assert header[:8] == b"\x89PNG\r\n\x1a\n"
assert struct.unpack(">II", header[16:24]) == (128, 128)
PNG
grep -Fq 'Image("MoldLogo")' apps/ios/Sources/Widgets/ActivityBrandIcon.swift
printf 'iOS notification branding ok\n'
