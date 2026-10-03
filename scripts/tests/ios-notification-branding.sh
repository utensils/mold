#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
activity=apps/ios/Sources/Widgets/GenerationLiveActivity.swift
if grep -E 'UIImage|checkmark|wand.and.sparkles|exclamationmark.triangle.fill' "$activity"; then exit 1; fi
python3 - <<'CHECK'
from pathlib import Path
source = Path("apps/ios/Sources/Widgets/GenerationLiveActivity.swift").read_text()
# ActivityKit owns the Lock Screen material and its foreground environment.
# A separately resolved UIKit background can be light while text is white.
assert '.activityBackgroundTint(nil)' in source, 'Live Activities must use the system background material'
for path in Path("apps/ios/Sources/Widgets").glob('*.swift'):
    widget = path.read_text()
    assert 'preferredColorScheme' not in widget and '.environment(\\.colorScheme' not in widget, 'Do not force widget appearance'
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
# Notification Center resolves the bundle icon's light/dark slots. Use the
# same authored image in both so iOS need not synthesize a darkened logo.
python3 - <<'ICON'
import json
from pathlib import Path
root = Path("apps/ios/Sources/Companion/Resources/Assets.xcassets/AppIcon.appiconset")
images = json.loads((root / "Contents.json").read_text())["images"]
def appearance(image):
    return next((a["value"] for a in image.get("appearances", []) if a["appearance"] == "luminosity"), "any")
slots = {appearance(image): image for image in images}
assert "dark" in slots, "Notification Center needs an explicit dark icon variant"
assert slots["any"]["filename"] == slots["dark"]["filename"], "Keep the logo identical across appearances"
for image in slots.values():
    assert (root / image["filename"]).is_file()
ICON
