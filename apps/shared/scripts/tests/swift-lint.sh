#!/usr/bin/env bash
# swift-lint.sh must FAIL on each violation it exists to catch, and pass on
# clean code. A lint that silently stopped matching would let both native
# apps drift while every run said "ok".
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
lint="$here/swift-lint.sh"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

fail() { echo "FAIL: $*"; exit 1; }
expect_fail() { if "$lint" "$@" >/dev/null 2>&1; then fail "$* passed a violation"; fi; }
expect_pass() { "$lint" "$@" >/dev/null 2>&1 || fail "$* rejected clean code"; }

mkdir -p "$scratch/clean" "$scratch/color" "$scratch/uicolor" "$scratch/nscolor" "$scratch/a11y" \
  "$scratch/hidden" "$scratch/optout" "$scratch/big"
cat > "$scratch/clean/Good.swift" <<'EOF'
import SwiftUI
struct Good: View {
    var body: some View { Label("Queue", systemImage: "list.bullet.indent").foregroundStyle(.tint) }
}
EOF
echo 'let c = Color(red: 1, green: 0, blue: 0)' > "$scratch/color/Bad.swift"
echo 'let c = UIColor(red: 1, green: 0, blue: 0, alpha: 1)' > "$scratch/uicolor/Bad.swift"
echo 'let c = NSColor(red: 1, green: 0, blue: 0, alpha: 1)' > "$scratch/nscolor/Bad.swift"
echo 'let i = Image(systemName: "star")' > "$scratch/a11y/Bad.swift"
# Hiding a glyph is not a reason on its own: the file must say why.
printf 'let i = Image(systemName: "star").accessibilityHidden(true)\n' > "$scratch/hidden/Bad.swift"
printf '// a11y: decorative\nlet i = Image(systemName: "star").accessibilityHidden(true)\n' > "$scratch/optout/Good.swift"
for n in $(seq 1 160); do echo "// line $n"; done > "$scratch/big/Big.swift"

expect_pass color "$scratch/clean"
expect_fail color "$scratch/color"
expect_fail color "$scratch/uicolor"
expect_pass a11y "$scratch/clean"
expect_fail a11y "$scratch/a11y"
expect_fail color "$scratch/nscolor"
expect_fail a11y "$scratch/hidden"
expect_pass a11y "$scratch/optout"
"$lint" size "$scratch/big" | grep -q "Big.swift (160 lines)" || fail "size missed a 160-line file"
[ -z "$("$lint" size "$scratch/clean")" ] || fail "size flagged a small file"
TYPE_MAX=100 "$lint" type-size "$scratch/big" | grep -q "large type: Big" || fail "type-size missed an oversized type"
expect_fail color "$scratch/does-not-exist"
expect_fail nonsense "$scratch/clean"

echo "  swift-lint tests ok"
