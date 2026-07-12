#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/prune-desktop-nightly-assets.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

mock_bin="$tmp/bin"
mkdir -p "$mock_bin"
export MOCK_GH_ASSETS="$tmp/assets.json"
export MOCK_GH_DELETIONS="$tmp/deletions.txt"

cat > "$mock_bin/gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "release view" ]]; then
  cat "$MOCK_GH_ASSETS"
elif [[ "$1 $2" == "release delete-asset" ]]; then
  printf '%s\n' "$4" >> "$MOCK_GH_DELETIONS"
else
  echo "unexpected gh invocation: $*" >&2
  exit 1
fi
EOF
chmod +x "$mock_bin/gh"

current="0.16.1-nightly.112"
jq -n --arg current "$current" '
  def assets($version; $created):
    [
      {name: "Mold_\($version)_aarch64.app.tar.gz", createdAt: $created},
      {name: "Mold_\($version)_aarch64.app.tar.gz.sig", createdAt: $created},
      {name: "Mold_\($version)_aarch64.dmg", createdAt: $created}
    ];
  {
    assets:
      ([range(101; 113)]
        | map(. as $run
          | "0.16.1-nightly.\($run)" as $version
          | assets($version; "2026-07-12T\((($run - 100) | tostring | if length == 1 then "0" + . else . end)):00:00Z"))
        | add)
      + [
        {name: "mold-desktop-nightly.json", createdAt: "2026-07-12T12:01:00Z"},
        {name: "mold-aarch64-apple-darwin.tar.gz", createdAt: "2026-07-12T12:01:00Z"},
        {name: "Mold_0.16.1_aarch64.dmg", createdAt: "2026-07-12T12:01:00Z"}
      ]
  }
' > "$MOCK_GH_ASSETS"

PATH="$mock_bin:$PATH" "$script" \
  --repository utensils/mold \
  --release-tag latest \
  --current-version "$current" \
  --keep 3 > /dev/null

[[ -f "$MOCK_GH_DELETIONS" ]] || fail "no old assets were pruned"
[[ "$(wc -l < "$MOCK_GH_DELETIONS" | tr -d ' ')" == 27 ]] \
  || fail "expected 27 assets from nine old builds to be pruned"

if grep -Fq "$current" "$MOCK_GH_DELETIONS"; then
  fail "current manifest target was pruned"
fi
if grep -Fq 'nightly.111' "$MOCK_GH_DELETIONS"; then
  fail "newest retained build was pruned"
fi
if grep -Fq 'nightly.110' "$MOCK_GH_DELETIONS"; then
  fail "second retained build was pruned"
fi
grep -Fq 'nightly.109' "$MOCK_GH_DELETIONS" \
  || fail "oldest non-retained build was not pruned"
if grep -Fq 'mold-desktop-nightly.json' "$MOCK_GH_DELETIONS" \
  || grep -Fq 'mold-aarch64-apple-darwin.tar.gz' "$MOCK_GH_DELETIONS" \
  || grep -Fq 'Mold_0.16.1_aarch64.dmg' "$MOCK_GH_DELETIONS"; then
  fail "an unrelated release asset was pruned"
fi

if PATH="$mock_bin:$PATH" "$script" \
  --repository utensils/mold \
  --release-tag latest \
  --current-version "$current" \
  --keep 0 > /dev/null 2>&1; then
  fail "zero retention count was accepted"
fi

echo "PASS: prune-desktop-nightly-assets"
