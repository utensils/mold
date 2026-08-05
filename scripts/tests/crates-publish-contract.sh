#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
publisher="$repo_root/scripts/release/publish-crates.sh"
workflow="$repo_root/.github/workflows/release.yml"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }

"$publisher" --list > "$tmp/publish-order"

# Every Mold workspace crate must appear exactly once in the publisher.
for manifest in "$repo_root"/crates/*/Cargo.toml; do
  sed -n 's/^name = "\(mold-ai[^" ]*\)"/\1/p' "$manifest" | head -1
done | sort > "$tmp/workspace-packages"
sort "$tmp/publish-order" > "$tmp/published-packages"
diff -u "$tmp/workspace-packages" "$tmp/published-packages" \
  || fail "publisher does not cover every workspace crate exactly once"

# Every internal dependency must precede its dependent.
position_of() { awk -v package="$1" '$0 == package { print NR; exit }' "$tmp/publish-order"; }
for manifest in "$repo_root"/crates/*/Cargo.toml; do
  package=$(sed -n 's/^name = "\(mold-ai[^" ]*\)"/\1/p' "$manifest" | head -1)
  package_position=$(position_of "$package")
  while IFS= read -r dependency; do
    dependency_position=$(position_of "$dependency")
    [[ -n "$dependency_position" ]] || fail "$package depends on unpublished $dependency"
    ((dependency_position < package_position)) \
      || fail "$dependency must be published before $package"
  done < <(sed -n 's/.*package = "\(mold-ai-[^"]*\)".*/\1/p' "$manifest")
done

# The workflow must use the single guarded publisher rather than drifting into
# a second hand-maintained package list.
grep -q 'run: scripts/release/publish-crates.sh' "$workflow" \
  || fail "release workflow does not invoke the crates publisher"
if sed -n '/^  publish:/,/^  publish-aur:/p' "$workflow" | grep -q 'cargo publish'; then
  fail "release workflow bypasses the guarded crates publisher"
fi

# Exercise partial-release recovery without network access. Pretend core and db
# already exist, then prove they are skipped while every remaining crate is
# published in order and checked for index visibility.
mkdir -p "$tmp/bin"
printf '%s\n' mold-ai-core mold-ai-db > "$tmp/registry-state"
cat > "$tmp/bin/curl" <<'EOF'
#!/usr/bin/env bash
url="${*: -1}"
package="${url%/*}"
package="${package##*/}"
if grep -qx "$package" "$MOCK_REGISTRY_STATE"; then
  printf 200
else
  printf 404
fi
EOF
cat > "$tmp/bin/cargo" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  publish)
    shift
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "-p" ]]; then
        package="$2"
        break
      fi
      shift
    done
    [[ -n "${package:-}" ]]
    printf '%s\n' "$package" >> "$MOCK_REGISTRY_STATE"
    printf '%s\n' "$package" >> "$MOCK_PUBLISH_LOG"
    ;;
  info)
    package_version="$2"
    package="${package_version%@*}"
    grep -qx "$package" "$MOCK_REGISTRY_STATE"
    ;;
  *) exit 1 ;;
esac
EOF
chmod +x "$tmp/bin/curl" "$tmp/bin/cargo"

MOCK_REGISTRY_STATE="$tmp/registry-state" \
MOCK_PUBLISH_LOG="$tmp/publish-log" \
CARGO_REGISTRY_TOKEN=test \
CARGO_INDEX_INTERVAL_SECONDS=0 \
PATH="$tmp/bin:$PATH" \
  "$publisher" > "$tmp/publisher-output"

grep -q '^Already published: mold-ai-core@' "$tmp/publisher-output" \
  || fail "partial release did not skip existing core version"
grep -q '^Already published: mold-ai-db@' "$tmp/publisher-output" \
  || fail "partial release did not skip existing db version"
grep -xvF -f <(printf '%s\n' mold-ai-core mold-ai-db) "$tmp/publish-order" \
  > "$tmp/expected-recovery-publishes"
diff -u "$tmp/expected-recovery-publishes" "$tmp/publish-log" \
  || fail "partial-release recovery published the wrong crates or order"

echo "PASS: crates publish contract"
