#!/usr/bin/env bash
# One Candle package identity across every Mold cargo root.
#
# `candle_core::Tensor` and `candle_core::Error` are NOMINAL types: two copies
# of Candle in one dependency graph are two unrelated types, and every call
# site that hands a tensor from one to the other stops typechecking. #1393
# moved the workspace onto the renamed `candle-*-mold` fork packages (which
# `[patch.crates-io]` cannot unify, because a patch must keep the patched
# package's name) but left `crates/mold-candle`'s `candle-flash-attn` pinned to
# crates.io, whose copy depends on the upstream-named `candle-core`. The CUDA
# build broke for four consecutive `main` merges (#1399) because the only gate
# that compiles `--features flash-attn` is a 90-minute `push`-only job.
#
# This runs on the release-contract route instead: no CUDA, no kernels, and it
# fails on the manifest edit that would cause the break rather than on the
# build an hour later.
#
# Both cargo roots that have no Candle at all today — the mobile app and its
# plugin crate, which are remote-only by construction — are audited too, so
# adding a registry Candle there is a failure rather than a blind spot.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

# Discovered rather than listed: a new cargo root or workspace member must not
# be able to opt out of this contract by being absent from an array here.
mapfile -t manifests < <(git ls-files -- '*Cargo.toml' ':!:tmp/*' | sort)
mapfile -t locks < <(git ls-files -- '*Cargo.lock' ':!:tmp/*' | sort)

test "${#manifests[@]}" -ge 12 ||
  fail "found only ${#manifests[@]} cargo manifests; run this from a full checkout"
test "${#locks[@]}" -ge 3 ||
  fail "found only ${#locks[@]} lockfiles; every cargo root must have one"

# ---------------------------------------------------------------------------
# 0. Every cargo package in the repo is resolved by one of the lockfiles
#    audited below.
#
# The lock audit in step 2 is the only assertion that can see a TRANSITIVE
# Candle dependency, so a package whose graph no tracked lockfile resolves is
# an unauditable hole rather than a passing case. Deliberately NOT keyed on an
# `[workspace]` table: a standalone root that simply omits one would skip the
# check, which is the same enumerate-what-exists-today mistake that let #1399
# through. Every manifest answers, and it answers by NAME — a workspace member
# is resolved by its workspace's lock, `apps/mobile/plugins` by the mobile
# app's lock (it is a path dependency of it), and anything genuinely
# unresolved fails.
# ---------------------------------------------------------------------------
lock_names="$(awk '/^name = /{gsub(/"/, "", $3); print $3}' "${locks[@]}" | sort -u)"

for manifest in "${manifests[@]}"; do
  package="$(sed -nE '/^\[package\]/,/^\[[a-z]/ s/^name = "([^"]+)".*$/\1/p' "$manifest" | head -1)"

  if [ -z "$package" ]; then
    # A virtual manifest (workspace root with no package of its own) has no
    # graph to resolve; anything else with no name is malformed.
    grep -qE '^\[workspace\]' "$manifest" ||
      fail "$manifest declares neither a [package] name nor a [workspace]; this contract cannot tell which lockfile resolves it"
    continue
  fi

  grep -qx "$package" <<<"$lock_names" ||
    fail "no tracked lockfile resolves $manifest ($package appears in none of: ${locks[*]}), so a transitive Candle dependency there would be invisible to this contract; commit its Cargo.lock, or make it a path dependency of a root that has one"
done

# The authority is mold-inference's `candle-core`: `desktop-candle-lock-sync.sh`
# already treats that line as the source of truth for the desktop lockfile.
authority="$(
  sed -nE \
    's|^candle-core = \{ package = "candle-core-mold", git = "([^"]+)", rev = "([0-9a-f]{40})" \}.*$|\1 \2|p' \
    crates/mold-inference/Cargo.toml
)"
test -n "$authority" ||
  fail "could not read the candle fork git url and revision from crates/mold-inference/Cargo.toml"

authority_url="${authority%% *}"
authority_rev="${authority##* }"
# The exact string cargo writes for this dependency. Comparing whole sources —
# rather than just the trailing commit — rejects a second URL, a `branch=`/
# `tag=` query, or any other distinct source that happens to resolve to the
# same commit; cargo treats each of those as its own package identity.
authority_source="git+${authority_url}?rev=${authority_rev}#${authority_rev}"

# ---------------------------------------------------------------------------
# 1. Every declared candle dependency names the same fork revision.
# ---------------------------------------------------------------------------
declarations=0
for manifest in "${manifests[@]}"; do
  while IFS= read -r line; do
    declarations=$((declarations + 1))
    name="${line%% =*}"

    grep -Fq "git = \"$authority_url\"" <<<"$line" ||
      fail "$manifest declares $name from a different source than $authority_url; every candle crate must come from one fork revision or its \`Tensor\` type stops unifying"
    grep -Fq "rev = \"$authority_rev\"" <<<"$line" ||
      fail "$manifest pins $name to a different revision than $authority_rev; every candle crate must come from one fork revision or its \`Tensor\` type stops unifying"
    # `version` is permitted BESIDE git+rev and nowhere else: cargo ignores it
    # while resolving locally (the git source wins) and crates.io requires it
    # of any dependency in a published manifest, so banning it outright would
    # forbid the only shape `cargo package -p mold-ai-candle` can accept.
    # `branch`, `tag`, and `path` are different sources, so they stay banned.
    if grep -Eq '(branch|tag|path) = "[^"]+"' <<<"$line"; then
      fail "$manifest gives $name a branch, tag, or path source; only an exact \`rev\` pin on the fork keeps one package identity"
    fi
  done < <(grep -E '^candle[a-z0-9_-]* = ' "$manifest" || true)
done

# A manifest set that matched nothing would pass every assertion above.
test "$declarations" -ge 13 ||
  fail "found only $declarations candle dependency declarations across ${#manifests[@]} manifests; expected at least the 13 in mold-candle, mold-inference, mold-server, and the desktop root"

# ---------------------------------------------------------------------------
# 2. No lockfile resolves a second candle identity.
#
# This is the assertion that actually holds: a manifest audit cannot see a
# TRANSITIVE crates.io consumer of `candle-core`, which is exactly the shape
# `candle-flash-attn` had. A root with no Candle at all (the mobile app and
# its plugin crate) passes by having nothing to report, not by being skipped.
# ---------------------------------------------------------------------------
for lock in "${locks[@]}"; do
  # A candle package with NO `source` is a path dependency — a vendored or
  # local copy, which is a second identity that carries no revision to compare.
  # `source` always follows `version` for a package that has one, so a name
  # with no source before the next field is the case to reject.
  sources="$(
    awk '
      function flush() {
        if (name ~ /^candle/) print name "\t" (source == "" ? "<path-dependency>" : source)
        name = ""; source = ""
      }
      /^\[\[package\]\]$/ { flush(); next }
      /^name = / { gsub(/"/, "", $3); name = $3; next }
      /^source = / { gsub(/"/, "", $3); source = $3; next }
      END { flush() }
    ' "$lock"
  )"

  test -n "$sources" || {
    # Roots that legitimately contain no Candle: the phone app is remote-only.
    echo "  $lock: no candle packages"
    continue
  }

  while IFS=$'\t' read -r name source; do
    test "$source" = "$authority_source" ||
      fail "$lock resolves $name from \`$source\`, not \`$authority_source\`; a registry, path, or second git source is a distinct package identity and puts two \`Tensor\` types in one graph (#1399)"
  done <<<"$sources"

  echo "  $lock: $(wc -l <<<"$sources") candle packages, all from the pinned fork"
done

# ---------------------------------------------------------------------------
# 3. The H3 qualification record names the payload that is actually compiled.
#
# `H3_FLASH_ATTN_SOURCE` is serialized as the FlashAttention provenance of a
# release-candidate build. It was the crates.io archive checksum until #1399,
# which is precisely the constant that stopped describing anything real when
# the dependency moved. It lives in Rust and the lockfile lives here, so this
# is the one place that can compare them.
# ---------------------------------------------------------------------------
provenance="crates/mold-candle/src/minimax_h3/attention.rs"
test -f "$provenance" || fail "missing $provenance"
recorded="$(
  sed -n '/^pub const H3_FLASH_ATTN_SOURCE/,/);/p' "$provenance" \
    | sed -nE 's/^[[:space:]]*"([^"]*)".*$/\1/p' \
    | tr -d '\n'
)"
test -n "$recorded" ||
  fail "could not read H3_FLASH_ATTN_SOURCE from $provenance"
test "$recorded" = "$authority_source" ||
  fail "$provenance records the H3 FlashAttention payload as \`$recorded\`, but cargo resolves \`$authority_source\`; a qualification record naming a payload that was not compiled is worse than none"

echo "PASS: every candle crate resolves to $authority_source"
