#!/usr/bin/env bash
# Contract for scripts/ci/affected-packages.py, the diff-scoped Rust CI selector.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
script="$repo_root/scripts/ci/affected-packages.py"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

decide() {
  printf '%s\n' "$@" | python3 "$script" --paths-from -
}

expect() {
  local label="$1" key="$2" want="$3"; shift 3
  local got
  got="$(decide "$@" | sed -n "s/^${key}=//p")"
  [[ "$got" == "$want" ]] || fail "$label: ${key} was '${got}', wanted '${want}'"
}

# A leaf crate pulls in only itself and its dependents.
expect "tui-only change" scope packages "crates/mold-tui/src/app.rs"
expect "tui-only change" names "mold-ai mold-ai-tui" "crates/mold-tui/src/app.rs"

# A discord change reaches only the binary that embeds the bot.
expect "discord-only change" names "mold-ai mold-ai-discord" "crates/mold-discord/src/lib.rs"

# The core crate fans out to every crate that depends on it (candle does not),
# and the answer is the precise package list, not a blanket workspace.
core_names="$(decide "crates/mold-core/src/types.rs" | sed -n 's/^names=//p')"
for must in mold-ai mold-ai-core mold-ai-inference mold-ai-server mold-ai-tui mold-ai-discord; do
  grep -qw "$must" <<< "$core_names" || fail "core change does not reach $must (got: $core_names)"
done

# Cross-crate include_str! pins are edges Cargo cannot see.
expect "cli change reaches server via include_str" names "mold-ai mold-ai-server" "crates/mold-cli/src/commands/generate.rs"

# Anything that changes how everything builds or is tested is the workspace.
for path in Cargo.lock Cargo.toml .cargo/config.toml .config/nextest.toml \
  .github/workflows/ci.yml crates/mold-server/build.rs \
  crates/mold-server/build_support/h3_server_features.rs vendor/xatlas/xatlas.cpp \
  tests/fixtures/hunyuan3d/transformer21.safetensors scripts/ci/affected-packages.py; do
  expect "workspace-wide path $path" scope workspace "$path"
done

# The feature union only names selected packages: cargo refuses `pkg/feature`
# for a package outside the selection.
expect "tui-only features" features "mold-ai/discord,mold-ai/expand,mold-ai/mdns,mold-ai/metrics,mold-ai/mp4,mold-ai/preview,mold-ai/pulid,mold-ai/tui,mold-ai/webp" "crates/mold-tui/src/app.rs"
candle_features="$(decide crates/mold-candle/src/lib.rs | sed -n 's/^features=//p')"
grep -q "mold-ai-inference/pulid" <<< "$candle_features" || fail "a candle change does not reach the inference union"
grep -q "mold-ai-core/" <<< "$candle_features" && fail "a candle change must not name core features (core does not depend on candle)"
expect "docs-only features" features "" "README.md"
grep -q "mold-ai-inference/h3-private-uat" <<< "$(decide crates/mold-inference/src/x.rs | sed -n 's/^features=//p')" \
  || fail "an inference change does not turn on the private H3 record runtime"
grep -q "mold-ai-inference/h3," <<< "$(decide Cargo.lock | sed -n 's/^features=//p')" \
  && fail "the PR union must not enable mold-ai-inference/h3 (it flips the record runtime off)"

# `--all` is what a push to main uses: the workspace and the whole union.
[[ "$(python3 "$script" --all | sed -n 's/^packages=//p')" == "--workspace" ]] || fail "--all is not the workspace"
grep -q "mold-ai-server/mdns" <<< "$(python3 "$script" --all | sed -n 's/^features=//p')" || fail "--all omits the server union"

# Non-Rust paths touch nothing.
expect "docs-only change" scope none "website/guide/prompting.md" "changelog.d/x.md" "studio/lib/a.ts"
expect "docs-only change" packages "" "README.md"

# Mixed: a leaf crate plus docs is still just the leaf.
expect "leaf plus docs" names "mold-ai mold-ai-tui" "crates/mold-tui/src/app.rs" "README.md"

# Every cross-crate include_str! in the tree must be a listed edge.
while IFS=: read -r file _ line; do
  case "$file" in crates/*/src/*) ;; *) continue ;; esac
  including_dir="${file%%/src/*}"
  target="$(sed -E 's/.*include_(str|bytes)!\("([^"]+)".*/\2/' <<< "$line")"
  resolved="$(cd "$repo_root/$(dirname "$file")" && python3 -c "import os,sys; print(os.path.normpath(os.path.join(os.getcwd(), sys.argv[1])))" "$target")"
  rel="${resolved#"$repo_root"/}"
  case "$rel" in
    crates/*) included_dir="$(cut -d/ -f1-2 <<< "$rel")" ;;
    tests/fixtures/*) continue ;;
    *) continue ;;
  esac
  [[ "$included_dir" == "$including_dir" ]] && continue
  grep -Fq "\"$included_dir\"" "$script" \
    || fail "cross-crate include in $file reaches $included_dir, which EXTRA_EDGES does not list"
done < <(grep -rnE 'include_(str|bytes)!\("\.\./\.\./' "$repo_root/crates" --include='*.rs' | sed "s#^$repo_root/##")

echo "PASS: affected-packages contract"
