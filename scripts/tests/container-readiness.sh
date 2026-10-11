#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
# Cargo resolves all workspace members before the dependency-cache build.
python3 - <<'CHECK'
from pathlib import Path
import tomllib
prefix = Path("Dockerfile").read_text().split("RUN cargo build", 1)[0]
for member in tomllib.loads(Path("Cargo.toml").read_text())["workspace"]["members"]:
    assert f"COPY {member}/Cargo.toml {member}/Cargo.toml" in prefix, f"missing cached manifest: {member}"
    assert f"{member}/src" in prefix, f"missing cached source stub: {member}"
CHECK
source_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
asset="mold-container-digests-$source_sha.json"
mkdir "$scratch/fragments" "$scratch/bin" "$scratch/published"
for target in sm80 sm86 sm89 sm90 sm100 sm120; do
  suffix="-$target"
  [[ "$target" != sm89 ]] || suffix=""
  jq -n --arg target "$target" --arg tag "sha-$source_sha$suffix" \
    --arg digest "sha256:$(printf '%064d' 1)" \
    '{target: $target, tag: $tag, digest: $digest}' > "$scratch/fragments/$target.json"
done
scripts/create-container-digest-manifest.sh "$scratch/$asset" ghcr.io/utensils/mold \
  "sha-$source_sha" "$source_sha" "$scratch"/fragments/*.json
if scripts/create-container-digest-manifest.sh "$scratch/wrong.json" ghcr.io/utensils/mold \
  sha-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb "$source_sha" "$scratch"/fragments/*.json 2>/dev/null; then
  echo 'accepted mismatching source identity' >&2; exit 1
fi
cat > "$scratch/bin/gh" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == api ]]; then printf '%s\n' "$READINESS_MAIN"; exit; fi
operation="$2"
shift 3
if [[ "$operation" == download ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in --pattern) asset="$2";; --dir) dest="$2";; esac
    shift 2
  done
  [[ -f "$READINESS_TEST/published/$asset" ]] || exit 1
  cp "$READINESS_TEST/published/$asset" "$dest/"
else
  [[ "$operation" == upload && $# == 3 && "$2" == --repo ]] || exit 9
  echo upload >> "$READINESS_TEST/calls"
  if [[ -f "$READINESS_TEST/race" ]]; then
    cp "$READINESS_TEST/race" "$READINESS_TEST/published/$(basename "$1")"
    exit 1
  fi
  [[ ! -e "$READINESS_TEST/published/$(basename "$1")" && ! -f "$READINESS_TEST/unavailable" ]] || exit 1
  cp "$1" "$READINESS_TEST/published/"
fi
MOCK
cat > "$scratch/bin/docker" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1 $2" == 'buildx imagetools' ]] || exit 9
if [[ "$3" == create ]]; then
  printf '%s\n' "$*" >> "$READINESS_TEST/docker-calls"
  image="${!#}"
  printf '%s\n' "${image#*@}" > "$READINESS_TEST/digest"
else
  cat "$READINESS_TEST/digest"
fi
MOCK
chmod +x "$scratch/bin/gh" "$scratch/bin/docker"
export PATH="$scratch/bin:$PATH" READINESS_TEST="$scratch" READINESS_MAIN="$source_sha"
scripts/publish-container-readiness.sh utensils/mold "$source_sha" "$scratch/$asset" >/dev/null
cp "$scratch/published/$asset" "$scratch/first.json"
# Same source rebuilt with a different digest must preserve the first pin.
jq '.targets[].digest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"' "$scratch/$asset" > "$scratch/rebuilt.json"
scripts/publish-container-readiness.sh utensils/mold "$source_sha" "$scratch/rebuilt.json" >/dev/null
cmp "$scratch/published/$asset" "$scratch/first.json"
[[ "$(wc -l < "$scratch/calls" | tr -d ' ')" == 1 ]]
# A lost create race preserves the winner only after validation.
rm "$scratch/published/$asset"
cp "$scratch/first.json" "$scratch/race"
scripts/publish-container-readiness.sh utensils/mold "$source_sha" "$scratch/rebuilt.json" >/dev/null
cmp "$scratch/published/$asset" "$scratch/first.json"
jq 'del(.targets.sm80)' "$scratch/first.json" > "$scratch/published/$asset"
if scripts/publish-container-readiness.sh utensils/mold "$source_sha" "$scratch/rebuilt.json" >/dev/null 2>&1; then
  echo 'accepted incomplete existing readiness' >&2; exit 1
fi
rm "$scratch/published/$asset" "$scratch/race"
touch "$scratch/unavailable"
if scripts/publish-container-readiness.sh utensils/mold "$source_sha" "$scratch/rebuilt.json" >/dev/null 2>&1; then
  echo 'mistook publication/API failure for readiness' >&2; exit 1
fi
scripts/promote-container-latest.sh utensils/mold "$source_sha" "$scratch/first.json" >/dev/null
[[ "$(wc -l < "$scratch/docker-calls" | tr -d ' ')" == 6 ]]
rm "$scratch/docker-calls"
READINESS_MAIN=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  scripts/promote-container-latest.sh utensils/mold "$source_sha" "$scratch/first.json" >/dev/null
[[ ! -e "$scratch/docker-calls" ]]
echo 'container readiness contracts: ok'
