#!/usr/bin/env bash
# The prune must survive having nothing to prune, and must never touch a
# sibling's asset.
#
# **Fails today**: the prune was a `grep | grep -vx | sort -V | head -n -9`
# pipeline inline in the workflow. On the FIRST nightly publish the only
# matching asset is the one just uploaded, so `grep -vx` matches nothing and
# exits 1; `pipefail` propagates it and `errexit` kills the step -- AFTER the
# pointer flip, so the publish succeeded and the run went red anyway. And
# `^Mold-.*\.dmg$` would have matched a future desktop or CLI asset on the
# `latest` release the two apps share.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PRUNE="$HERE/prune-native-nightly-assets.sh"

run() {
  printf '%s' "$1" | "$PRUNE" --repository utensils/mold --current-asset "$2" --keep "${3:-10}" --list-only
}

only_current='{"assets":[{"name":"Mold-native-0.1.0-nightly.100.dmg"}]}'
out=$(run "$only_current" "Mold-native-0.1.0-nightly.100.dmg") \
  || { echo "FAIL: nothing to prune must not fail the job" >&2; exit 1; }
if [[ -n "$out" ]]; then
  echo "FAIL: it wanted to delete something on a one-asset release: $out" >&2
  exit 1
fi

# A sibling's assets on the SAME release are not ours to delete.
siblings='{"assets":[
  {"name":"Mold-native-0.1.0-nightly.100.dmg"},
  {"name":"Mold_0.29.0_aarch64.dmg"},
  {"name":"Mold-macos-arm64.dmg"},
  {"name":"mold-desktop-nightly.json"},
  {"name":"mold-native-appcast-nightly.xml"}
]}'
out=$(run "$siblings" "Mold-native-0.1.0-nightly.100.dmg" 1)
if [[ -n "$out" ]]; then
  echo "FAIL: it selected an asset that is not a native nightly DMG: $out" >&2
  exit 1
fi

# And it really does prune: eleven generations, keep 3, the two newest older
# ones survive beside the current one.
many='{"assets":['
for n in 101 102 103 104 105 106 107 108 109 110 111; do
  many+="{\"name\":\"Mold-native-0.1.0-nightly.$n.dmg\"},"
done
many="${many%,}]}"
out=$(run "$many" "Mold-native-0.1.0-nightly.111.dmg" 3)
expected=$'Mold-native-0.1.0-nightly.108.dmg\nMold-native-0.1.0-nightly.107.dmg\nMold-native-0.1.0-nightly.106.dmg\nMold-native-0.1.0-nightly.105.dmg\nMold-native-0.1.0-nightly.104.dmg\nMold-native-0.1.0-nightly.103.dmg\nMold-native-0.1.0-nightly.102.dmg\nMold-native-0.1.0-nightly.101.dmg'
if [[ "$out" != "$expected" ]]; then
  echo "FAIL: wrong generations pruned" >&2
  echo "got:      $out" >&2
  echo "expected: $expected" >&2
  exit 1
fi
# The two newest older ones, and the current one, must NOT be in the list.
for keeper in 110 109 111; do
  if printf '%s\n' "$out" | grep -q "nightly.$keeper.dmg"; then
    echo "FAIL: it pruned a generation it was told to keep: $keeper" >&2
    exit 1
  fi
done

echo "native nightly prune ok"
