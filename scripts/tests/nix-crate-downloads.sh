#!/usr/bin/env bash
# Check real fetch derivations, including Crane's own helper dependencies.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

system=${NIX_CRATE_SYSTEM:-$(nix eval --impure --raw --expr builtins.currentSystem)}
for output in mold.cargoVendorDir mold-desktop.cargoDeps; do
  drv=$(nix eval --raw ".#packages.$system.$output.drvPath")
  nix derivation show --recursive "$drv" > "$tmp/derivations.json"
  python3 - "$tmp/derivations.json" "$output" "$drv" "$tmp/helper.drv" <<'PY'
import json
import sys

from pathlib import Path


def attrs(derivation):
    env = derivation.get("env", {})
    return env | derivation.get("structuredAttrs", json.loads(env.get("__json", "{}")))


document = json.load(open(sys.argv[1]))
derivations = {Path(key).name: value for key, value in document.get("derivations", document).items()}


def inputs(name):
    drv = derivations[name]
    return [Path(key).name for key in drv.get("inputDrvs", drv.get("inputs", {}).get("drvs", {}))]


vendor = Path(sys.argv[3]).name
if sys.argv[2] == "mold.cargoVendorDir":
    helpers = [attrs(drv) for drv in derivations.values()
               if attrs(drv).get("name", "").startswith("crane-utils-")]
    assert len(helpers) == 1, "Expected exactly one Crane helper"
    helper = next(name for name, drv in derivations.items()
                  if attrs(drv).get("out") == helpers[0]["out"])
    Path(sys.argv[4]).write_text("/nix/store/" + helper)
    vendor = next(name for name, drv in derivations.items()
                  if attrs(drv).get("out") == helpers[0]["cargoDeps"])

# importCargoLock links unpacked packages, each depending on its crate archive.
# Inspect those archives, not unrelated tools used to build the vendor directory.
archives = {source for package in inputs(vendor) for source in inputs(package)
            if attrs(derivations[source]).get("name", "").startswith("crate-")}
assert archives, "No vendored crate archives found"
for name in archives:
    urls = attrs(derivations[name]).get("urls", [])
    if isinstance(urls, str):
        urls = urls.split()
    assert urls and all(url.startswith("https://static.crates.io/crates/") for url in urls), (name, urls)
print(f"PASS: {sys.argv[2]}: {len(archives)} vendored archives use static crate downloads")
PY
done

# Fetch URLs alone cannot prove Cargo accepts the generated source config.
# Compile the small helper on native hosts to catch duplicate registry aliases.
if [[ "$system" == "$(nix eval --impure --raw --expr builtins.currentSystem)" ]]; then
  nix build --no-link "$(cat "$tmp/helper.drv")^*"
fi
