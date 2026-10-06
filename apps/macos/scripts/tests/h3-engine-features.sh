#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
python3 - "$root" <<'PY'
from pathlib import Path
import sys, tomllib
root = Path(sys.argv[1])
manifest = tomllib.loads((root / 'apps/macos/rust/mold-macos-ffi/Cargo.toml').read_text())
expected = {'metal', 'mold-server/h3', 'mold-server/mesh-texture', 'mold-server/mesh-matting', 'mold-server/mesh-delight'}
assert set(manifest['features'].get('shipping-metal', [])) == expected, 'native shipping Metal graph must include H3 and reviewed mesh edges'
assert '--features shipping-metal' in (root / 'apps/macos/Makefile').read_text(), 'native distribution must build shipping-metal'
PY
