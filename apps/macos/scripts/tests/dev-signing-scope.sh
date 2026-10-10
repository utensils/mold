#!/usr/bin/env bash
# Keep developer signing off SwiftPM resource bundles without losing app identity.
set -euo pipefail
cd "$(dirname "$0")/../.."
identity='Developer ID Application: Fixture (TESTTEAM)'
for config in Debug Release; do
  command=$(make -n build CONFIG="$config" MOLD_DEV_SIGN_IDENTITY="$identity")
  if [[ "$command" == *' CODE_SIGN_IDENTITY='* ]]; then
    echo 'FAIL: global CODE_SIGN_IDENTITY leaks into SwiftPM targets' >&2
    exit 1
  fi
  expected="$identity"
  [[ "$config" != Release ]] || expected=-
  [[ "$command" == *"MOLD_APP_SIGN_IDENTITY=\"$expected\""* ]] || {
    echo "FAIL: $config lost intended app identity" >&2; exit 1;
  }
done
command=$(make -n build MOLD_DEV_SIGN_IDENTITY=)
[[ "$command" == *'MOLD_APP_SIGN_IDENTITY="-"'* ]] || {
  echo 'FAIL: no identity must retain ad hoc fallback' >&2; exit 1;
}
python3 - <<'PY'
from pathlib import Path
spec = Path('project.yml').read_text()
app = spec.split('  Mold:\n', 1)[1].split('  MoldTests:\n', 1)[0]
assert 'CODE_SIGN_IDENTITY: "$(MOLD_APP_SIGN_IDENTITY)"' in app, 'app must consume its scoped identity'
assert 'MOLD_APP_SIGN_IDENTITY: "-"' in spec, 'direct Xcode builds need ad hoc fallback'
PY
echo 'developer signing scope ok'
