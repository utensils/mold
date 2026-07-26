#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/qualify-cuda-sm86.sh"
schema="$repo_root/docs/qualification/cuda-sm86-report.schema.json"

bash -n "$runner"
help_text="$("$runner" --help)"
grep -Fq 'sm86/PTX-JIT regression' <<<"$help_text"
if grep -Fq 'sm89' <<<"$help_text"; then
  echo "qualification help still claims sm89 can run on RTX 3090" >&2
  exit 1
fi
jq -e '
  .properties.hardware_qualified.type == "boolean"
  and (.properties.tests.required | index("sm86_attention_image_smoke")) != null
  and (.properties.tests.required | index("sm86_ptx_image_smoke")) != null
  and (.properties.tests.required | index("sm86_video_smoke")) != null
  and (.properties.tests.required | index("sm86_chained_video_smoke")) != null
  and .properties.host.properties.devices.minItems == 1
  and .properties.artifacts.required == ["sm86"]
  and (.allOf | length) > 0
  and (."$defs".test_result.allOf | length) > 0
' "$schema" >/dev/null

test_root="$(mktemp -d)"
trap 'rm -rf "$test_root"' EXIT
mkdir -p "$test_root/bin"
cat >"$test_root/bin/nvidia-smi" <<'EOF'
#!/bin/sh
echo 'GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, NVIDIA GeForce RTX 3090, 8.6, 999.0'
echo 'GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb, NVIDIA GeForce RTX 3090, 8.6, 999.0'
EOF
cat >"$test_root/fake-mold" <<'EOF'
#!/bin/sh
output=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf 'not a png, not CUDA\n' >"$output"
printf 'Using CUDA device 0\n'
printf 'attention backend selected backend=math\n'
exit 0
EOF
chmod +x "$test_root/bin/nvidia-smi" "$test_root/fake-mold"
printf 'schema = "mold.chain.v1"\n' >"$test_root/chain.toml"
report="$test_root/report.json"
if output="$(
  PATH="$test_root/bin:$PATH" "$runner" \
    --release-tag v0.20.2 \
    --sm86-binary "$test_root/fake-mold" \
    --sm89-binary "$test_root/fake-mold" \
    --image-model flux-fixture \
    --video-model ltx-video-fixture \
    --chain-script "$test_root/chain.toml" \
    --report "$report" 2>&1
)"; then
  echo "qualification runner accepted the exact fake-attestation exploit" >&2
  exit 1
fi
grep -Fq 'unknown argument: --sm89-binary' <<<"$output" \
  || { echo "runner still accepts the impossible sm89-on-3090 qualification input" >&2; exit 1; }
[[ ! -e "$report" ]] \
  || { echo "runner wrote a report after rejecting fake artifact identity" >&2; exit 1; }

printf 'not a png, not CUDA\n' >"$test_root/not-a-png"
if "$repo_root/scripts/verify-png-artifact.py" \
  "$test_root/not-a-png" 256 256 >/dev/null 2>&1; then
  echo "PNG verifier accepted a text artifact" >&2
  exit 1
fi

incomplete="$test_root/incomplete-true.json"
cat >"$incomplete" <<'EOF'
{
  "schema_version": "mold.cuda.sm86.qualification.v3",
  "source_sha": "0000000000000000000000000000000000000000",
  "release_tag": "v0.20.2",
  "hardware_qualified": true,
  "provenance": {"official_release_manifest_verified": false},
  "host": {"devices": []},
  "artifacts": {"sm86": {}},
  "tests": {}
}
EOF
if "$repo_root/scripts/validate-cuda-qualification-report.py" "$incomplete" >/dev/null 2>&1; then
  echo "relational validator accepted an incomplete hardware_qualified=true report" >&2
  exit 1
fi

if find "$repo_root/docs/qualification" -maxdepth 1 -type f \
  -name 'cuda-sm86-qualification*.json' -print -quit | grep -q .; then
  echo "a qualification result was checked in without an authorized hardware run" >&2
  exit 1
fi

echo "CUDA qualification contract: ok"
