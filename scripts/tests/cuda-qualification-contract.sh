#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/qualify-cuda-sm86.sh"
ptx_probe="$repo_root/scripts/probe-cuda-embedded-ptx.py"
schema="$repo_root/docs/qualification/cuda-sm86-report.schema.json"

bash -n "$runner"
[[ -x "$ptx_probe" ]] \
  || { echo "embedded PTX probe is missing or not executable" >&2; exit 1; }
help_text="$("$runner" --help)"
grep -Fq 'exact embedded sm86 PTX module' <<<"$help_text"
if grep -Fq 'CUDA_FORCE_PTX_JIT=1' <<<"$help_text" \
  || grep -Fq 'CUDA_FORCE_PTX_JIT=1' "$runner"; then
  echo "qualification still forces every CUDA library through PTX JIT" >&2
  exit 1
fi
grep -Fq -- '--query-compute-apps=pid,gpu_uuid' "$runner" \
  || { echo "qualification does not bind CUDA observations to an exact process PID" >&2; exit 1; }
grep -Fq 'sm89 artifact is expected to fail on this sm86 hardware' <<<"$help_text" \
  || { echo "qualification help omits the negative sm89 compatibility invariant" >&2; exit 1; }
grep -Fq 'never as a positive smoke' <<<"$help_text" \
  || { echo "qualification help does not reject positive sm89 qualification" >&2; exit 1; }
jq -e '
  .properties.hardware_qualified.type == "boolean"
  and (.properties.tests.required | index("sm86_attention_image_smoke")) != null
  and (.properties.tests.required | index("sm86_ptx_image_smoke")) != null
  and (.properties.tests.required | index("sm86_video_smoke")) != null
  and (.properties.tests.required | index("sm86_chained_video_smoke")) != null
  and .properties.host.properties.devices.minItems == 1
  and .properties.artifacts.required == ["sm86"]
  and .properties.schema_version.const == "mold.cuda.sm86.qualification.v4"
  and (."$defs".test_result.required | index("embedded_ptx_module_loaded")) != null
  and (."$defs".test_result.required | index("embedded_ptx_probe_sha256")) != null
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
  || { echo "positive runner accepts the impossible sm89-on-3090 input" >&2; exit 1; }
[[ ! -e "$report" ]] \
  || { echo "runner wrote a report after rejecting fake artifact identity" >&2; exit 1; }

printf 'not a png, not CUDA\n' >"$test_root/not-a-png"
if "$repo_root/scripts/verify-png-artifact.py" \
  "$test_root/not-a-png" 256 256 >/dev/null 2>&1; then
  echo "PNG verifier accepted a text artifact" >&2
  exit 1
fi

artifact_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
output_sha="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
log_sha="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
probe_path="$test_root/embedded-ptx-probe.json"
jq -n --arg artifact_sha "$artifact_sha" '{
  expected_target: "sm_86",
  artifact_sha256: $artifact_sha,
  loaded: true,
  attempts: [{loaded: true, cuda_result: 0}]
}' >"$probe_path"
probe_sha="$(sha256sum "$probe_path" | awk '{print $1}')"
valid_report="$test_root/valid-report.json"
jq -n \
  --arg artifact_sha "$artifact_sha" \
  --arg output_sha "$output_sha" \
  --arg log_sha "$log_sha" \
  --arg probe_path "$probe_path" \
  --arg probe_sha "$probe_sha" '
  def result($ptx):
    {
      status: "passed",
      exit_code: 0,
      selected_gpu_uuid: "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
      cuda_work_observed: true,
      media_decoded: true,
      width: 256,
      height: 256,
      frame_count: 1,
      output_sha256: $output_sha,
      log_sha256: $log_sha,
      embedded_ptx_module_loaded: $ptx,
      embedded_ptx_probe_path: (if $ptx then $probe_path else "" end),
      embedded_ptx_probe_sha256: (if $ptx then $probe_sha else "" end)
    };
  {
    schema_version: "mold.cuda.sm86.qualification.v4",
    source_sha: "0000000000000000000000000000000000000000",
    release_tag: "v0.20.2",
    hardware_qualified: true,
    provenance: {official_release_manifest_verified: true},
    host: {
      devices: [{
        uuid: "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        name: "NVIDIA GeForce RTX 3090",
        compute_capability: "8.6"
      }]
    },
    artifacts: {
      sm86: {
        cuda_target: "sm_86",
        trusted_checksum_verified: true,
        elf_target_verified: true,
        ptx_target_verified: true,
        source_identity_verified: true,
        expected_sha256: $artifact_sha,
        actual_sha256: $artifact_sha
      }
    },
    tests: {
      sm86_attention_image_smoke: result(false),
      sm86_ptx_image_smoke: result(true),
      sm86_video_smoke: result(false),
      sm86_chained_video_smoke: result(false)
    }
  }' >"$valid_report"
"$repo_root/scripts/validate-cuda-qualification-report.py" \
  "$valid_report" >/dev/null \
  || { echo "relational validator rejected bound PTX evidence" >&2; exit 1; }
forged_report="$test_root/forged-report.json"
jq '.tests.sm86_ptx_image_smoke.embedded_ptx_probe_sha256 =
  "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"' \
  "$valid_report" >"$forged_report"
if "$repo_root/scripts/validate-cuda-qualification-report.py" \
  "$forged_report" >/dev/null 2>&1; then
  echo "relational validator accepted forged PTX evidence" >&2
  exit 1
fi

incomplete="$test_root/incomplete-true.json"
cat >"$incomplete" <<'EOF'
{
  "schema_version": "mold.cuda.sm86.qualification.v4",
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
