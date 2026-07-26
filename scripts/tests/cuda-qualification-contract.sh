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
grep -Fq -- '--sm86-archive' <<<"$help_text"
grep -Fq -- '--sm89-binary' <<<"$help_text"
grep -Fq -- '--sm89-archive' <<<"$help_text"
grep -Fq 'exact-ELF negative CUDA Driver control' <<<"$help_text"
if grep -Fq 'CUDA_FORCE_PTX_JIT=1' <<<"$help_text" \
  || grep -Fq 'CUDA_FORCE_PTX_JIT=1' "$runner"; then
  echo "qualification still forces every CUDA library through PTX JIT" >&2
  exit 1
fi
grep -Fq -- '--query-compute-apps=pid,gpu_uuid' "$runner" \
  || { echo "qualification does not bind CUDA observations to an exact process PID" >&2; exit 1; }
grep -Fq 'mandatory exact-ELF negative CUDA Driver control' <<<"$help_text" \
  || { echo "qualification help omits the negative sm89 compatibility invariant" >&2; exit 1; }
grep -Fq 'never treated as runtime generation or hardware qualification' <<<"$help_text" \
  || { echo "qualification help does not reject positive sm89 qualification" >&2; exit 1; }
jq -e '
  .properties.hardware_qualified.type == "boolean"
  and (.properties.tests.required | index("sm86_attention_image_smoke")) != null
  and (.properties.tests.required | index("sm86_ptx_image_smoke")) != null
  and (.properties.tests.required | index("sm86_video_smoke")) != null
  and (.properties.tests.required | index("sm86_chained_video_smoke")) != null
  and .properties.host.properties.devices.minItems == 1
  and .properties.artifacts.required == ["sm86", "sm89"]
  and .properties.schema_version.const == "mold.cuda.sm86.qualification.v5"
  and (."$defs".test_result.required | index("embedded_ptx_module_loaded")) != null
  and (."$defs".test_result.required | index("embedded_ptx_probe_sha256")) != null
  and (."$defs".test_result.required | index("compute_observation_path")) != null
  and (."$defs".test_result.required | index("compute_observation_sha256")) != null
  and (.properties.artifacts.required | index("sm89")) != null
  and (.properties.compatibility_controls.required | index("sm89_driver_negative")) != null
  and (.allOf | length) > 0
  and (."$defs".test_result.allOf | length) > 0
' "$schema" >/dev/null

test_root="$(mktemp -d)"
trap 'rm -rf "$test_root"' EXIT
valid_report="$test_root/cuda-sm86-qualification.json"
evidence_dir="${valid_report}.d"
mkdir -p "$evidence_dir" "$test_root/sm86" "$test_root/sm89"

source_sha="0123456789abcdef0123456789abcdef01234567"
source_short="${source_sha:0:7}"
uuid="GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
printf 'sm86 exact binary %s\n' "$source_short" >"$test_root/sm86/mold"
printf 'sm89 exact binary %s\n' "$source_short" >"$test_root/sm89/mold"
sm86_binary="$test_root/sm86/mold"
sm89_binary="$test_root/sm89/mold"
sm86_archive="$test_root/mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz"
sm89_archive="$test_root/mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz"
tar -czf "$sm86_archive" -C "$test_root/sm86" mold
tar -czf "$sm89_archive" -C "$test_root/sm89" mold
sm86_binary_sha="$(sha256sum "$sm86_binary" | awk '{print $1}')"
sm89_binary_sha="$(sha256sum "$sm89_binary" | awk '{print $1}')"
sm86_archive_sha="$(sha256sum "$sm86_archive" | awk '{print $1}')"
sm89_archive_sha="$(sha256sum "$sm89_archive" | awk '{print $1}')"

provenance_path="$evidence_dir/mold-release-provenance.json"
jq -n \
  --arg source "$source_sha" \
  --arg sm86_archive "$sm86_archive_sha" \
  --arg sm86_binary "$sm86_binary_sha" \
  --arg sm89_archive "$sm89_archive_sha" \
  --arg sm89_binary "$sm89_binary_sha" '{
    schema_version: "mold.release.provenance.v1",
    release_tag: "v0.20.2",
    source_sha: $source,
    artifacts: {
      sm86: {
        filename: "mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz",
        archive_sha256: $sm86_archive,
        binary_sha256: $sm86_binary,
        cuda_target: "sm_86"
      },
      sm89: {
        filename: "mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz",
        archive_sha256: $sm89_archive,
        binary_sha256: $sm89_binary,
        cuda_target: "sm_89"
      }
    }
  }' >"$provenance_path"
provenance_sha="$(sha256sum "$provenance_path" | awk '{print $1}')"
checksums_path="$evidence_dir/SHA256SUMS"
printf '%s  %s\n' \
  "$sm86_archive_sha" "mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz" \
  "$sm89_archive_sha" "mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz" \
  "$provenance_sha" "mold-release-provenance.json" >"$checksums_path"
checksums_sha="$(sha256sum "$checksums_path" | awk '{print $1}')"

tests_json='{}'
for label in \
  sm86_attention_image_smoke \
  sm86_ptx_image_smoke \
  sm86_video_smoke \
  sm86_chained_video_smoke; do
  case "$label" in
    sm86_attention_image_smoke|sm86_ptx_image_smoke) extension=".png" ;;
    *) extension=".mp4" ;;
  esac
  output_path="$evidence_dir/${label}${extension}"
  log_path="$evidence_dir/${label}.log"
  observation_path="$evidence_dir/${label}-compute-observations.csv"
  printf 'decoded fixture %s\n' "$label" >"$output_path"
  printf '%s\n' \
    '2026-07-26T00:00:00Z  INFO mold_inference::attention: attention backend selected backend=Math' \
    >"$log_path"
  printf '%s\n' \
    'polled_at_utc,generation_root_pid,observed_pid,gpu_uuid' \
    "2026-07-26T00:00:00Z,100,200,$uuid" >"$observation_path"
  output_sha="$(sha256sum "$output_path" | awk '{print $1}')"
  log_sha="$(sha256sum "$log_path" | awk '{print $1}')"
  observation_sha="$(sha256sum "$observation_path" | awk '{print $1}')"
  ptx_loaded=false
  ptx_path=""
  ptx_sha=""
  if [[ "$label" == sm86_ptx_image_smoke ]]; then
    ptx_loaded=true
    ptx_path="$evidence_dir/${label}-embedded-ptx.json"
    jq -n --arg artifact_sha "$sm86_binary_sha" '{
      expected_target: "sm_86",
      observed_targets: ["sm_86"],
      malformed_modules: [],
      incomplete_modules: [],
      artifact_sha256: $artifact_sha,
      loaded: true,
      attempts: [{loaded: true, cuda_result: 0}]
    }' >"$ptx_path"
    ptx_sha="$(sha256sum "$ptx_path" | awk '{print $1}')"
  fi
  result="$(
    jq -n \
      --arg command "fixture $label" \
      --arg output_path "$output_path" \
      --arg output_sha "$output_sha" \
      --arg log_path "$log_path" \
      --arg log_sha "$log_sha" \
      --arg observation_path "$observation_path" \
      --arg observation_sha "$observation_sha" \
      --arg uuid "$uuid" \
      --argjson ptx_loaded "$ptx_loaded" \
      --arg ptx_path "$ptx_path" \
      --arg ptx_sha "$ptx_sha" '{
        status: "passed",
        exit_code: 0,
        command: $command,
        output_path: $output_path,
        output_sha256: $output_sha,
        log_path: $log_path,
        log_sha256: $log_sha,
        compute_observation_path: $observation_path,
        compute_observation_sha256: $observation_sha,
        selected_gpu_uuid: $uuid,
        generation_root_pid: 100,
        observed_compute_pid: 200,
        cuda_work_observed: true,
        cuda_work_evidence: "raw PID and GPU observation",
        media_decoded: true,
        width: 256,
        height: 256,
        frame_count: 1,
        embedded_ptx_module_loaded: $ptx_loaded,
        embedded_ptx_probe_path: $ptx_path,
        embedded_ptx_probe_sha256: $ptx_sha
      }'
  )"
  tests_json="$(jq -n --argjson current "$tests_json" --arg key "$label" \
    --argjson result "$result" '$current + {($key): $result}')"
done

negative_command_path="$evidence_dir/sm89-driver-negative.command"
negative_command="probe exact sm89 ELF on $uuid"
printf '%s\n' "$negative_command" >"$negative_command_path"
negative_command_sha="$(sha256sum "$negative_command_path" | awk '{print $1}')"
negative_probe_path="$evidence_dir/sm89-driver-negative.json"
jq -n --arg artifact_sha "$sm89_binary_sha" '{
  expected_target: "sm_89",
  observed_targets: ["sm_89"],
  malformed_modules: [],
  incomplete_modules: [],
  artifact_sha256: $artifact_sha,
  expect_incompatible: true,
  loaded: false,
  attempts: [{
    loaded: false,
    cuda_result: 218,
    cuda_error_name: "CUDA_ERROR_INVALID_PTX"
  }]
}' >"$negative_probe_path"
negative_probe_sha="$(sha256sum "$negative_probe_path" | awk '{print $1}')"

jq -n \
  --arg source_sha "$source_sha" \
  --arg provenance_path "$provenance_path" \
  --arg provenance_sha "$provenance_sha" \
  --arg checksums_path "$checksums_path" \
  --arg checksums_sha "$checksums_sha" \
  --arg uuid "$uuid" \
  --arg sm86_binary "$sm86_binary" \
  --arg sm86_archive "$sm86_archive" \
  --arg sm86_binary_sha "$sm86_binary_sha" \
  --arg sm86_archive_sha "$sm86_archive_sha" \
  --arg sm89_binary "$sm89_binary" \
  --arg sm89_archive "$sm89_archive" \
  --arg sm89_binary_sha "$sm89_binary_sha" \
  --arg sm89_archive_sha "$sm89_archive_sha" \
  --arg source_short "$source_short" \
  --argjson tests "$tests_json" \
  --arg negative_command "$negative_command" \
  --arg negative_command_path "$negative_command_path" \
  --arg negative_command_sha "$negative_command_sha" \
  --arg negative_probe_path "$negative_probe_path" \
  --arg negative_probe_sha "$negative_probe_sha" '{
    schema_version: "mold.cuda.sm86.qualification.v5",
    source_sha: $source_sha,
    release_tag: "v0.20.2",
    started_at: "2026-07-26T00:00:00Z",
    finished_at: "2026-07-26T00:01:00Z",
    hardware_qualified: true,
    provenance: {
      official_release_manifest_url: "https://github.com/utensils/mold/releases/download/v0.20.2/mold-release-provenance.json",
      official_release_manifest_path: $provenance_path,
      official_release_manifest_sha256: $provenance_sha,
      official_release_manifest_verified: true,
      checksum_manifest_url: "https://github.com/utensils/mold/releases/download/v0.20.2/SHA256SUMS",
      checksum_manifest_path: $checksums_path,
      checksum_manifest_sha256: $checksums_sha
    },
    host: {
      hostname: "fixture-host",
      devices: [{
        uuid: $uuid,
        name: "NVIDIA GeForce RTX 3090",
        compute_capability: "8.6",
        driver_version: "999.0"
      }]
    },
    artifacts: {
      sm86: {
        path: $sm86_binary,
        archive_path: $sm86_archive,
        expected_sha256: $sm86_binary_sha,
        actual_sha256: $sm86_binary_sha,
        expected_archive_sha256: $sm86_archive_sha,
        actual_archive_sha256: $sm86_archive_sha,
        cuda_target: "sm_86",
        trusted_checksum_verified: true,
        elf_target_verified: true,
        ptx_target_verified: true,
        source_identity_verified: true,
        version_output: ("mold fixture " + $source_short)
      },
      sm89: {
        path: $sm89_binary,
        archive_path: $sm89_archive,
        expected_sha256: $sm89_binary_sha,
        actual_sha256: $sm89_binary_sha,
        expected_archive_sha256: $sm89_archive_sha,
        actual_archive_sha256: $sm89_archive_sha,
        cuda_target: "sm_89",
        trusted_checksum_verified: true,
        elf_target_verified: true,
        ptx_target_verified: true,
        source_identity_verified: true,
        version_output: ("mold fixture " + $source_short)
      }
    },
    workloads: {
      image_model: "flux-fixture",
      video_model: "ltx-video-fixture",
      chain_script: "/fixture/chain.toml"
    },
    tests: $tests,
    compatibility_controls: {
      sm89_driver_negative: {
        status: "passed",
        exit_code: 0,
        command: $negative_command,
        command_path: $negative_command_path,
        command_sha256: $negative_command_sha,
        selected_gpu_uuid: $uuid,
        probe_path: $negative_probe_path,
        probe_sha256: $negative_probe_sha,
        expected_incompatibility: true
      }
    }
  }' >"$valid_report"

validator="$repo_root/scripts/validate-cuda-qualification-report.py"
"$validator" "$valid_report" >/dev/null \
  || { echo "validator rejected fully bound qualification evidence" >&2; exit 1; }

report_backup="$test_root/report.backup"
cp "$valid_report" "$report_backup"
expect_report_rejection() {
  local label="$1"
  local filter="$2"
  jq "$filter" "$report_backup" >"$valid_report"
  if "$validator" "$valid_report" >/dev/null 2>&1; then
    echo "validator accepted forged report: $label" >&2
    exit 1
  fi
  cp "$report_backup" "$valid_report"
}
expect_report_rejection "missing schema-required field" 'del(.tests.sm86_video_smoke.log_path)'
expect_report_rejection "unexpected schema field" '.forged = true'
expect_report_rejection "wrong evidence path" \
  '.tests.sm86_attention_image_smoke.log_path = "/tmp/wrong.log"'
expect_report_rejection "forged artifact hash" \
  '.artifacts.sm86.actual_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"'
expect_report_rejection "forged output hash" \
  '.tests.sm86_attention_image_smoke.output_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"'
expect_report_rejection "forged observation relationship" \
  '.tests.sm86_attention_image_smoke.observed_compute_pid = 201'
expect_report_rejection "forged negative control" \
  '.compatibility_controls.sm89_driver_negative.expected_incompatibility = false'

drift_schema="$test_root/schema-with-unsupported-keyword.json"
jq '.properties.schema_version.unsupportedFutureKeyword = true' \
  "$schema" >"$drift_schema"
if "$validator" "$valid_report" --schema "$drift_schema" \
  >"$test_root/schema-drift.stdout" 2>"$test_root/schema-drift.stderr"; then
  echo "validator ignored unsupported schema keyword drift" >&2
  exit 1
fi
grep -Fq 'unsupported schema keyword' "$test_root/schema-drift.stderr" \
  || { echo "schema drift failure did not identify unsupported keyword" >&2; exit 1; }

mutate_and_reject() {
  local label="$1"
  local evidence_path="$2"
  local backup="$test_root/mutation.backup"
  cp "$evidence_path" "$backup"
  printf 'mutation\n' >>"$evidence_path"
  if "$validator" "$valid_report" >/dev/null 2>&1; then
    echo "validator accepted mutated evidence: $label" >&2
    exit 1
  fi
  cp "$backup" "$evidence_path"
}
mutate_and_reject "binary" "$sm86_binary"
mutate_and_reject "archive" "$sm86_archive"
mutate_and_reject "output" "$evidence_dir/sm86_attention_image_smoke.png"
mutate_and_reject "log" "$evidence_dir/sm86_attention_image_smoke.log"
mutate_and_reject "PID/GPU observation" \
  "$evidence_dir/sm86_attention_image_smoke-compute-observations.csv"
mutate_and_reject "PTX probe" \
  "$evidence_dir/sm86_ptx_image_smoke-embedded-ptx.json"
mutate_and_reject "sm89 negative probe" "$negative_probe_path"
mutate_and_reject "release provenance" "$provenance_path"
mutate_and_reject "checksum manifest" "$checksums_path"

math_log="$evidence_dir/sm86_attention_image_smoke.log"
cp "$math_log" "$test_root/math-log.backup"
printf '%s\n' 'unrelated message mentioning math but no backend selection' >"$math_log"
forged_math_sha="$(sha256sum "$math_log" | awk '{print $1}')"
jq --arg sha "$forged_math_sha" \
  '.tests.sm86_attention_image_smoke.log_sha256 = $sha' \
  "$report_backup" >"$valid_report"
if "$validator" "$valid_report" >/dev/null 2>&1; then
  echo "validator accepted arbitrary math text as backend selection" >&2
  exit 1
fi
cp "$test_root/math-log.backup" "$math_log"
cp "$report_backup" "$valid_report"

ptx_evidence="$evidence_dir/sm86_ptx_image_smoke-embedded-ptx.json"
cp "$ptx_evidence" "$test_root/ptx-evidence.backup"
jq '.loaded = false | .attempts[0].loaded = false' \
  "$test_root/ptx-evidence.backup" >"$ptx_evidence"
forged_ptx_sha="$(sha256sum "$ptx_evidence" | awk '{print $1}')"
jq --arg sha "$forged_ptx_sha" \
  '.tests.sm86_ptx_image_smoke.embedded_ptx_probe_sha256 = $sha' \
  "$report_backup" >"$valid_report"
if "$validator" "$valid_report" >/dev/null 2>&1; then
  echo "validator accepted relationally forged PTX evidence with a matching hash" >&2
  exit 1
fi
cp "$test_root/ptx-evidence.backup" "$ptx_evidence"
cp "$report_backup" "$valid_report"

assert_forged_probe_inventory_rejected() {
  local label="$1"
  local probe_path="$2"
  local report_filter="$3"
  local inventory_field="$4"
  local backup="$test_root/probe-inventory.backup"
  cp "$probe_path" "$backup"
  jq --arg field "$inventory_field" \
    '.[$field] = [{"offset": 42, "reason": ("forged " + $field)}]' \
    "$backup" >"$probe_path"
  local forged_sha
  forged_sha="$(sha256sum "$probe_path" | awk '{print $1}')"
  jq --arg sha "$forged_sha" "$report_filter" \
    "$report_backup" >"$valid_report"
  if "$validator" "$valid_report" >/dev/null 2>&1; then
    echo "validator accepted non-empty PTX inventory: $label" >&2
    exit 1
  fi
  cp "$backup" "$probe_path"
  cp "$report_backup" "$valid_report"
}
assert_forged_probe_inventory_rejected \
  "sm86 positive malformed probe" \
  "$ptx_evidence" \
  '.tests.sm86_ptx_image_smoke.embedded_ptx_probe_sha256 = $sha' \
  malformed_modules
assert_forged_probe_inventory_rejected \
  "sm86 positive incomplete probe" \
  "$ptx_evidence" \
  '.tests.sm86_ptx_image_smoke.embedded_ptx_probe_sha256 = $sha' \
  incomplete_modules
assert_forged_probe_inventory_rejected \
  "sm89 negative malformed probe" \
  "$negative_probe_path" \
  '.compatibility_controls.sm89_driver_negative.probe_sha256 = $sha' \
  malformed_modules
assert_forged_probe_inventory_rejected \
  "sm89 negative incomplete probe" \
  "$negative_probe_path" \
  '.compatibility_controls.sm89_driver_negative.probe_sha256 = $sha' \
  incomplete_modules

printf 'not a png, not CUDA\n' >"$test_root/not-a-png"
if "$repo_root/scripts/verify-png-artifact.py" \
  "$test_root/not-a-png" 256 256 >/dev/null 2>&1; then
  echo "PNG verifier accepted a text artifact" >&2
  exit 1
fi

if find "$repo_root/docs/qualification" -maxdepth 1 -type f \
  -name 'cuda-sm86-qualification*.json' -print -quit | grep -q .; then
  echo "a qualification result was checked in without an authorized hardware run" >&2
  exit 1
fi

echo "CUDA qualification contract: ok"
