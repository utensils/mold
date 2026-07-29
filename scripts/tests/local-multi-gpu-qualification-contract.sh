#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/qualify-local-multi-gpu.py"
validator="$repo_root/scripts/validate-local-multi-gpu-report.py"
schema="$repo_root/docs/qualification/local-multi-gpu-report.schema.json"

[[ -x "$runner" ]] || {
  echo "local multi-GPU qualification runner is missing or not executable" >&2
  exit 1
}
[[ -x "$validator" ]] || {
  echo "local multi-GPU report validator is missing or not executable" >&2
  exit 1
}

help_text="$("$runner" --help)"
for option in \
  --binary \
  --models-dir \
  --model-artifact \
  --request \
  --expected-gpu-uuid \
  --port \
  --report \
  --selector-contract-command; do
  grep -Fq -- "$option" <<<"$help_text" || {
    echo "qualification help is missing $option" >&2
    exit 1
  }
done

grep -Fq 'never touches an existing server' <<<"$help_text"
grep -Fq 'does not claim ambiguous-prefix hardware' <<<"$help_text"
grep -Fq -- '--query-compute-apps=pid,gpu_uuid' "$runner"
grep -Fq 'MOLD_DB_PATH' "$runner"
grep -Fq 'MOLD_OUTPUT_DIR' "$runner"
grep -Fq 'MOLD_HOME' "$runner"
grep -Fq 'MOLD_DISPATCH_MODE' "$runner"
grep -Fq 'CUDA_VISIBLE_DEVICES' "$runner"
grep -Fq -- '--ro-bind' "$runner"
grep -Fq 'bwrap' "$runner"
grep -Fq 'local-2x-rtx3090-sm86' "$runner"
grep -Fq 'port 7680 is reserved' "$runner"

python3 -m py_compile "$runner" "$validator"

jq -e '
  .properties.schema_version.const == "mold.local.multi-gpu.qualification.v2"
  and .properties.qualification_profile.const == "local-2x-rtx3090-sm86"
  and .properties.hardware_qualified.type == "boolean"
  and .properties.candidate.required == ["path", "sha256", "version", "server_pid"]
  and .properties.host.properties.expected_gpu_uuids.minItems == 2
  and .properties.request.properties.job_count.minimum == 4
  and (.properties.request.required | index("artifacts")) != null
  and (.properties.evidence.items.required | index("kind")) != null
  and (.properties.checks.required | index("both_devices_discovered")) != null
  and (.properties.checks.required | index("both_devices_executed")) != null
  and (.properties.checks.required | index("busy_disable_drained")) != null
  and (.properties.checks.required | index("queue_replanned_after_disable")) != null
  and (.properties.checks.required | index("all_disabled_maintenance")) != null
  and (.properties.checks.required | index("queued_cancellation")) != null
  and (.properties.checks.required | index("restart_persistence")) != null
  and (.properties.checks.required | index("legacy_rollback")) != null
  and (.properties.checks.required | index("selector_matrix")) != null
  and (.properties.checks.required | index("client_projection")) != null
  and (.properties.checks.required | index("models_tree_unchanged")) != null
  and (.allOf | length) > 0
' "$schema" >/dev/null

python3 "$runner" --self-test
python3 "$repo_root/scripts/tests/local_multi_gpu_qualification_test.py"

grep -Fq 'scripts/qualify-local-multi-gpu.py' "$repo_root/.github/workflows/ci.yml"
grep -Fq 'scripts/validate-local-multi-gpu-report.py' "$repo_root/.github/workflows/ci.yml"
grep -Fq 'local-multi-gpu-qualification-contract.sh' "$repo_root/.github/workflows/ci.yml"

echo "local multi-GPU qualification contract passed"
