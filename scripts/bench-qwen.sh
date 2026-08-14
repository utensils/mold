#!/usr/bin/env bash
#
# bench-qwen.sh — reproducible Qwen-Image benchmark harness (issue #1049).
#
# Runs a fixed generation matrix against one mold binary, parses the CLI's own
# stage timings out of stderr, appends one JSON row per run to
# <out-dir>/rows.ndjson as it goes, and prints the whole run as a JSON array on
# stdout at the end (a human table goes to stderr while it runs). It is the
# measuring stick for the Qwen-Image performance milestone: every PR in that
# milestone is expected to move a number in this table, and `--gates` decides
# when the milestone is done.
#
# It is the *engine* benchmark. `scripts/qwen-2512-benchmark.sh` is the older,
# still-supported harness for the request path (local cold CLI, server model
# load, warm server streaming) at a small fixed size; this one owns the fixed
# resolution/quantization matrix and the milestone gates. Neither replaces the
# other, and their NDJSON schemas are deliberately separate.
#
# USAGE
#   scripts/bench-qwen.sh [options]
#   MOLD_BIN=./target/release/mold scripts/bench-qwen.sh --gates --reload-probe
#
#   --mold-bin PATH   mold binary. Default: $MOLD_BIN, else target/release/mold.
#   --out-dir DIR     Where per-run stderr transcripts and rows.ndjson land.
#                     Default: ${TMPDIR:-/tmp}/qwen-bench-<timestamp>
#   --out-image PATH  Generated image path (overwritten each run).
#                     Default: /tmp/bench-out.png
#   --prompt TEXT     Benchmark prompt. Default: the fixed fox prompt below.
#   --seed N          Default: 42
#   --repeats N       Runs per matrix config, for variance. Default 1. Every
#                     repeat is a separate cold process (see COLD vs WARM).
#   --skip-distilled  Do not run the optional distilled rows.
#   --reload-probe    Run the three-request server probe (see below), which is
#                     the only part of this harness that can observe a warm
#                     engine or a text-encoder reload. Needs curl.
#   --probe-host URL  Use an already-running server for the probe instead of
#                     starting one. Implies --reload-probe's transport, and the
#                     harness then never starts or stops a server.
#   --probe-distilled Also run each installed distilled model TWICE against the
#                     probe server. The second request is the warm row, which
#                     is the only warm distilled measurement this harness can
#                     make, and the row gate (a) times. Implies --reload-probe.
#                     --skip-distilled wins over it: the two together run no
#                     distilled work at all, rather than spending the GPU time
#                     on a measurement gate (a) then reports as skipped.
#   --probe-port N    Port for the server this harness starts. Default 7699.
#   --gates           After running, assert the milestone exit gates and exit
#                     non-zero listing the failures.
#   --dry-run         Print the planned matrix as JSON and exit. Runs nothing,
#                     needs no mold binary, no GPU — a cheap CI smoke. The plan
#                     has exactly the row count a real run records, including
#                     the rows a missing model or an early failure fills in.
#   -h, --help        This help.
#
# MATRIX (20 steps, seed 42, one fixed prompt, forced local, one row per run)
#   qwen-image-2512:q4   640x640    guidance 4.0  (CFG on)
#   qwen-image-2512:q4   1024x1024  guidance 4.0  (CFG on)
#   qwen-image-2512:q4   1328x1328  guidance 4.0  (CFG on)
#   qwen-image-2512:q4   1024x1024  guidance 1.0  (CFG off)
#   qwen-image-2512:q8   1328x1328  guidance 4.0  (CFG on)
#   qwen-image-lightning:fp8  1024x1024  guidance 1.0, manifest default steps
#   qwen-image-flash:q4       1024x1024  guidance 1.0, manifest default steps
#
# The q8 1328 row is expected to OOM today: a failed run is recorded as a row
# with status "oom_or_error", it never fails the harness. The distilled rows
# are optional — a model that is not installed is recorded as "model_missing"
# and never auto-pulled (a benchmark must not silently download 20 GB). Rows a
# config never got to run are recorded "not_run", so the row count of a run
# always equals the row count of its --dry-run plan.
#
# COLD vs WARM — read this before trusting a number
#   Every matrix row is its own `mold run --local` process, so its engine is
#   always cold: the weights are loaded for that one generation and dropped
#   with the process. Repeats measure variance, not warmth; the only thing a
#   repeat reuses is the OS page cache over the mmap'd weights.
#   `total_s` is the engine's own generation_time_ms, which starts AFTER
#   `load_for_request` (crates/mold-cli/src/commands/generate.rs), so it covers
#   prompt encoding, denoising, and VAE decode but NOT weight loading. Weight
#   loading is reported separately as te_load_s / transformer_load_s.
#   A warm engine (resident weights, a prompt-conditioning cache that can hit,
#   a text encoder that reloads instead of loading) exists only inside one
#   server process, so it is measured only by --reload-probe: three requests to
#   one `mold serve`, rendered by the same client stage lines (mold-cli's SSE
#   renderer prints the identical `✓ <stage> [12.3s]` output).
#     probe_cold    fresh server, weights load                 warm=false
#     probe_warm    same prompt again, engine resident         warm=true
#     probe_reload  DIFFERENT prompt, so the conditioning
#                   cache misses and the text encoder must
#                   reload or unpark                           warm=true
#   With --probe-distilled the same server then serves each installed distilled
#   model twice:
#     distilled_cold  first request for that model, weights load  warm=false
#     distilled_warm  same prompt again, engine resident          warm=true
#   Gate (a) is a WARM budget, so distilled_warm is the only row it TIMES. The
#   matrix's own distilled rows stay cold `mold run --local` processes and are
#   reported, but they are not what the gate is measured on: a cold row carries
#   a 20 GB checkpoint load that a 25 s budget was never meant to include. An
#   ERROR still fails the gate wherever it lands, distilled_cold included — a
#   failed cold request records the warm row as not_run, so a warm-only read
#   would let the likeliest failure (OOM on weight load) pass silently.
#
# BASELINE — 2026-08-14, RTX 4090, worktree sha 7a115622, 20 steps, cold rows
#   config                          s/step   total_s
#   q4 640x640    CFG on              1.17      27.1
#   q4 1024x1024  CFG off             1.87      43.3
#   q4 1024x1024  CFG on              3.61     119.8
#   q4 1328x1328  CFG on              8.48     180.9
#   q8 1328x1328  CFG on                —      OOM
#   Setup cost, NOT part of total_s: cold text-encoder load ~35.1 s.
#   Single samples, one process each, no variance control. The 1024 CFG-on row
#   leaves ~47 s of total_s outside denoise where the 1328 row leaves ~11 s;
#   that residual is unexplained and is exactly the kind of thing --repeats and
#   a re-measure on the current binary are for. Treat the table as a starting
#   point to reproduce, not as a certified result.
#
# EXIT GATES (--gates) — these are the milestone's targets, not today's truth.
#   (a) with --probe-distilled: every installed distilled model's WARM request
#       (distilled_warm) finishes in <= 25s, and neither of its probe requests
#       errored
#   (b) qwen-image-2512:q8 at 1328 CFG on completes (status ok, no OOM)
#   (c) qwen-image-2512:q4 at 1328, 20 steps, CFG on: total_s <= 140
#   (d) with --reload-probe: the probe_reload request pays <= 5s of text-encoder
#       setup — its reload/unpark stage, or its load stage, or 0 when the
#       encoder stayed resident and no text-encoder stage ran at all.
# Gates that reference not-yet-landed work FAIL today, and that is correct:
# (b) fails until the q8 1328 memory work lands. Gates whose opt-in flag is
# absent are reported as skipped, never as failures — (a) needs
# --probe-distilled, (d) needs --reload-probe. A green --gates run over the
# full matrix is the milestone's completion signal.
#
# Gate (c)'s budget is 140 s, recalibrated from the milestone's own end state:
# 148.4 s cold total at 1328 with batched CFG still off. The 110 s it replaced
# was set before any of the engine work was measured. It is a ceiling to hold,
# not a prediction — re-measure and tighten it when batched CFG lands at that
# resolution (#1048 item 2).
set -euo pipefail

BENCH_PROMPT_DEFAULT="a photorealistic red fox standing in a snowy forest clearing, morning light"
# Deliberately different subject so the probe's third request cannot hit the
# prompt-conditioning cache and must reload the text encoder.
BENCH_PROMPT_ALT="a weathered brass diving helmet on a workbench, harsh side light"
BENCH_STEPS=20
BENCH_SEED_DEFAULT=42
BENCH_DISTILLED_MODELS=("qwen-image-lightning:fp8" "qwen-image-flash:q4")
BENCH_PROBE_ROLES=("probe_cold" "probe_warm" "probe_reload")
BENCH_DISTILLED_PROBE_ROLES=("distilled_cold" "distilled_warm")

die() {
  echo "error: $*" >&2
  exit 2
}

usage() {
  sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" | sed -e 's/^#\{0,1\} \{0,1\}//' -e '$d'
}

# ---------------------------------------------------------------------------
# Transcript parsing (sourced and asserted by scripts/tests/bench-qwen-parse.sh)
# ---------------------------------------------------------------------------

# Progress bars redraw with carriage returns and every stage line is coloured;
# both have to go before any of this is line-oriented.
strip_ansi() {
  tr '\r' '\n' | sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g'
}

# Seconds from a `  ✓ <stage> [12.3s]` line matching an extended regex.
# The third argument picks which match wins: "first" (default) for a load that
# happens once per process, "last" for a stage that can repeat within one run.
stage_seconds() {
  local transcript="$1" pattern="$2" pick="${3:-first}"
  [[ -f "$transcript" ]] || return 0
  strip_ansi < "$transcript" | awk -v pat="$pattern" -v pick="$pick" '
    $0 ~ pat {
      if (match($0, /\[[0-9]+(\.[0-9]+)?s\]/)) {
        value = substr($0, RSTART + 1, RLENGTH - 3)
        if (pick == "first") {
          if (found == 0) { kept = value; found = 1 }
        } else {
          kept = value; found = 1
        }
      }
    }
    END { if (found) print kept }
  '
}

# Seconds from the final `✓ Done — <model> in 119.8s (seed: 42)` line.
transcript_total_seconds() {
  local transcript="$1"
  [[ -f "$transcript" ]] || return 0
  strip_ansi < "$transcript" | awk '
    /Done/ {
      if (match($0, / in [0-9]+(\.[0-9]+)?s/)) {
        last = substr($0, RSTART + 4, RLENGTH - 5)
      }
    }
    END { if (last != "") print last }
  '
}

# Step count the engine actually ran, from `  ✓ Denoising (20 steps) [72.2s]`.
# This is how a distilled checkpoint's manifest default steps are recorded.
transcript_steps() {
  local transcript="$1"
  [[ -f "$transcript" ]] || return 0
  strip_ansi < "$transcript" | awk '
    /Denoising \(/ {
      if (match($0, /\([0-9]+ steps?\)/)) {
        last = substr($0, RSTART + 1, RLENGTH - 2)
        sub(/ steps?/, "", last)
      }
    }
    END { if (last != "") print last }
  '
}

transcript_has_cache_hit() {
  local transcript="$1"
  [[ -f "$transcript" ]] || return 1
  strip_ansi < "$transcript" | grep -q '\[cache hit\]'
}

json_num() {
  if [[ -z "${1:-}" ]]; then
    printf 'null'
  else
    printf '%s' "$1"
  fi
}

# parse_transcript <stderr-log> -> {total_s, denoise_s, te_load_s, te_reload_s,
#                                   transformer_load_s, transformer_reload_s,
#                                   vae_s, steps, cache_hit}
# Every field is null when the transcript does not contain it. Nothing is
# inferred: a run that died before denoising reports null timings, not zeroes.
#
# The load and reload stages are deliberately separate fields. The Qwen
# pipeline's stage names are "Loading Qwen2.5 text encoder …" for the cold load
# and "Reloading Qwen2.5 encoder" / "Unparking Qwen2.5 encoder (CPU→GPU)" for
# the in-process residency path (crates/mold-inference/src/qwen_image/
# pipeline.rs), and "Loading Qwen-Image transformer …" vs "Reloading Qwen-Image
# transformer". Folding them together reports a 3 s reload as a cold load, or
# hides a reload behind a null. The leading `[^A-Za-z]` guards keep the load
# patterns from matching the tail of "Reloading".
parse_transcript() {
  local transcript="$1"
  local total denoise te_load te_reload transformer transformer_reload vae steps cache_hit
  total="$(transcript_total_seconds "$transcript")"
  denoise="$(stage_seconds "$transcript" 'Denoising' last)"
  te_load="$(stage_seconds "$transcript" '(^|[^A-Za-z])Loading Qwen2[.]5 (text )?encoder' first)"
  te_reload="$(stage_seconds "$transcript" '(Reloading|Unparking) Qwen2[.]5 (text )?encoder' last)"
  transformer="$(stage_seconds "$transcript" '(^|[^A-Za-z])Loading Qwen-Image transformer' first)"
  transformer_reload="$(stage_seconds "$transcript" 'Reloading Qwen-Image transformer' last)"
  vae="$(stage_seconds "$transcript" 'VAE decode' last)"
  steps="$(transcript_steps "$transcript")"
  cache_hit=false
  if transcript_has_cache_hit "$transcript"; then
    cache_hit=true
  fi
  jq -cn \
    --argjson total_s "$(json_num "$total")" \
    --argjson denoise_s "$(json_num "$denoise")" \
    --argjson te_load_s "$(json_num "$te_load")" \
    --argjson te_reload_s "$(json_num "$te_reload")" \
    --argjson transformer_load_s "$(json_num "$transformer")" \
    --argjson transformer_reload_s "$(json_num "$transformer_reload")" \
    --argjson vae_s "$(json_num "$vae")" \
    --argjson steps "$(json_num "$steps")" \
    --argjson cache_hit "$cache_hit" \
    '{
      total_s: $total_s,
      denoise_s: $denoise_s,
      te_load_s: $te_load_s,
      te_reload_s: $te_reload_s,
      transformer_load_s: $transformer_load_s,
      transformer_reload_s: $transformer_reload_s,
      vae_s: $vae_s,
      steps: $steps,
      cache_hit: $cache_hit
    }'
}

# bench_row <model> <width> <height> <steps|default> <guidance> <status> <mode>
#           <role> <repeat> <warm> <parsed-json> <git_sha> <mold_version>
#           <gpu_name> <timestamp>
bench_row() {
  local model="$1" width="$2" height="$3" steps_req="$4" guidance="$5"
  local status="$6" mode="$7" role="$8" repeat="$9" warm="${10}"
  local parsed="${11}" git_sha="${12}" mold_version="${13}" gpu_name="${14}" ts="${15}"
  local steps_json="null"
  if [[ "$steps_req" != "default" && -n "$steps_req" ]]; then
    steps_json="$steps_req"
  fi
  jq -cn \
    --arg model "$model" \
    --arg status "$status" \
    --arg mode "$mode" \
    --arg role "$role" \
    --arg git_sha "$git_sha" \
    --arg mold_version "$mold_version" \
    --arg gpu_name "$gpu_name" \
    --arg timestamp "$ts" \
    --argjson width "$width" \
    --argjson height "$height" \
    --argjson steps_req "$steps_json" \
    --argjson guidance "$guidance" \
    --argjson repeat "$repeat" \
    --argjson warm "$warm" \
    --argjson parsed "$parsed" \
    '
    (($parsed.steps // $steps_req)) as $steps
    | {
      model: $model,
      width: $width,
      height: $height,
      steps: $steps,
      guidance: $guidance,
      cfg: ($guidance > 1),
      status: $status,
      mode: $mode,
      role: $role,
      repeat: $repeat,
      total_s: $parsed.total_s,
      denoise_s: $parsed.denoise_s,
      s_per_step: (
        if $parsed.denoise_s != null and $steps != null and $steps > 0
        then (($parsed.denoise_s / $steps) * 100 | round) / 100
        else null
        end
      ),
      te_load_s: $parsed.te_load_s,
      te_reload_s: $parsed.te_reload_s,
      transformer_load_s: $parsed.transformer_load_s,
      transformer_reload_s: $parsed.transformer_reload_s,
      vae_s: $parsed.vae_s,
      cache_hit: $parsed.cache_hit,
      warm: $warm,
      git_sha: $git_sha,
      mold_version: $mold_version,
      gpu_name: $gpu_name,
      timestamp: $timestamp
    }'
}

# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------

# The revision the *benchmarked binary* came from cannot be read out of the
# binary, so this is best effort and says so: it is the HEAD of the work tree
# that contains the binary, with "-dirty" appended when that tree has
# uncommitted changes. A binary that is not inside a git work tree — anything
# resolved from PATH, an installed release — is "unknown" rather than the
# current directory's HEAD, which would attribute the measurement to a revision
# that never built it. Pair it with mold_version.
detect_git_sha() {
  local bin="$1" resolved dir sha
  resolved="$(command -v -- "$bin" 2>/dev/null || printf '%s' "$bin")"
  case "$resolved" in
    */*) ;;
    *)
      printf 'unknown'
      return
      ;;
  esac
  dir="$(cd "$(dirname -- "$resolved")" 2>/dev/null && pwd -P)" || {
    printf 'unknown'
    return
  }
  [[ "$(git -C "$dir" rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]] || {
    printf 'unknown'
    return
  }
  sha="$(git -C "$dir" rev-parse --short HEAD 2>/dev/null)" || {
    printf 'unknown'
    return
  }
  [[ -n "$sha" ]] || {
    printf 'unknown'
    return
  }
  if [[ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]]; then
    printf '%s-dirty' "$sha"
  else
    printf '%s' "$sha"
  fi
}

detect_mold_version() {
  local bin="$1" out
  out="$("$bin" --version 2>/dev/null | strip_ansi | head -n 1)" || {
    printf 'unknown'
    return
  }
  if [[ -z "$out" ]]; then
    printf 'unknown'
  else
    printf '%s' "$out"
  fi
}

detect_gpu_name() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || printf 'unknown'
  else
    printf 'unknown'
  fi
}

# `mold info <model>` prints `Status: Installed` / `Status: Not installed`
# locally, without a server. Probing it keeps `mold run --local` from
# auto-pulling a missing checkpoint mid-benchmark.
model_installed() {
  local model="$1" info
  info="$("$mold_bin" info "$model" 2>/dev/null | strip_ansi)" || return 1
  if grep -Eq 'Status:[[:space:]]*Not installed' <<<"$info"; then
    return 1
  fi
  grep -Eq 'Status:[[:space:]]*Installed' <<<"$info"
}

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

table_header_printed=0

print_table_header() {
  [[ "$table_header_printed" -eq 0 ]] || return 0
  table_header_printed=1
  printf '%-26s %-11s %5s %4s %-7s %5s %-14s %8s %8s %8s %8s %8s %7s\n' \
    MODEL SIZE STEPS CFG MODE WARM STATUS TOTAL_S S/STEP TE_S TE_RLD_S XF_S VAE_S >&2
  printf '%s\n' "$(printf '─%.0s' $(seq 1 134))" >&2
}

print_table_row() {
  local row="$1"
  print_table_header
  jq -r '
    def n: if . == null then "—" else tostring end;
    [
      .model,
      "\(.width)x\(.height)",
      (.steps | n),
      (if .cfg then "on" else "off" end),
      .mode,
      (if .warm then "yes" else "no" end),
      .status,
      (.total_s | n),
      (.s_per_step | n),
      (.te_load_s | n),
      (.te_reload_s | n),
      (.transformer_load_s | n),
      (.vae_s | n)
    ] | @tsv' <<<"$row" \
    | awk -F'\t' '{ printf "%-26s %-11s %5s %4s %-7s %5s %-14s %8s %8s %8s %8s %8s %7s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13 }' >&2
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# record_row <model> <width> <height> <steps> <guidance> <status> <mode> <role>
#            <repeat> <warm> <parsed-json>
record_row() {
  local row
  row="$(bench_row "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" \
    "$git_sha" "$mold_version" "$gpu_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
  printf '%s\n' "$row" >> "$rows_file"
  print_table_row "$row"
}

# record_status <model> <width> <height> <steps> <guidance> <status> <mode>
#               <role> <repeat> <warm>
# A row for a run that produced no transcript: model_missing, not_run, planned.
record_status() {
  record_row "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "$(parse_transcript /dev/null)"
}

# run_case <model> <width> <height> <steps|default> <guidance> <mode> <role>
#          <repeat> <warm> <prompt> <label>
# Records exactly one row. A failed generation is a recorded row, never a
# harness failure. `mode` is local (a forced-local `mold run --local`) or
# server (a request to the probe's `mold serve`, which the CLI renders with the
# same stage lines through its SSE progress renderer).
run_case() {
  local model="$1" width="$2" height="$3" steps="$4" guidance="$5"
  local mode="$6" role="$7" repeat="$8" warm="$9" prompt="${10}" label="${11}"
  local transcript="$out_dir/$label.stderr.log"
  local status_text="ok" exit_code=0
  local args=("run" "$model" "$prompt"
    "--width" "$width" "--height" "$height"
    "--guidance" "$guidance" "--seed" "$seed" "-o" "$out_image")
  if [[ "$mode" == "local" ]]; then
    args=("run" "--local" "${args[@]:1}")
  fi
  if [[ "$steps" != "default" ]]; then
    args+=("--steps" "$steps")
  fi

  echo "→ $label: $model ${width}x${height} steps=$steps guidance=$guidance mode=$mode warm=$warm" >&2
  set +e
  if [[ "$mode" == "server" ]]; then
    MOLD_HOST="$probe_host" "$mold_bin" "${args[@]}" \
      > "$out_dir/$label.stdout.log" 2> "$transcript"
  else
    "$mold_bin" "${args[@]}" > "$out_dir/$label.stdout.log" 2> "$transcript"
  fi
  exit_code=$?
  set -e

  local parsed
  parsed="$(parse_transcript "$transcript")"
  if [[ "$exit_code" -ne 0 ]] || [[ "$(jq -r '.total_s' <<<"$parsed")" == "null" ]]; then
    status_text="oom_or_error"
  fi

  record_row "$model" "$width" "$height" "$steps" "$guidance" \
    "$status_text" "$mode" "$role" "$repeat" "$warm" "$parsed"
  [[ "$status_text" == "ok" ]]
}

# One row per line: model|width|height|steps|guidance
core_matrix() {
  cat <<EOF
qwen-image-2512:q4|640|640|$BENCH_STEPS|4.0
qwen-image-2512:q4|1024|1024|$BENCH_STEPS|4.0
qwen-image-2512:q4|1328|1328|$BENCH_STEPS|4.0
qwen-image-2512:q4|1024|1024|$BENCH_STEPS|1.0
qwen-image-2512:q8|1328|1328|$BENCH_STEPS|4.0
EOF
}

# Whether the distilled probe actually runs.
#
# `--probe-distilled` asks for distilled work and `--skip-distilled` refuses
# all of it, so the two together must mean "no distilled work" — otherwise the
# harness spends the GPU time and then gate (a) reports itself skipped, which
# is spending a measurement nobody reads. Plan, run, and the dead-server
# fan-out all read this one predicate so they cannot disagree.
distilled_probe_runs() {
  [[ "$probe_distilled" -eq 1 && "$skip_distilled" -eq 0 ]]
}

emit_plan() {
  local model width height steps guidance spec i
  while IFS='|' read -r model width height steps guidance; do
    [[ -n "$model" ]] || continue
    for ((i = 0; i < repeats; i++)); do
      record_status "$model" "$width" "$height" "$steps" "$guidance" \
        planned local matrix "$i" false
    done
  done < <(core_matrix)
  if [[ "$skip_distilled" -eq 0 ]]; then
    for spec in "${BENCH_DISTILLED_MODELS[@]}"; do
      for ((i = 0; i < repeats; i++)); do
        record_status "$spec" 1024 1024 default 1.0 planned local matrix "$i" false
      done
    done
  fi
  if [[ "$reload_probe" -eq 1 ]]; then
    local role warm
    for ((i = 0; i < ${#BENCH_PROBE_ROLES[@]}; i++)); do
      role="${BENCH_PROBE_ROLES[$i]}"
      warm=true
      [[ "$role" != "probe_cold" ]] || warm=false
      record_status "qwen-image-2512:q4" 1024 1024 "$BENCH_STEPS" 4.0 \
        planned server "$role" "$i" "$warm"
    done
  fi
  if distilled_probe_runs; then
    for spec in "${BENCH_DISTILLED_MODELS[@]}"; do
      emit_distilled_probe_rows "$spec" planned
    done
  fi
}

# The two per-model rows the distilled probe records, in either the plan or a
# run that could not happen. Kept in one place so plan and run cannot disagree
# on cardinality, which is the whole point of the plan-vs-run diff.
emit_distilled_probe_rows() {
  local spec="$1" status="$2" i role warm
  for ((i = 0; i < ${#BENCH_DISTILLED_PROBE_ROLES[@]}; i++)); do
    role="${BENCH_DISTILLED_PROBE_ROLES[$i]}"
    warm=true
    [[ "$role" != "distilled_cold" ]] || warm=false
    record_status "$spec" 1024 1024 default 1.0 "$status" server "$role" "$i" "$warm"
  done
}

# run_config <model> <width> <height> <steps> <guidance> <label-prefix>
# Always records exactly $repeats rows: model_missing when the checkpoint is
# absent, not_run for the repeats a failure cut short. The plan and the run
# therefore agree on cardinality, which is what makes a plan-vs-result diff
# meaningful.
run_config() {
  local model="$1" width="$2" height="$3" steps="$4" guidance="$5" prefix="$6"
  local i label failed=0
  if ! model_installed "$model"; then
    echo "skip: $model is not installed (never auto-pulled by this harness)" >&2
    for ((i = 0; i < repeats; i++)); do
      record_status "$model" "$width" "$height" "$steps" "$guidance" \
        model_missing local matrix "$i" false
    done
    return 0
  fi
  for ((i = 0; i < repeats; i++)); do
    if [[ "$failed" -eq 1 ]]; then
      record_status "$model" "$width" "$height" "$steps" "$guidance" \
        not_run local matrix "$i" false
      continue
    fi
    label="$(printf '%s-%s' "$prefix" "$i")"
    if ! run_case "$model" "$width" "$height" "$steps" "$guidance" \
      local matrix "$i" false "$prompt" "$label"; then
      # The run failed (an expected OOM, for instance): its repeats would only
      # reproduce the same failure, so record them without paying for them.
      echo "  run failed; recording remaining repeats of this config as not_run" >&2
      failed=1
    fi
  done
}

# ---------------------------------------------------------------------------
# Server probe — the only warm measurement this harness can make
# ---------------------------------------------------------------------------

probe_server_pid=""

stop_probe_server() {
  [[ -n "$probe_server_pid" ]] || return 0
  kill "$probe_server_pid" 2>/dev/null || true
  wait "$probe_server_pid" 2>/dev/null || true
  probe_server_pid=""
}

# Starts `mold serve` on 127.0.0.1:$probe_port and waits for /api/status.
# Returns non-zero (without killing the harness) when it never comes up.
start_probe_server() {
  local log="$out_dir/probe-server.log" waited=0
  probe_host="http://127.0.0.1:$probe_port"
  echo "probe: starting $mold_bin serve --port $probe_port (log: $log)" >&2
  # Gate (d) measures the parking capability, which is opt-in (#1044): the
  # probe server runs with MOLD_KEEP_TE_RAM=1 so a reload is an unpark, not a
  # disk re-read. An externally supplied --probe-host keeps its own config.
  MOLD_KEEP_TE_RAM=1 "$mold_bin" serve --port "$probe_port" --bind 127.0.0.1 > "$log" 2>&1 &
  probe_server_pid=$!
  trap stop_probe_server EXIT
  while ((waited < 180)); do
    if ! kill -0 "$probe_server_pid" 2>/dev/null; then
      echo "probe: server exited during startup; see $log" >&2
      probe_server_pid=""
      return 1
    fi
    if probe_status_ok; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "probe: server did not answer /api/status within ${waited}s; see $log" >&2
  stop_probe_server
  return 1
}

# Readiness/auth-aware status check: a server started with MOLD_API_KEY
# answers /api/status only with the key attached, so send it when we have it.
probe_status_ok() {
  local -a auth=()
  [[ -z "${MOLD_API_KEY:-}" ]] || auth=(-H "x-api-key: $MOLD_API_KEY")
  curl -fsS "${auth[@]}" "$probe_host/api/status" >/dev/null 2>&1
}

# The selected probe server's own inventory — a model installed locally is not
# evidence the server has it, and vice versa.
probe_model_installed() {
  local model="$1"
  local -a auth=()
  [[ -z "${MOLD_API_KEY:-}" ]] || auth=(-H "x-api-key: $MOLD_API_KEY")
  curl -fsS "${auth[@]}" "$probe_host/api/models" 2>/dev/null \
    | jq -e --arg m "$model" 'any(.[]; .name == $m and (.installed // true))' >/dev/null 2>&1
}

record_base_probe_rows_as() {
  local status="$1" i role warm
  for ((i = 0; i < ${#BENCH_PROBE_ROLES[@]}; i++)); do
    role="${BENCH_PROBE_ROLES[$i]}"
    warm=true
    [[ "$role" != "probe_cold" ]] || warm=false
    record_status "qwen-image-2512:q4" 1024 1024 "$BENCH_STEPS" 4.0 \
      "$status" server "$role" "$i" "$warm"
  done
}

record_probe_rows_as() {
  local status="$1" spec
  record_base_probe_rows_as "$status"
  # The distilled probe shares this server, so it shares its fate.
  if distilled_probe_runs; then
    for spec in "${BENCH_DISTILLED_MODELS[@]}"; do
      emit_distilled_probe_rows "$spec" "$status"
    done
  fi
}

# Three requests to ONE server process: cold, warm-same-prompt, and a
# different prompt that must miss the conditioning cache and reload the text
# encoder. This is the only place a warm engine or a reload can be observed —
# a `mold run --local` process holds its engine for exactly one generation.
run_probe() {
  local started_here=0 server_died=0 base_present=1 i role warm probe_prompt
  if [[ -z "$probe_host" ]]; then
    if ! model_installed "qwen-image-2512:q4"; then
      base_present=0
      echo "skip: reload probe needs qwen-image-2512:q4 installed" >&2
      record_base_probe_rows_as model_missing
      if ! distilled_probe_runs; then
        return 0
      fi
      # An installed Lightning/Flash still gets its warm measurement: the
      # base model gates only its own rows, not the shared server.
    fi
    if ! start_probe_server; then
      record_probe_rows_as not_run
      return 0
    fi
    started_here=1
  else
    echo "probe: using already-running server $probe_host" >&2
    if ! probe_status_ok; then
      echo "probe: $probe_host did not answer /api/status" >&2
      record_probe_rows_as not_run
      return 0
    fi
    if ! probe_model_installed "qwen-image-2512:q4"; then
      base_present=0
      echo "skip: $probe_host does not report qwen-image-2512:q4 installed" >&2
      record_base_probe_rows_as model_missing
      if ! distilled_probe_runs; then
        return 0
      fi
    fi
  fi

  for ((i = 0; i < ${#BENCH_PROBE_ROLES[@]}; i++)); do
    [[ "$base_present" -eq 1 ]] || break
    role="${BENCH_PROBE_ROLES[$i]}"
    warm=true
    if [[ "$role" == "probe_cold" ]]; then
      # An externally supplied server may already hold the engine or cached
      # conditioning: its initial state is unknown, not cold.
      if [[ "$started_here" -eq 1 ]]; then warm=false; else warm=null; fi
    fi
    probe_prompt="$prompt"
    [[ "$role" != "probe_reload" ]] || probe_prompt="$BENCH_PROMPT_ALT"
    if ! run_case "qwen-image-2512:q4" 1024 1024 "$BENCH_STEPS" 4.0 \
      server "$role" "$i" "$warm" "$probe_prompt" "$role"; then
      # A dead server cannot answer the remaining requests: a fatal CUDA fault
      # stops the process on purpose (see CLAUDE.md), so record the rest
      # instead of timing out against a socket nobody is listening on.
      local rest rest_role rest_warm
      for ((rest = i + 1; rest < ${#BENCH_PROBE_ROLES[@]}; rest++)); do
        rest_role="${BENCH_PROBE_ROLES[$rest]}"
        rest_warm=true
        [[ "$rest_role" != "probe_cold" ]] || rest_warm=false
        record_status "qwen-image-2512:q4" 1024 1024 "$BENCH_STEPS" 4.0 \
          not_run server "$rest_role" "$rest" "$rest_warm"
      done
      server_died=1
      break
    fi
  done

  if distilled_probe_runs; then
    if [[ "$server_died" -eq 1 ]]; then
      # The distilled probe's first act is a `/api/models` curl. Against a
      # server this loop just concluded is dead that fails, and a failed
      # inventory read is recorded as `model_missing` — reporting a crashed
      # server as "the model is not installed". These rows never ran.
      echo "probe: skipping the distilled probe, the probe server is gone" >&2
      local spec
      for spec in "${BENCH_DISTILLED_MODELS[@]}"; do
        emit_distilled_probe_rows "$spec" not_run
      done
    else
      run_distilled_probe
    fi
  fi

  if [[ "$started_here" -eq 1 ]]; then
    stop_probe_server
  fi
}

# Gate (a) is a warm budget, and a warm engine only exists inside one server
# process — so each installed distilled model is asked for the SAME prompt
# twice against the probe server. The first request loads its weights; the
# second finds them resident and its prompt conditioning cached, which is the
# measurement the gate reads.
run_distilled_probe() {
  local m spec i role warm
  for ((m = 0; m < ${#BENCH_DISTILLED_MODELS[@]}; m++)); do
    spec="${BENCH_DISTILLED_MODELS[$m]}"
    if ! probe_model_installed "$spec"; then
      echo "skip: $probe_host does not report $spec installed" >&2
      emit_distilled_probe_rows "$spec" model_missing
      continue
    fi
    for ((i = 0; i < ${#BENCH_DISTILLED_PROBE_ROLES[@]}; i++)); do
      role="${BENCH_DISTILLED_PROBE_ROLES[$i]}"
      warm=true
      [[ "$role" != "distilled_cold" ]] || warm=false
      if ! run_case "$spec" 1024 1024 default 1.0 \
        server "$role" "$i" "$warm" "$prompt" "$(printf '%s-%s' "${spec//[:.]/_}" "$role")"; then
        # A dead server cannot answer what is left, for this model or the next.
        # The fan-out covers the models AFTER this one only — iterating the
        # whole array re-emits rows for models that already completed, and the
        # plan-vs-run cardinality diff this function exists to protect then
        # sees more rows than were planned.
        local rest rest_role rest_warm later
        for ((rest = i + 1; rest < ${#BENCH_DISTILLED_PROBE_ROLES[@]}; rest++)); do
          rest_role="${BENCH_DISTILLED_PROBE_ROLES[$rest]}"
          rest_warm=true
          [[ "$rest_role" != "distilled_cold" ]] || rest_warm=false
          record_status "$spec" 1024 1024 default 1.0 \
            not_run server "$rest_role" "$rest" "$rest_warm"
        done
        for ((later = m + 1; later < ${#BENCH_DISTILLED_MODELS[@]}; later++)); do
          emit_distilled_probe_rows "${BENCH_DISTILLED_MODELS[$later]}" not_run
        done
        return 0
      fi
    done
  done
}

run_matrix() {
  local model width height steps guidance spec
  while IFS='|' read -r model width height steps guidance; do
    [[ -n "$model" ]] || continue
    run_config "$model" "$width" "$height" "$steps" "$guidance" \
      "$(printf '%s-%sx%s-g%s' "${model//[:.]/_}" "$width" "$height" "$guidance")"
  done < <(core_matrix)

  if [[ "$skip_distilled" -eq 0 ]]; then
    for spec in "${BENCH_DISTILLED_MODELS[@]}"; do
      run_config "$spec" 1024 1024 default 1.0 \
        "$(printf '%s-1024x1024-distilled' "${spec//[:.]/_}")"
    done
  fi

  if [[ "$reload_probe" -eq 1 ]]; then
    run_probe
  fi
}

# ---------------------------------------------------------------------------
# Milestone exit gates
# ---------------------------------------------------------------------------

gate_failures=()
gate_skips=()

check_gates() {
  local detail

  # (a) every installed distilled model's WARM request finishes in <= 25s.
  # A matrix row is a cold `mold run --local` process that pays a 20 GB
  # checkpoint load the 25 s budget was never meant to cover, so the *budget*
  # is read off the probe's distilled_warm row and nothing else. But an error
  # is a failure of the promise wherever it lands, and the likeliest failure
  # (OOM on weight load) lands on the COLD request — which then records the
  # warm row as `not_run`, so a warm-only gate would silently pass on whichever
  # models happened to succeed. Both probe roles are therefore scanned for
  # errors, while only the warm row is timed. Without --probe-distilled there
  # is no warm measurement at all and the gate reports itself skipped.
  if [[ "$probe_distilled" -ne 1 ]]; then
    gate_skips+=("(a) distilled warm total_s <= 25: re-run with --probe-distilled to evaluate it")
  elif [[ "$skip_distilled" -eq 1 ]]; then
    gate_skips+=("(a) distilled warm total_s <= 25: re-run without --skip-distilled to evaluate it")
  else
    detail="$(jq -s -r '
      [ .[] | select(.role == "distilled_cold" or .role == "distilled_warm")
            | select(.status != "model_missing" and .status != "planned") ] as $rows
      | ($rows | map(select(.status != "ok" and .status != "not_run"))
               | map("\(.model) \(.role) status=\(.status)")) as $errored
      | ($rows | map(select(.role == "distilled_warm" and .status == "ok"))) as $warm
      | ($warm | map(select(.total_s == null or .total_s > 25))
               | map("\(.model) total_s=\(.total_s)")) as $slow
      | if (($errored + $slow) | length) > 0 then
          (($errored + $slow) | join(", "))
        elif ($warm | length) == 0 then
          "no installed distilled model produced a warm run"
        else "" end' "$rows_file")"
    if [[ -n "$detail" ]]; then
      gate_failures+=("(a) distilled warm total_s <= 25: $detail")
    fi
  fi

  # (b) q8 1328 CFG on completes
  detail="$(jq -s -r '
    [ .[] | select(.role == "matrix" and .model == "qwen-image-2512:q8"
                   and .width == 1328 and .cfg == true) ] as $rows
    | if ($rows | length) == 0 then "no q8 1328 CFG row was recorded"
      elif ($rows | map(select(.status == "ok")) | length) == 0 then
        "status=\($rows | map(.status) | unique | join(","))"
      else "" end' "$rows_file")"
  if [[ -n "$detail" ]]; then
    gate_failures+=("(b) q8 1328 CFG status == ok: $detail")
  fi

  # (c) q4 1328 20-step CFG total_s <= 140. Every matrix row is a cold process
  # and total_s excludes weight loading, so there is no warm row to prefer.
  # 140 is recalibrated from the milestone's own end state — 148.4 s measured at
  # 1328 with batched CFG still off — replacing a 110 that predated every
  # engine measurement. Tighten it once batched CFG lands at this resolution.
  detail="$(jq -s -r --argjson steps "$BENCH_STEPS" '
    [ .[] | select(.role == "matrix" and .model == "qwen-image-2512:q4"
                   and .width == 1328 and .cfg == true
                   and .status == "ok" and .steps == $steps) ] as $rows
    | if ($rows | length) == 0 then "no successful q4 1328 CFG row was recorded"
      else ($rows | map(select(.total_s == null or .total_s > 140))
                  | map("total_s=\(.total_s)") | join(", "))
      end' "$rows_file")"
  if [[ -n "$detail" ]]; then
    gate_failures+=("(c) q4 1328 total_s <= 140: $detail")
  fi

  # (d) the probe's different-prompt request pays <= 5s of text-encoder setup.
  # A reload or unpark stage is the measurement; a plain load stage counts too
  # (the encoder was gone rather than parked); no text-encoder stage at all
  # means it stayed resident, which costs nothing and passes.
  if [[ "$reload_probe" -eq 1 ]]; then
    detail="$(jq -s -r '
      [ .[] | select(.role == "probe_reload") ] as $rows
      | if ($rows | length) == 0 then "the reload probe recorded no row"
        else ($rows[0]) as $r
          | if $r.status != "ok" then "probe_reload status=\($r.status)"
            else (($r.te_reload_s // $r.te_load_s // 0)) as $cost
              | if $cost > 5 then "text-encoder setup=\($cost)s" else "" end
            end
        end' "$rows_file")"
    if [[ -n "$detail" ]]; then
      gate_failures+=("(d) reload-probe text-encoder setup <= 5s: $detail")
    fi
  else
    gate_skips+=("(d) reload-probe text-encoder setup <= 5s: re-run with --reload-probe to evaluate it")
  fi

  local skipped
  for skipped in ${gate_skips[@]+"${gate_skips[@]}"}; do
    echo "gate skipped: $skipped" >&2
  done

  if [[ "${#gate_failures[@]}" -eq 0 ]]; then
    echo "gates: all evaluated milestone gates pass" >&2
    return 0
  fi
  echo "gates: ${#gate_failures[@]} failure(s)" >&2
  local failure
  for failure in "${gate_failures[@]}"; do
    echo "  FAIL $failure" >&2
  done
  return 1
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

main() {
  mold_bin="${MOLD_BIN:-target/release/mold}"
  out_dir=""
  out_image="/tmp/bench-out.png"
  prompt="$BENCH_PROMPT_DEFAULT"
  seed="$BENCH_SEED_DEFAULT"
  repeats=1
  skip_distilled=0
  reload_probe=0
  probe_distilled=0
  probe_host=""
  probe_port=7699
  gates=0
  dry_run=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mold-bin)
        [[ $# -ge 2 ]] || die "--mold-bin requires a value"
        mold_bin="$2"
        shift 2
        ;;
      --out-dir)
        [[ $# -ge 2 ]] || die "--out-dir requires a value"
        out_dir="$2"
        shift 2
        ;;
      --out-image)
        [[ $# -ge 2 ]] || die "--out-image requires a value"
        out_image="$2"
        shift 2
        ;;
      --prompt)
        [[ $# -ge 2 ]] || die "--prompt requires a value"
        prompt="$2"
        shift 2
        ;;
      --seed)
        [[ $# -ge 2 ]] || die "--seed requires a value"
        seed="$2"
        shift 2
        ;;
      --repeats)
        [[ $# -ge 2 ]] || die "--repeats requires a value"
        repeats="$2"
        shift 2
        ;;
      --skip-distilled)
        skip_distilled=1
        shift
        ;;
      --reload-probe)
        reload_probe=1
        shift
        ;;
      --probe-host)
        [[ $# -ge 2 ]] || die "--probe-host requires a value"
        probe_host="$2"
        reload_probe=1
        shift 2
        ;;
      --probe-distilled)
        probe_distilled=1
        reload_probe=1
        shift
        ;;
      --probe-port)
        [[ $# -ge 2 ]] || die "--probe-port requires a value"
        probe_port="$2"
        shift 2
        ;;
      --gates)
        gates=1
        shift
        ;;
      --dry-run)
        dry_run=1
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        die "unknown option: $1"
        ;;
    esac
  done

  [[ "$seed" =~ ^[0-9]+$ ]] || die "--seed must be an integer"
  [[ "$repeats" =~ ^[0-9]+$ ]] && [[ "$repeats" -ge 1 ]] || die "--repeats must be >= 1"
  [[ "$probe_port" =~ ^[0-9]+$ ]] || die "--probe-port must be an integer"
  command -v jq >/dev/null 2>&1 || die "required command not found: jq"

  if [[ -z "$out_dir" ]]; then
    out_dir="${TMPDIR:-/tmp}/qwen-bench-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir -p "$out_dir"
  rows_file="$out_dir/rows.ndjson"
  : > "$rows_file"

  if [[ "$dry_run" -eq 1 ]]; then
    git_sha="$(detect_git_sha "${BASH_SOURCE[0]}")"
    mold_version="unknown"
    gpu_name="$(detect_gpu_name)"
    emit_plan
    jq -s '.' "$rows_file"
    return 0
  fi

  if [[ ! -x "$mold_bin" ]]; then
    command -v "$mold_bin" >/dev/null 2>&1 || die "mold binary not found: $mold_bin (set MOLD_BIN or --mold-bin)"
  fi
  if [[ "$reload_probe" -eq 1 ]]; then
    command -v curl >/dev/null 2>&1 || die "--reload-probe requires curl"
  fi
  git_sha="$(detect_git_sha "$mold_bin")"
  mold_version="$(detect_mold_version "$mold_bin")"
  gpu_name="$(detect_gpu_name)"

  echo "mold:  $mold_bin ($mold_version, git $git_sha)" >&2
  echo "gpu:   $gpu_name" >&2
  echo "rows:  $rows_file (appended as each run finishes)" >&2
  echo "logs:  $out_dir" >&2

  run_matrix

  jq -s '.' "$rows_file"

  if [[ "$gates" -eq 1 ]]; then
    check_gates
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
