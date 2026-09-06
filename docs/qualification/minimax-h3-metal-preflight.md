# Offline H3 Metal budget preflight

This tool checks an **exported budget snapshot**, not a running model. It
cannot mint an executable H3 owner, prove instrumentation, authenticate the
export or decide that a GPU slot is available. Every result says
`launch_ready: false`, including a fitting snapshot. It uses Python's standard
library and starts no child process, device, model or network request.

```sh
python3 scripts/minimax-h3-metal-preflight.py /external/evidence/budget.json \
  > /external/evidence/budget-audit.json
python3 scripts/tests/minimax-h3-metal-preflight-test.py
```

Exit 0 means the supplied arithmetic fits the supplied snapshot; 1 means a
named capacity refusal; 2 means malformed input. None authorizes a launch.
Tests are CPU-only and run in CI and `scripts/ci-local.sh contracts`.

## Export contract

A qualification exporter must read the actual prepared
`H3FactoryTargetBudgetInput` and frozen request, not transcribe checkpoint
sizes or infer a Turbo budget from a base model. The current production
struct has no JSON serialization or standalone allocation-free report CLI;
that exporter still needs recovery or implementation. This auditor does not
supply missing runtime facts and an operator-written JSON file is not proof
that the request minted those values.

Input schema: `mold.h3-metal-budget-snapshot.v1`, with exactly these fields:

- `identities`: `source_commit`, `candle_commit` (40 lowercase hex digits),
  `executable_sha256`, `request_sha256`, `plan_sha256`, `budget_sha256`
  (64 lowercase hex digits). These are labels to verify against the capture,
  not authenticated authority merely because they have the right shape.
- `phase_bytes`: all 30 exact `*_phase_host_bytes` / `*_phase_device_bytes`
  fields for the 15 phases in `private_h3_unified_target_peak_bytes`:
  `reference_decode`, `reference_preprocess`, `reference_visual_encode`,
  `reference_audio_encode`, `vae_load`, `qwen_encode`, `qwen_transfer`,
  `condition_encode`, `noise_allocation`, `transformer_load`, `denoise`,
  `visual_decode`, `audio_decode`, `waveform_transfer`, `mux`. Preserve zero
  fields as well as nonzero fields. Extra or missing fields are refused.
- `owner_projection`: `device_bytes` must equal the maximum host-plus-device
  sum **within a phase**; `additional_host_bytes` must be zero. This is only
  the final Metal projection; the per-phase host columns remain intact.
- `snapshot`: `available_bytes` and `device_headroom_bytes`, captured on the
  actual execution host. No host is contacted and freshness is not inferred.
- `native_allocation_ceiling_bytes`: the proposed positive ceiling, not
  evidence that the executing allocator actually enforces it.

Byte counts must be unsigned 64-bit JSON integers, never booleans, fractions,
strings or overflowing phase sums. Duplicate JSON keys are refused. The
report retains both host/device columns, binding phases and the combined
peak. A source contract test pins the phase list to the Rust authority.

Capacity checks preserve the campaign's 24 GiB starting availability and
12 GiB host floor. The combined phase maximum must fit current device
headroom and host availability minus that floor. The ceiling must cover the
largest planned device phase, remain below device headroom, and leave room
for the largest planned host phase plus the host floor. This last ceiling
check is intentionally conservative: the ceiling is a global permission to
allocate, so it must not consume host-phase residency. It is distinct from
the exact within-phase sum used for the admission projection.

These are snapshot consistency checks, not runtime bounds proven on Metal.
Passing does not establish that actual native allocations stay below planned
phase values, or that sampled RSS and native allocations can be added (they
may overlap on unified memory).

## Still missing before default-resolution H3 execution

1. Recover/hash or implement the qualification-only **pre-allocation native
   ceiling**, covering every allocation path in the pinned Candle Metal
   backend. `metal_memory_guard.rs` is cooperative sampling; it is not a
   native allocator ceiling. CUDA behavior and the production Candle pin must
   remain unchanged unless separately reviewed.
2. Recover or implement the external watchdog and owned process-group
   cleanup harness. Verify baseline refusal, invalid/stale telemetry,
   pressure/swap abort, timeout, cancellation and descendant cleanup using
   CPU test doubles before any model run. There is currently **no executable
   guarded default-resolution launcher in this change**.
3. Export the actual per-request phase budget and capture native phase
   entry/exit and high-water measurements. The existing
   `private_runtime_observer.rs` reads Linux `/proc/self/status` and carries
   CUDA process fields; a macOS report needs its own truthful capture path.
4. Verify the **actual configured external Mold library on the execution
   Mac**. Historical `/Volumes/ExternalStorage/...` paths are not proof of
   today's configuration. Read that binary's `mold config path`,
   `mold config get models_dir`, `mold config get output_dir` and corresponding
   `config where` results, retain active profile/environment overrides,
   resolve defaults/symlinks, and verify the paths lie on the intended mounted
   external volume. A server's configuration must be checked on that server;
   the client host's values do not establish the server's library.
5. Retain downloaded model artifacts in that external library. Retain every
   generated image/video **in the library with gallery provenance**. Raw
   tensors, logs and diagnostics may use the external evidence directory,
   but it is not a substitute media destination. An isolated configuration
   must still target the verified library; do not infer the default output
   path when `output_dir` is unset.
6. Obtain H3's exclusive slot after Z-Image finishes, revalidate live
   headroom and instrumentation, and then run cases sequentially. The user
   released the Metal hold, but that does not grant H3 a concurrent slot.
7. Inspect every resulting image and representative video frames/contact
   sheets **and motion**, against prompt, conditioning and oracle. Record
   artifacts and uncertainty honestly. H3 audio also needs listening review.
   Decode success, file existence and numerical parity alone do not satisfy
   visual qualification; no UAT pass may be claimed without that inspection.

At preparation time on HPE, the execution Mac's configuration and retained
unshipped instrumentation were not accessible in this checkout. No actual
external-library path or default-shape budget has therefore been verified
by this tool. No GPU work was performed for these prerequisites.
