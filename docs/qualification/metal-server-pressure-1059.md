# Metal server pressure qualification: issue #1059

Status on 2026-09-05: **acceptance remains open**. Two real server render
attempts reached encoder loading and were stopped by the external host-memory
guard. Neither produced an output. Sustained pressure-margin rendering and
chain-stage reservation qualification remain unproven.

Tracked in [#1059](https://github.com/utensils/mold/issues/1059), following the
accounting implementation in [#1593](https://github.com/utensils/mold/pull/1593)
and working-set policy in [#1592](https://github.com/utensils/mold/pull/1592).
This follow-up authorized real inference; the earlier implementation-only
restriction did not apply to these attempts.

## Source and environment

- Source: `e1bf871c799fd6dc88628dc69fcbb98fbd43c692`, clean at build time.
- Binary: `mold 0.27.1 (e1bf871 2026-09-05)`.
- Binary SHA-256:
  `eb22435cb54fee9dcfcb925551b2f4104278ea7c5c8cbc6e88df7f64599f85d1`.
- Apple M4 Max, 48 GiB RAM, macOS 26.5.2 (25F84).
- Observed `iogpu.wired_limit_mb = 0` (automatic); Metal recommended capacity
  40,200,896,512 bytes. No kernel limit or boot policy was changed.
- Independent external build target; Nix Rust 1.95.0; four build jobs.
  Build completed successfully in 6m 34s.
- Scratch server bound to `127.0.0.1:17659`, with separate configuration,
  database, queue ownership and output directories. The second attempt used
  another fresh home to prevent replay of the interrupted first job.
- Existing model files were reused. The second attempt added only the Q8 UMT5
  companion in scratch storage, with symlinks to existing model components.
  No existing model/output files or shared services were modified by this run.
- The binary had no embedded SPA bundle. Qualification used the real server
  APIs; no browser/UI qualification was attempted.

Build invocation, from the recorded source:

```sh
nix develop -c sh -c 'CARGO_TARGET_DIR=/Volumes/ExternalStorage/mold-1059-qualification/target CARGO_BUILD_JOBS=4 cargo build --locked --profile dev-fast -p mold-ai --features metal,preview,mp4'
```

## Guard and workload

An independently reviewed external supervisor sampled native `vm_stat` and
`sysctl` observations separately from HTTP polling. Available memory was
`(Pages free + Pages inactive) * page_size`, matching the server's macOS
authority. It stopped the scratch workload at:

- available memory below 12 GiB;
- swap growth above 256 MiB from that attempt's baseline;
- non-normal kernel pressure, failed/stale native sampling, helper failure,
  or the bounded session timeout.

The prepared pressure helper had a 6 GiB ceiling, 64 MiB random allocations,
a 14 GiB growth cutoff and parent-loss/timeout cleanup. **It remained at zero
for both attempts.** After the first abort, the supervisor shortened forced
server cleanup from five seconds to half a second for a guard-triggered stop.
Graceful shutdown was requested before escalation. Scratch children were
verified absent before releasing the shared heavy-work reservation.

Native sampling stopped at the abort trigger and resumed for the final
post-cleanup sample. It does not bound memory during the intervening cleanup
gap. In the first attempt, the server's RSS log continued rising from about
12.3 GB to 14.3 GB during its five-second shutdown grace. The 12 GiB threshold
was a stop trigger, not a guaranteed minimum throughout shutdown; the table
reports the observed abort sample, not an unobserved whole-process minimum.

Both requests used `wan21-t2v-1.3b:bf16`, 512×288, 17 frames, 16 fps,
30 steps, guidance 6, seed 1059 and MP4 output, without prompt expansion:

> Medium wide shot in soft morning light. A red fox walks slowly through fresh
> snow in a quiet pine forest. Its paws lift small puffs of powder. The camera
> remains steady, showing the fox in profile against dark green trees.

Authoritative placement previews returned `planned`, with no pending downloads
or missing components. Initial preview latency was checkpoint hashing, confirmed
by a process sample. Each durable singleton was submitted once through
`POST /api/generation-batches`; the observed queue reached encoder loading.

## Observed results

| Observation                                   | Manifest FP16 UMT5                     | Explicit Q8 UMT5                       |
| --------------------------------------------- | -------------------------------------- | -------------------------------------- |
| Job ID                                        | `08c5e366-7807-4518-bdde-801e69eda7b0` | `c01bc057-9d99-4942-94be-deeae0919591` |
| Batch ID                                      | `77aafa6a-7055-4508-8313-1843bedc05a6` | `507185e0-d005-4f71-8ea2-eb9694c2125b` |
| Runtime reached                               | Loading UMT5 encoder                   | Loading UMT5 encoder                   |
| Guard elapsed from dispatch, approximately    | 16 seconds                             | 4 seconds                              |
| Native available memory at abort trigger      | 11.95 GiB                              | 11.17 GiB                              |
| Maximum sampled Metal allocation while active | 11.91 GiB                              | 11.53 GiB                              |
| Swap growth during sampled attempt            | 0 MiB                                  | 0 MiB                                  |
| Kernel pressure during sampled attempt        | Normal                                 | Normal                                 |
| Available memory after child cleanup          | 24.46 GiB                              | 26.75 GiB                              |
| Completed/decoded output                      | None                                   | None                                   |

The second encoder was selected with `MOLD_UMT5_VARIANT=q8` in the isolated
server environment. The downloaded file was
`city96/umt5-xxl-encoder-gguf/umt5-xxl-encoder-Q8_0.gguf`, exactly
6,043,068,256 bytes, matching the manifest. SHA-256:
`2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc`.

The external guard used a 12 GiB trigger on raw available memory. Production
derives headroom from an 8 GiB host floor and Metal working-set observations;
scheduled reservations also constrain admission. These are different checks,
so neither threshold alone establishes which policy is stricter. Crossing the
external guard does not by itself prove that production's floor was violated
or that the admission estimate was wrong. The attempts do not justify a
scheduler-policy change on their own.

## Evidence and remaining acceptance

Raw evidence is retained locally at
`/Volumes/ExternalStorage/mold-1059-qualification/evidence/`:

- `build.log`, source/binary/encoder identities and request/response JSON;
- `fp16-guarded-abort/`: first attempt's native/API samples, runtime log,
  baseline, abort, final sample and summary;
- `q8-guarded-abort/`: the corresponding second-attempt evidence;
- `preview-hang.sample.txt`: checkpoint hashing sample;
- `plan.md`, plus `../supervise.py`: reviewed plan and the second attempt's
  executed guard; the first attempt's earlier script version was not retained;
- `evidence-sha256.json`: integrity hashes for the retained evidence files.

Samples include `/api/status`, `/api/resources`, `/api/devices`, `/api/queue`
and `/api/activity`. Public queue plans expose lease-derived work identity,
device, execution fingerprint and activity phase; they do not expose every
internal reservation byte or ledger transition. No numerical reservation
release claim is made from those APIs.

The immediate prerequisite is a configuration and live host headroom that can
complete an unpressured baseline while preserving the guard. A quieter host or
a separately qualified smaller encoder configuration may provide that; neither
was established here. Other encoder tiers remain untested. Do not infer a
universal minimum-RAM requirement from these two guarded attempts.

Before closing #1059, still require:

- a completed, decoded normal server render;
- measured actual peak and safe, sustained near-margin acceptance/refusal,
  distinguishing durable queue admission from device dispatch;
- an authored two-stage chain with distinct stage work identities, correct
  device/fingerprint contracts, stage ownership transitions and final release;
- successful scratch unload and another real render proving clean recovery.

No artificial-pressure run or chain submission was made after the baseline
guards fired. There is no passing inference or output-correctness claim in this
record, and the final acceptance checkbox in #1059 stays unchecked.
