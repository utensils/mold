# Metal memory policy validation

Branch: `feat/metal-memory-policy`. Original implementation base: `c6b925473`;
rebased onto patch repair `087f779af`. Delivery uses the existing
PR [#1592](https://github.com/utensils/mold/pull/1592), reopened while v0.27.1
publication completes. Its CI may run concurrently; merge follows patch
publication and exact-head feature checks, with branch protection unchanged.

## Completed local checks

Commands used the repository's resolved Nix development environment. The
following checks passed before the final Rust review corrections:

- Core policy: 7 tests; CLI administration: 17 tests; CLI black-box: 3 tests.
- Server Metal registry and OpenAPI reference-resolution tests.
- CPU workspace all-targets clippy and Metal all-targets clippy for the core,
  inference, server and CLI packages, with warnings denied.
- Studio: 1,779 tests; web: 1,871 tests; desktop: 6,150 tests (9,800 total).
- Web, desktop and mobile production builds; frontend architecture, formatting
  and dead-code checks.
- The complete local documentation route, including generated reference
  verification and the VitePress build.
- Browser fixture UAT of the actual shared DevicePanel at 1280 × 800 and
  390 × 844. Automatic, explicit, failed-probe and older-host states were
  exercised; screenshots were inspected; no console errors or horizontal
  overflow remained. This was a fixture, not a live inference server.

Failing-before/fixed-after regression evidence covers core arithmetic,
headroom above a reduced limit, host-floor deficits, administration rollback,
boot persistence and the DevicePanel. Final review added failing-before tests
for zero resident attribution and unchecked missing-device construction.
Injected snapshots also exercise combined probe errors, optional sysctl
absence, warm-cache admission after a limit reduction and unknown telemetry.
Temporary-directory persistence tests use native `plutil` on the exact frozen
plist fixture. No test writes to the real LaunchDaemons directory.

## Platform and qualification limits

The broad local contracts route also includes Linux/CUDA distribution and
qualification scripts. Eleven unrelated platform/environment checks could
not pass on this Mac: absent `readelf`/`flock`, macOS archive metadata,
architecture-specific CUDA assumptions, and `/var` symlink aliasing in a local
qualification fixture. They are not evidence of successful CUDA qualification.
The initial docs-format failure from that broad run was corrected and the
complete docs route subsequently passed. CUDA runtime tests were not run.

Initial read-only research observed macOS 26.5.2, M4 Max, 48 GiB physical RAM,
`iogpu.wired_limit_mb = 0`, and a Metal recommendation of 40,200,896,512 bytes.
These are observations of that machine at that time, not portable defaults.
CLI black-box tests verified early local status without config/DB initialization
or use of `MOLD_HOST`, malformed-input rejection and non-root refusal.

No privileged kernel write, live boot-policy installation, reboot, model
load/download or inference qualification was performed. Automated native GPU
snapshot qualification is explicitly ignored. The scoped Metal filter also
matched an existing allocator smoke test: it allocated two 64 × 64 F32 tensors
around a pool sweep and passed. This small allocation check is not model
inference or workload qualification; subsequent injected-only runs exclude it.
Injected tests and Metal compilation validate policy and code paths without
claiming a real workload result.
Recommendation refresh after a kernel increase/reset remains conservatively
bounded by the observed Metal recommendation until a fresh process reports a
new value, as documented in the user guide.

## Review-correction checks

The refreshed CPU workspace clippy and Metal package clippy passed with
`--all-targets -- -D warnings`. Injected inference tests passed in CPU (8)
and Metal (6) configurations; the Metal native snapshot test stayed ignored.
The warm-cache admission regression also passed in a Metal-enabled server.
After updating obsolete RAM-only Metal fixtures, the broader scheduler suite
passed all 150 tests and the memory-preflight suite passed all 27 tests.

Final refresh on implementation commit `32152e7bf`: core policy 8/8, CPU
inference filter 8/8, Metal injected filter 6/6 (one native probe ignored),
CPU workspace and Metal package all-targets clippy with warnings denied, and
Rust formatting all passed. The full device-registry suite also passed 14/14.
Fable's final review of the follow-up returned no findings and verified the
actual pinned Candle revision. The completion commit only records this
evidence, closes the checklist and corrects a test comment.

## Patch-base integration

The rebase preserves #1593's shared host-memory observation and unified phase
budgets. Metal policy collection now accepts that same host observation, so
server telemetry does not take a second Mach sample or substitute an estimated
available value when the authoritative observation is missing. A regression
covers zero, known and unavailable host headroom. The newer unified scheduler
fixtures provide the Metal policy that admission now requires.

Before these adjustments, the rebased server tests failed to compile at the
new snapshot field and helper call sites. Afterward, all 3 unified-memory
tests, the new shared-host regression, and all 18 resource tests passed.

On rebased implementation `f088e6d97`, CPU workspace and Metal package
all-targets clippy passed with warnings denied, along with 67 preflight and
71 registry-filtered tests. The Metal injected filter passed all 6 tests
with its native snapshot probe ignored and allocator smoke test excluded.
An existing post-upscale recovery test exceeded its two-second deadline
in two runs; it subsequently passed unchanged in isolation on both the
patch base (0.72 s) and feature branch (0.70 s). This is recorded as a
parallel-run timeout: the complete scheduler suite subsequently passed all
152 tests with `--test-threads=1`, while the parallel rerun exceeded the same
recovery deadline. No timeout or production behavior was changed to obtain
that result. The Metal-enabled warm-cache admission regression also passed.
