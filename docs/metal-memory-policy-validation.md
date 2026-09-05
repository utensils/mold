# Metal memory policy validation

Branch: `feat/metal-memory-policy`. Implementation base: `c6b925473`.
PR [#1592](https://github.com/utensils/mold/pull/1592) is closed, unmerged, at the
user's request. Its running CI was cancelled; subsequent validation is local.
No replacement PR or automatic CI run is required for this delivery.

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
