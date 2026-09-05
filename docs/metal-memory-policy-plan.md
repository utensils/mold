# macOS Metal memory policy

Branch: `feat/metal-memory-policy`. One long-running branch and one PR; commit
and push coherent milestones. Base: `c6b925473` (origin/main at planning).

## Outcome and boundaries

Mold must discover the macOS GPU working-set limit, respect it in local and
server memory decisions, explain it to clients, and let an administrator
explicitly configure it. A user selecting 16384 MiB must not have the scheduler
continue advertising all installed RAM as available GPU capacity.

The kernel setting is machine-wide. This feature is a conservative Mold
admission/allocation-planning policy, not a claim that Metal enforces an exact
per-process memory cap. Neither the setting nor a planning estimate reserves
memory against other applications. Existing host pressure, swap growth,
streaming synchronization, OOM recovery, and host-ledger guards remain active.

No new inference pipeline, MLX dependency, residency-set allocator, Candle fork
revision, HTTP privilege escalation, GUI privileged helper, arbitrary sysctl
interface, or background setting-reconciliation loop is needed. The desktop,
web and mobile surfaces consume host telemetry; administrative changes are
performed locally on the inference host. This includes explicit optional boot
persistence through a fixed root-owned LaunchDaemon, not SMAppService packaging.

## Evidence

- Local read-only probe: macOS 26.5.2, M4 Max, 48 GiB RAM,
  `iogpu.wired_limit_mb=0`, Metal recommendation 40200896512 bytes (37.44 GiB).
  Zero is not zero capacity. The sample command would LOWER this machine's limit.
- Apple defines `recommendedMaxWorkingSetSize` as a performance recommendation:
  <https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize>.
- MLX documents the system sysctl and queries the effective limit through Metal:
  <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html>.
  Its process residency-set control is separate from this system setting.
- Apple launchd lifecycle:
  <https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html>.
- Current `device.rs` exposes RAM-based discovery, `free_vram_bytes`, stable
  `metal_unified_capacity_with_safety_floor`, and incremental
  `metal_live_allocation_budget`. Stable capacity currently preserves
  `max(15% RAM, 8 GiB)` but does not clamp to Metal's recommendation.
- Callers include LTX-2 adaptive streaming, scheduler reclaim, execution-plan
  preparation, variant dependencies, and H3 worker post-drop validation.
- Ordinary scheduler capacity actually comes from `resources::metal_snapshot`
  through `device_registry::SchedulerDeviceProjection`, then
  `schedulable_available_vram_bytes`; its terminal failure classifier separately
  reads `total_vram_bytes_by_device_id`. Both ceilings must become policy-aware.

## Shared contract and accounting

Add a small core module for serializable telemetry and pure policy; put native
sampling behind macOS/Metal gates in inference. Missing telemetry stays unknown,
never a fabricated zero. Sysctl absence is distinct from a failed read; raw zero
means automatic. Keep installed RAM visible as hardware inventory.

The snapshot describes raw sysctl MiB (and read status), physical RAM, live
available RAM, Metal recommended working set, this Mold Metal device's allocated
bytes, host safety floor, effective total capacity and incremental headroom.
Probe eagerly during Metal discovery (no models or tensor allocations), using
the memoized device, and refresh from the background sampler. Distinguish no
Metal build, non-macOS, and a failed supported probe. CPU-only/non-Mac paths retain
their existing behavior. A failed supported Metal probe prevents that device's
admission with explicit unavailable diagnostics; eager discovery removes the
lazy-start deadlock. Fable suggested a RAM fallback; this is intentionally not
accepted because it would defeat an administrator's smaller limit exactly when
it cannot be read. A missing optional sysctl can still use a valid recommendation.
Explicit positive raw limits further clamp a stale recommendation, never increase
it. Sysctl is unsigned 32-bit MiB (verify native ABI); conversion is checked.

Total policy capacity = min(Metal recommendation, explicit positive sysctl limit
when present, RAM minus host safety floor). Incremental headroom is separately
bounded by live available RAM and capacity minus current Metal allocations.
Streaming additionally preserves the existing live host floor. A resident-byte
charge must occur exactly once: scheduler reservations and cache credits must
be traced before choosing the total or incremental accessor at each call site.
Do not infer other applications' GPU allocations from this process's Metal API.
Use Mach `free + inactive` as this feature's ONE live host-available sample in
both native inference and registry policy telemetry. Preserve existing generic
sysinfo RAM display separately; never derive policy headroom from its used bytes.
Keep Metal allocated bytes in the additive policy block, NOT the legacy
`mold_used_bytes` field, whose population activates CUDA attribution/cache logic.

| Consumer                                     | Budget and accounting                                                                                                            |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Hardware inventory                           | Installed RAM remains total hardware memory                                                                                      |
| Registry/scheduler                           | Policy incremental headroom plus bounded reclaimable cache credit, capped by current policy total; no CUDA attribution inference |
| Terminal feasibility classifier              | Current policy total, never worker startup RAM                                                                                   |
| `free_vram_bytes` / `usable_free_vram_bytes` | Incremental policy headroom; existing optional user reserve is subtracted once                                                   |
| H3 stable admission                          | Whole-process policy total, preserving its separate host check                                                                   |
| Server pre-drop guard                        | Incremental plus known reclaimable bytes, capped by policy total                                                                 |
| Server post-drop guard                       | Synchronize/sweep pool, then fresh incremental policy; zero cache credit                                                         |
| LTX-2 adaptive residency                     | Incremental policy intersected with existing live host floor and dispatch grant                                                  |
| Local/chain/warm paths                       | Trace their device facts and revalidate before generation; eager/CPU options cannot bypass the Metal policy for GPU allocations  |

Pure tests must pin the credit cap; a 48 GiB injected host changes stable capacity
from 40 GiB to 37.44 GiB, while a 16 GiB host retains the existing 8 GiB floor.
Update affected `device.rs` tests, H3 plan fixtures and `variant_dependencies`
tests to use injected samples instead of depending on this developer's sysctl.

Use the memoized Candle device (`device::metal_device`) for samples so allocated
bytes refer to the device used for inference. Discovery must not recurse into
itself. Cache only device identity, not budget values. Server status handlers
consume background telemetry rather than opening Metal or waiting on GPU work.
Read the limit again at dispatch/post-drop and existing safe streaming boundaries.
Shrinking a limit stops admitting more work or reduces residency at a supported
boundary; do not asynchronously free buffers used by an active command buffer.

## Administrative interface

Use an explicit local `mold system metal-memory` command group:

- `status [--json]`: sample the local host; reports unsupported and unknown states.
- `set <MiB> [--persist]`: validate a nonzero integer, root privilege, sysctl
  support and RAM floor; record old value, write through fixed native sysctl,
  verify readback and report observed Metal recommendation. Never invoke sudo
  internally or read a password. Failure must not print success. Recommend
  restarting the inference process if the Metal recommendation remains stale.
- `reset [--persist]`: explicitly set automatic (zero); with persistence also
  remove only Mold's owned boot policy. Without persistence, report any boot
  policy that will reapply on restart.

Parse and handle this group before DB/config migrations and model initialization,
so running the administrative command as root cannot create root-owned user
state or accidentally launch inference. Follow the existing `Commands::Skill`
early-return precedent for the entire group including status. Do not load Config
or read MOLD_HOST at all; local scope is explicit in help/output.

Persistence uses a fixed label and path in `/Library/LaunchDaemons`, fixed
`/usr/sbin/sysctl` ProgramArguments, a decimal validated value and RunAtLoad.
No user-writable executable, shell interpolation, paths from environment, broad
sudoers entry, or persisted root Mold process. File is root:wheel, mode 0644;
reject symlinks and foreign contents. Use atomic installation, serialize Mold
administrative writers, preserve previous owned configuration on failure, and
report partial state if live and persistent updates cannot both be completed.
Validate plist with plutil. Prefer a boot-only policy loaded at next boot over
launchctl kickstart races and a daemon that continuously overwrites settings.
Reset removes the owned file, leaving unrelated system services intact.
For an already-bootstrapped owned job, reset also boots out exactly its fixed
label; classify the documented not-loaded result separately from real failures.
The validated maximum is min(u32::MAX MiB, RAM minus max(15% RAM, 8 GiB)); no
force/unsafe bypass is added. Automated tests never use real /Library paths.

## Milestones and acceptance criteria

- [x] M0: Review this plan with requested Claude Fable; resolve valid findings
      and record exact model and review evidence before implementation.
- [x] M1: TDD pure budget arithmetic and wire compatibility. Add native read-only
      snapshot with automatic/explicit/unsupported/error states. Verify read-only
      native results against sysctl and a lightweight Metal probe, without models.
- [x] M2: Route discovery capacity, free/incremental queries, stable large-model
      capacity, streaming and server reclaim/dispatch through shared policy. Audit
      ledger arithmetic to avoid double subtraction and preserve host/CUDA behavior.
      Test limit reduction, allocations above limit, cache release, stale positive
      override, unknown probes, saturated arithmetic and 16/48 GiB machines.
- [x] M3: Add local administrative CLI with injectable I/O for unprivileged
      tests. Test parsing, root rejection, unsupported OS/key, unsafe sizes, write
      failure, mismatched readback, reset and explicit local-host semantics.
- [x] M4: Add opt-in persistence with temporary-directory tests for atomic
      replace, permissions, foreign file/symlink refusal, reset, concurrent writers,
      rollback/partial failure. Verify generated plist with native plutil. Avoid
      mutating this machine's real kernel or boot configuration during automated tests.
- [x] M5: Expose additive host telemetry through existing server/device status
      authority and shared Studio contracts; render effective limit and available
      headroom in existing GPU/device details on affected clients. Older hosts omit
      the section; remote clients show host values, never local system values.
      Keep CLI/TUI/MCP host inspection consistent where they project that authority.
- [x] M6: Update one changelog fragment, README, CLAUDE.md, CLI skill renderer,
      website CLI/performance/API docs and affected app docs. All examples parse.
- [x] M7: Run scoped Nix tests/checks, CPU and Metal compilation, applicable local
      CI routes, read-only native research and CLI black-box UAT, rendered affected
      frontend UAT, independent
      final diff review and resolve findings. Publish/update one PR and wait for
      local checks on the final head. The user subsequently requested closing PR
      #1592 to stop CI usage: keep it closed, continue on the same pushed branch,
      and report completion without reopening or merging.

## Validation details and remaining uncertainty

Tests must fail before each behavior is implemented. Native privileged writes,
reboot persistence and large-model inference are separate controlled UAT: do
not claim them validated by mocks, read-only probes or unit tests. Before any
live limit change, inspect active GPU jobs and other consumers, record the old
value, use a safe test value and define restore-on-failure. A reboot is not part
of unattended validation. This implementation can ship with those explicit
validation limitations if privilege or an idle test host is unavailable.

Determine whether the existing Metal device refreshes its recommendation after
a sysctl change; do not recreate Candle devices while tensors are alive. Positive
raw override clamping handles decreases conservatively. For a reset or increase,
retain the lower observed recommendation until a fresh inference process reports
the new value. Feature-detect this sysctl; do not promise undocumented kernel ABI
availability across all macOS releases or mutate related low-watermark keys.

Maintain this checklist and PR evidence as milestones are committed/pushed.

## Plan review disposition

Claude Fable 5.1 (`claude-fable-5-1`) reviewed the original plan against
`c6b925473`; findings are recorded in `docs/metal-memory-policy-review.md`.
Accepted: real scheduler/terminal-classifier routing, one Mach live sample,
in-engine free accessor, eager discovery probe, separate allocation attribution,
pool sweep, per-site accounting table, deterministic fixtures, early CLI return,
unsigned sysctl bounds, bootout, lenient client telemetry and docs obligations.
Two recommendations are intentionally adapted: an actual failed Metal probe
stays fail-closed after eager discovery, rather than silently using RAM; and
all milestones stay in ONE PR as explicitly requested, with separate commits.
