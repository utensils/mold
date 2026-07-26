# Multi-GPU runtime, scheduler, and UI — implementation handoff

**Date:** 2026-07-25

**Status:** Architecture reviewed; ready to split into implementation phases

**Baseline:** `origin/main` at `7cdb6aeefc5b`

**Planning branch:** `feature/multi-gpu-scheduler-v2`

**Scope:** Make every CUDA device visible to Mold a first-class, schedulable
resource; keep one-device behavior correct; coordinate every GPU consumer;
surface all devices and planned work on every client; and preserve all existing
image, video, chain, placement, and failure semantics.

## 1. Executive decision

Mold will have one authoritative device registry and one scheduling authority.
All unstarted GPU work remains in a global, re-plannable queue until a worker
accepts a lease. Each enabled CUDA GPU or CUDA-visible MIG compute instance
gets one dedicated worker thread and at most one active Mold stage. Metal
remains a single schedulable device. A generation or chain stage uses one GPU;
there is no tensor or pipeline parallelism.

The planner is deterministic and work-conserving, with one deliberate
exception: it may leave a device idle for a bounded interval when waiting for
an already-warm compatible device is predicted to finish the work sooner than
cold-loading it on the idle device. That exception is capped, visible, and
subject to the starvation bound.

The work is not one mega-change. It is seven independently mergeable phases:

1. identity, telemetry, and a read-only device API;
2. the scheduler core and migration of ordinary GPU work;
3. dynamic device lifecycle and settings;
4. durable chain integration and concurrent chains;
5. learned estimates, planned queue APIs, and full client UI;
6. server-owned adaptive batch execution, beginning with the F0 cancellation
   and atomic-publication substrate before any adaptive dispatch;
7. sm86/sm100 distribution and documentation.

The local 2×RTX 3090 host is the real-hardware development and qualification
target. Synthetic inventories cover 0, 1, 2, 8, 16, and 64 devices, including
12 GiB budgets, B200-like devices, heterogeneous fleets, and MIG topology.
Real Lambda 8×B200, real 12 GiB, and real MIG qualification are explicitly
deferred and must not be launched automatically or represented as completed.
The architecture must nevertheless contain no 1- or 2-GPU assumptions.

## 2. Locked product decisions

These are implementation requirements, not open questions:

- All discovered, allowed GPUs are enabled by default.
- Any number of CUDA devices exposed by the CUDA runtime is supported by the
  data model and planner. Synthetic acceptance covers up to 64 devices.
- One generation, batch child, utility stage, upscale, or chain stage uses one
  GPU. No tensor parallelism, pipeline parallelism, NCCL, multi-node
  scheduling, or cross-GPU component placement.
- The default policy is Balanced SLA: user order and starvation safety are
  hard constraints, then the planner balances wait time, completion time,
  device utilization, and model-locality cost. There is no throughput/latency
  mode selector in V2.
- Each schedulable device has at most one active Mold GPU stage.
- CUDA supports multiple devices; Metal retains its existing single-device
  behavior. ROCm and mixed CUDA/Metal scheduling are out of scope.
- All ready, compatible devices should be used, subject to hard constraints,
  host-memory admission, CPU-only phases, and the bounded warm-model wait.
- Queue mutations trigger a 2-second sliding replan debounce with a 5-second
  maximum delay. Idle-device dispatch does not wait for that timer.
- The default bounded warm wait is 2 seconds. It is allowed only when the
  predicted warm completion is earlier than a cold start now.
- Running work is never preempted. Disable/drain waits for the current stage to
  finish.
- A hard-pinned job remains blocked when its device is disabled or unavailable;
  it is not silently moved.
- User queue order is a hard priority input. A ready job may be bypassed only
  for incompatibility, an explicit hard constraint, an active bounded warm
  wait, or to use a device the older job cannot use. Three younger starts is
  the starvation limit.
- Chain stages release their device at stage boundaries and carry sticky,
  non-binding affinity to the last device.
- CPU placement is automatic only under memory pressure and only for components
  with an explicit tested CPU implementation.
- Existing request, environment, and persisted per-model component placements
  are hard constraints. V2 starts applying persisted placement on the server,
  which currently saves but does not use it for generation.
- Text encoders and VAEs may move to CPU when their engine family supports it.
  Transformers remain on the assigned GPU except for existing block-level CPU
  offload mechanisms.
- LTX-2 Gemma uses the assigned GPU when feasible and otherwise CPU. The
  existing unreserved sibling-GPU fallback is removed.
- Every administrative client can enable or disable devices. Discord reports
  all devices but remains read-only.
- Device enablement is machine-wide, not profile-scoped. All-disabled is a
  valid maintenance mode with the server and administrative APIs still alive.
- Remote device mutations use the existing API-key regime. If an API key is
  configured it is required even for loopback requests.
- Queue and execution-estimate state is ordinary in-memory runtime state.
  Desired device enablement and learned estimates persist.
- Planned queue lanes, blocked reasons, ETAs, and confidence are visible on
  every interactive client, with compact renderings allowed on iPhone and TUI.
- Existing client-generated batches remain independent sibling requests during
  phases A–E. Raw server `batch_size > 1` is not redefined until phase F.
- Phase F batch parents preserve ordered `base_seed + index` semantics, fail
  logically atomically, and publish no partial parent result.
- Existing model families, quantizations, image/edit/source/LoRA paths, videos,
  generated audio, durable chains, legacy chain shims, chained videos,
  retakes, and prepared expansion batches must not regress.
- Official sm86 and sm100 server/CLI and container coverage are added, without
  removing current compatibility paths until migration gates pass.
- Scheduler timing controls are exposed under Advanced settings. Objective
  weights and internal scoring are not user-facing settings.
- `mold run --local` uses the shared scheduler for multi-item work; it does not
  grow a second placement policy.

## 3. Current-state diagnosis

The inspected main branch already discovers and reports both local RTX 3090s.
The regression is therefore not simply “CUDA only sees device 0” on the
development host. Effective multi-GPU support is fragmented across dispatch,
chains, utilities, telemetry, placement, and clients.

### 3.1 Backend facts

- `crates/mold-server/src/queue.rs::run_queue_dispatcher` takes work from a
  global lookahead and irreversibly `try_send`s it into a per-GPU
  `sync_channel(2)`. Once sent, unstarted work cannot participate in a later
  global replan.
- `DEFAULT_LOOKAHEAD_BUFFER = 8` and `DEFAULT_MAX_DEFERRALS = 3` already
  implement bounded warm-model deferral. V2 must preserve the economic intent,
  not only the numeric starvation limit.
- `crates/mold-server/src/gpu_pool.rs` creates one worker per selected GPU, but
  `GpuPool.workers` is a fixed `Vec`; runtime enable, drain, disable, and
  replacement do not exist.
- Ordinary dispatch, durable chain stages, administrative model loads,
  standalone upscalers, and prompt expansion use separate GPU-acquisition
  policies. They can disagree and double-book resources because no common
  lease exists.
- `ProductionStageExecutor` in
  `crates/mold-server/src/chain_job_runner.rs` selects a worker directly and
  bypasses `JobRegistry`. The current chain runner executes one chain at a
  time.
- Running ordinary generations cannot be cooperatively cancelled between
  denoise steps because `mold_inference::progress::ProgressCallback` returns
  `()`. Durable chain stages have a separate break-capable path.
- Fatal CUDA errors set a process-wide flag and terminate the server for
  supervision. This is intentional and remains load-bearing.
- `crates/mold-server/src/resources.rs::physical_ordinal_for_worker` maps
  numeric `CUDA_VISIBLE_DEVICES` entries but cannot correctly join UUID or MIG
  selectors.
- `resolve_explicit_placement_gpu` accepts one ordinal and rejects multiple
  ordinals. That matches the one-device-per-stage decision.
- LTX-2 Gemma and prompt expansion can currently choose sibling GPUs without a
  reservation.
- Linux `available_system_memory_bytes()` returns `None`, so existing host-RAM
  preflight is not a Linux-wide admission system.
- No official CUDA artifact currently enables the optional server `nvml`
  feature.

### 3.2 Batch facts

- Desktop, CLI, TUI, web/mobile flows create sibling requests and force each
  request to `batch_size = 1`.
- Sibling seeds use `base_seed.wrapping_add(index)`.
- `batch_id`, one-based `batch_index`, and `batch_count` already flow through
  request and output metadata.
- Inference pipelines return one image; queue paths consume the primary image.
  A raw API request with `batch_size > 1` does not currently provide a real
  server-side batch.
- Gallery writes and events are per-output and immediate. There is no parent
  staging or atomic publish mechanism.

### 3.3 Client facts

Some views already render multiple devices, but important paths still use
`gpus[0]` or legacy `gpu_info`, including compact host cards/popovers, remote
host capability ranking, TUI status, Discord status, and parts of mobile.
Queue and status types are duplicated across `mold-core`, `studio`, `web`, and
`desktop`; this work must migrate shared browser-safe contracts into `studio`
instead of adding another divergent copy.

### 3.4 Distribution facts

Current release tarballs target sm89 and sm120; the container matrix includes
sm80, sm89, sm90, and sm120; Nix exposes its current default and sm120; the AUR
binary package is tied to sm89. Documentation routes RTX 3090 users
inconsistently between sm80 and sm89/PTX-JIT paths.

The stale `multi-gpu` branch is hundreds of commits behind main. It is useful
only as history for the worker-per-device migration. It must not be rebased or
copied: among other obsolete assumptions, it proposed resetting CUDA primary
contexts in-process, which violates the current fatal-CUDA safety invariant.

### 3.5 Required regression baseline

Do not start V2 by assuming the recent “only one GPU works” report has one
cause. Before Phase A changes:

1. build and run current `origin/main` on the local 2×3090 host;
2. record `CUDA_VISIBLE_DEVICES`, `MOLD_GPUS`, CLI flags, `nvidia-smi -L`,
   `/api/status`, `/api/resources`, `/api/queue`, `mold ps`, and every client
   summary;
3. enqueue at least four same-model jobs, four mixed-model jobs, one upscale,
   one expanded prompt, one video, and two durable chains;
4. record which GPU each operation actually uses and which surfaces display;
5. capture warm/cold load, throughput, eligible-idle time, and queue placement;
6. add failing tests for each reproduced defect before changing code;
7. bisect only defects that reproduce on current main, using the smallest
   source/API/client symptom as the bisect oracle.

If current main uses both GPUs for ordinary jobs but a client shows one, record
that as a client/status regression. If chains or utilities serialize while
ordinary jobs spread, record that separately. The V2 architecture fixes the
shared causes, but the implementation issue/PR must retain each concrete
regression and its test so “scheduler rewrite” does not replace diagnosis.

## 4. Goals and non-goals

### 4.1 Goals

1. Detect every CUDA device the runtime exposes, preserve stable identity, and
   enable all allowed devices by default.
2. Keep every unstarted work unit globally reassignable.
3. Coordinate every Mold GPU consumer through the same lease authority.
4. Optimize completion time and SLA behavior across homogeneous and
   heterogeneous devices without starving old work.
5. Preserve model locality without allowing locality to strand hardware
   indefinitely.
6. Make device state, telemetry, queue placement, and control correct on every
   client.
7. Support dynamic drain/disable/re-enable without interrupting an active
   stage.
8. Make memory placement request-aware and coordinate host-RAM use across
   workers.
9. Integrate image, video, upscaling, durable chains, chained videos, retakes,
   prepared batches, and later server-owned batches.
10. Keep single-GPU and CPU-only behavior correct and inexpensive.
11. Make the planner independent of device count and test it at 64 devices.

### 4.2 Non-goals

- Splitting one denoising transformer across devices.
- Running a text encoder on an unleased sibling GPU.
- Managing MIG mode, creating GPU instances, or changing MIG profiles.
- Hiding or bypassing `CUDA_VISIBLE_DEVICES`.
- Preempting a running CUDA kernel or denoise loop to improve queue order.
- Multiple active Mold stages on one non-MIG visible device or on the same MIG
  instance. Distinct CUDA-visible MIG instances on one physical parent may
  each hold one lease.
- Multi-node scheduling inside one Mold server.
- Cloud provisioning or Lambda qualification during this implementation.
- Guaranteeing identical pixels across different execution-plan fingerprints,
  GPU architectures, attention kernels, or precision variants.
- Replacing the multi-host routing authority in web/desktop/iPhone; host
  selection remains above each server-local scheduler.

## 5. Terminology and invariants

### 5.1 Terms

- **Physical GPU:** an NVIDIA hardware device.
- **MIG:** NVIDIA Multi-Instance GPU, which partitions supported data-center
  GPUs into isolated compute/memory instances.
- **Visible device:** a device returned by the CUDA runtime after
  `CUDA_VISIBLE_DEVICES` and driver enumeration rules apply.
- **Stable device ID:** an opaque backend-qualified UUID stored and sent over
  APIs. Ordinals are not stable IDs.
- **Worker:** the dedicated OS thread, context, model cache, and command slot
  for one visible device.
- **Work unit:** the smallest scheduler-owned operation that needs a device,
  such as a generation, chain stage, upscale, or admin load.
- **Lease:** exclusive permission for one worker to execute one active work
  unit.
- **Parent:** a user-visible operation that may produce one or more work units,
  such as a chain or future server batch.
- **Ready:** all dependencies are satisfied and a work unit can be admitted on
  at least one device/placement.
- **Warm:** the exact resolved model/execution fingerprint is resident or
  cheaply reloadable on a worker.
- **Plan:** a versioned, tentative assignment/order/ETA snapshot. A plan is not
  a lease.

### 5.2 Safety invariants

1. A visible device has at most one active Mold lease.
2. A work unit has at most one active lease.
3. Only the owning worker thread creates, uses, unloads, or drops CUDA-backed
   engines for its device.
4. No unstarted work lives in a per-device FIFO.
5. A plan cannot start work after its version becomes stale.
6. A runtime component may use only the assigned device or an admitted CPU
   placement. It may not discover and use a sibling GPU on its own.
7. Mold never resets a CUDA primary context in-process; retained Candle/cudarc
   handles make that unsafe. A fatal CUDA classification stops the server/app
   for supervision, and the context is never reused.
8. Desired enablement may persist; health, activity, leases, and poison state
   do not.
9. `CUDA_VISIBLE_DEVICES` is a hard security/visibility boundary. Mold neither
   schedules nor exposes devices outside it.
10. Legacy API fields retain their existing meaning. New semantics use additive
    fields.

### 5.3 Work-conservation invariant

Let:

- `D` be enabled, healthy, idle devices;
- `W` be ready GPU work units;
- `edge_feasible(w, d)` include model support, hard pins, resolved placement,
  per-device VRAM, device health, and backend constraints;
- `host_delta(w, d)` be the incremental host-RAM reservation of that concrete
  placement;
- `M*` be the priority-preserving, aggregate-resource-admissible matching
  defined in §9.3.

Mold should hold `|M*|` active leases. `M*` never bypasses an older
individually admissible unit merely to admit more smaller units, but it
globally rematches already-admitted units so a flexible older job cannot
strand a specialist. If an idle device and ready work unit are compatible,
assignment begins within 250 ms p95 inside the documented queue-size envelope
unless one of these explicit conditions applies:

- dependency or source artifact is not ready;
- hard pin or backend requirement blocks the device;
- device is disabled, draining, unavailable, degraded, or process-poisoned;
- aggregate host-RAM admission blocks the resolved placement after
  reservations for older admitted work;
- the work unit is in a CPU-only phase;
- queue is globally paused;
- the work unit is inside a bounded warm wait that predicts an earlier
  completion than a cold start now.

The warm exception expires at the earliest of:

- its predicted warm-device availability;
- `scheduler.warm_wait_max_ms` after the hold began;
- the point at which cold-start-now becomes the earlier predicted finish;
- the job reaching the three-bypass limit.

An exception and its reason must be present in the queue plan and metrics.

## 6. Architecture

```text
HTTP / CLI / desktop embedded / chain runner
                    │ normalized parent requests
                    ▼
          SchedulerCoordinator (runtime)
    queue + dependencies + timers + leases + events
                    │ snapshot
                    ▼
       mold-scheduler (pure deterministic crate)
    candidates + estimates + constraints → versioned plan
                    │ immediate lease grant
                    ▼
       DeviceRegistry ───────────────────────────────┐
        │                │                │           │
        ▼                ▼                ▼           ▼
 CUDA worker A     CUDA worker B      MIG worker C   Metal worker
 rendezvous slot   rendezvous slot    rendezvous     rendezvous
 context/cache     context/cache      context/cache  device/cache
        │                │                │           │
        └──────── completion / telemetry / residency ┘
```

### 6.1 Crate and module boundary

Add `crates/mold-scheduler` (`mold-ai-scheduler`) as a pure Rust crate:

- no Axum, Tokio runtime, CUDA, Metal, SQLite, filesystem, or platform APIs;
- depends only on `mold-core` wire/domain primitives where appropriate;
- owns planner inputs/outputs, eligibility matching, fairness counters,
  deterministic tie-breaking, estimate-key normalization, and synthetic
  simulation;
- exposes traits/value objects for clocks, device snapshots, work snapshots,
  and estimate lookup;
- is directly testable with 0–64 fake devices.

`crates/mold-server/src/scheduler/` owns `SchedulerCoordinator`, runtime work
state, timers, lease grants, worker messages, SSE, persistence adapters, and
conversion to public API types.

`mold run --local` constructs the same pure planner plus a CLI runtime adapter.
It must not depend on Axum or import server route state. Single-item local runs
may take the fast path through a one-unit coordinator, but placement and
admission still come from the shared planner.

### 6.2 Worker command protocol

Replace depth-2 per-GPU job queues with a rendezvous-style protocol:

1. an idle worker publishes `Ready { device_id, generation }`;
2. the coordinator grants exactly one `LeaseGrant`;
3. the worker acknowledges the matching plan/lease generation before it may
   touch CUDA;
4. completion returns timings, residency, memory high-water, output or error,
   and the worker becomes ready again.

Use one `std::sync::mpsc::sync_channel(1)` carrying `LeaseGrant` per worker.
The coordinator may send only after consuming that worker generation's
`Ready` event, and the worker blocks waiting for the matching generation.
Capacity one transports the acknowledged grant; it is never a job buffer and
may not receive a follow-up before the worker publishes a new `Ready`.
Worker-to-coordinator ready/completion events use a Tokio unbounded sender.
`in_flight` becomes derived from leases, not an independently maintained
placement counter.

### 6.3 CUDA thread ownership

Each CUDA visible device has one dedicated OS thread:

- create the device context on that thread;
- load and run engines on that thread;
- perform cache transitions and device-backed drops on that thread;
- do not hand CUDA objects to Tokio’s general blocking pool;
- do not reset a primary context during disable, unload, OOM recovery, or
  fatal-error handling.

Dynamic disable finishes the current stage, rejects further grants, unloads
and drops cached CUDA objects on the owner thread, acknowledges shutdown, and
joins the thread. Dynamic enable starts a fresh owner thread and context. A
failed start produces `health=unavailable` and leaves desired enablement true
so an administrator can retry.

## 7. Device registry, identity, and lifecycle

### 7.1 Authoritative registry

One `DeviceRegistry` supplies discovery, worker construction, scheduling,
telemetry joins, status, settings, API mutation, and client display. Remove
parallel ordinal-only inventories.

Suggested internal record:

```rust
pub struct DeviceRecord {
    pub id: DeviceId,
    pub backend: Backend,
    pub visible_ordinal: Option<u32>,
    pub raw_cuda_uuid: Option<[u8; 16]>,
    pub device_kind: DeviceKind, // FullGpu | Mig | UnknownCuda | Metal
    pub nvml_uuid: Option<String>,
    pub physical_uuid: Option<String>,
    pub mig_uuid: Option<String>,
    pub mig_parent_uuid: Option<String>,
    pub mig_profile: Option<String>,
    pub pci_bus_id: Option<String>,
    pub name: String,
    pub compute_capability: Option<(u16, u16)>,
    pub total_memory_bytes: Option<u64>,
    pub desired_enabled: bool,
    pub startup_allowed: bool,
    pub admin_state: AdminState,
    pub health: HealthState,
    pub activity: ActivityState,
}
```

Public stable IDs are opaque strings:

- every CUDA device: `cuda:<32-lowercase-hex-digits>` derived directly from
  the 16 bytes returned by `cuDeviceGetUuid_v2`;
- Metal: `metal:default` while Mold supports one Metal device.

Clients must URL-encode IDs and never parse their components. The backend
prefix prevents cross-backend collisions. The CUDA ID deliberately does not
embed `GPU-` or `MIG-`: the driver API returns binary UUID bytes, while the
textual type prefix requires separate classification. This keeps identity
stable even when NVML is unavailable.

### 7.2 CUDA discovery

For every ordinal returned by CUDA after `CUDA_VISIBLE_DEVICES`:

1. initialize CUDA and obtain the `CUdevice`;
2. call `cuDeviceGetUuid_v2` (through the locked cudarc API);
3. derive the opaque stable ID from those exact 16 bytes;
4. collect device name, total memory, compute capability, and PCI attributes;
5. query CUDA GPU-instance/compute-instance attributes when supported to
   classify `FullGpu` versus `Mig`;
6. when NVML is present, try its typed full-GPU/MIG UUID lookup, validate that
   the resulting binary UUID matches CUDA, and join telemetry by UUID, never
   by ordinal;
7. if the device is a MIG compute instance, resolve its allowed parent
   metadata through NVML;
8. if neither CUDA attributes nor NVML can prove the type, retain
   `device_kind=UnknownCuda`, leave MIG metadata null, and schedule it using
   its CUDA-advertised capabilities;
9. build one record and one possible worker.

`cuDeviceGetUuid_v2` returns a unique MIG compute-instance UUID when the CUDA
device is in MIG mode. If UUID lookup fails, the device is visible but
`health=unavailable`; Mold must not fall back to persisting an ordinal identity.
No-NVML tests must cover both a proven MIG classification and
`UnknownCuda` fallback without changing the stable ID.

### 7.3 `CUDA_VISIBLE_DEVICES` and MIG

NVIDIA documents that `CUDA_VISIBLE_DEVICES` accepts indices, GPU UUIDs, and
MIG identifiers and remaps visible ordinals. Mold therefore treats ordinals
as a process-local display value only.

MIG rules:

- Mold does not enable MIG mode, create GPU instances, choose profiles, reset
  hardware, or require privileged device nodes.
- Mold creates workers only for devices CUDA actually enumerates.
- NVML devices outside `CUDA_VISIBLE_DEVICES` are not exposed by Mold.
- `MOLD_GPUS`/`--gpus` can further restrict the visible CUDA inventory but
  cannot expand it.
- Multiple MIG instances depend on driver/runtime enumeration rules. With
  current R570-era behavior, CUDA supports one compute instance per GPU
  instance and at most 64 MIG instances across GPUs. Older drivers may expose
  only one MIG device to a process.
- The local GeForce RTX 3090s are not MIG-capable, so they cannot provide real
  MIG qualification.
- No scheduling decision assumes P2P access between MIG instances.
- A MIG instance is scheduled using its own advertised memory and telemetry,
  not its parent GPU’s full memory.
- A physical parent containing multiple visible MIG workers may show aggregate
  telemetry, but capacity remains per instance.

### 7.4 Startup selection

Precedence remains CLI > environment > file config > all visible:

- `--gpus` and `MOLD_GPUS` accept stable IDs or legacy ordinals;
- numeric ordinals are resolved against the current visible startup inventory;
- Mold stable-ID prefixes and NVIDIA `GPU-...`/current `MIG-...` UUIDs are
  accepted only when unambiguous and expanded to the full stable ID;
- invalid or ambiguous selectors fail startup with the visible choices;
- missing, empty, or `all` retains the existing “all visible” behavior;
- the explicit token `none` selects maintenance mode with no workers.

Evolve `GpuSelection` to distinct `All | None | Specific(Vec<GpuSelector>)`.
CLI/environment parsing follows the tokens above. The TOML `gpus` field uses a
backward-compatible untagged representation: legacy numeric arrays keep
working, including `gpus = []` meaning All; `"all"` and `"none"` are explicit
keywords; string arrays hold stable/NVIDIA UUID selectors. Existing ordinal
config remains a startup allowlist and is not copied into machine-wide
enablement preferences. There is no prior persisted per-device enable/disable
state to migrate.

### 7.5 Persisted desired enablement

Add a machine-wide table in DB migration v12 at this baseline (renumber after
rebase if another migration lands first):

```sql
CREATE TABLE device_preferences (
    device_id       TEXT PRIMARY KEY,
    desired_enabled INTEGER NOT NULL CHECK (desired_enabled IN (0, 1)),
    updated_at      INTEGER NOT NULL
);
```

Rules:

- a newly seen allowed device defaults to enabled;
- only explicit user changes create/update a row;
- absent devices retain their preference for later return;
- `MOLD_DB_DISABLE=1` uses enabled-by-default, in-memory preferences and logs
  that changes will not persist;
- startup allowlist exclusions win over desired enablement and render as
  `startup_excluded`;
- health/activity/poison/drain state never persists;
- DB corruption recovery follows existing Mold DB behavior and therefore
  falls back to enabled-by-default after quarantine/recreation.

### 7.6 State machines

Administrative state:

```text
startup_excluded
enabled ── disable ──▶ draining ── current lease ends ──▶ disabled
   ▲                       │                                  │
   └──── enable ───────────┴──────── enable/start worker ─────┘
```

Health:

```text
healthy ── repeated recoverable failures ──▶ degraded
   ▲                     cooldown/probe          │
   └─────────────────────────────────────────────┘

any ── fatal CUDA ──▶ poisoned (transient) ──▶ process teardown
any ── discovery/start/device loss ──▶ unavailable
```

Activity is orthogonal: `idle | loading | generating | upscaling |
admin_loading | stopping`.

Behavior:

- `enabled=false` removes future eligibility immediately.
- If a lease is active, state becomes `draining`; that stage finishes.
- No buffered follow-up exists, so drain completion is unambiguous.
- Unpinned queued work is replanned.
- Hard-pinned work stays blocked with `device_disabled` or
  `device_unavailable`.
- Enabling during drain cancels the pending disable before worker shutdown.
- Enabling a stopped device starts a new worker asynchronously.
- All-disabled keeps health/status/devices/settings/gallery/models APIs alive.
  New GPU-required work returns typed HTTP 503 `no_schedulable_device`.
- Explicit CPU-only correctness work may run only when the requested engine
  supports it; Mold must not silently move normal CUDA generation to CPU.
- `health=poisoned` is visible only during teardown/post-mortem events.
  Scheduler logic may only reject work after a fatal classification.

## 8. Work-unit and lease model

### 8.1 Work-unit schema

Every GPU consumer becomes a work unit:

```rust
pub struct WorkUnit {
    pub id: WorkId,
    pub parent_id: ParentId,
    pub kind: WorkKind,
    pub model_fingerprint: ModelFingerprint,
    pub execution_fingerprint: Option<ExecutionFingerprint>,
    pub request_shape: RequestShape,
    pub queue_rank: u64,
    pub priority_class: PriorityClass,
    pub bypass_count: u8,
    pub hard_device_id: Option<DeviceId>,
    pub backend_requirement: Option<Backend>,
    pub affinity: Option<Affinity>,
    pub dependencies: Vec<Dependency>,
    pub candidate_placements: Vec<CandidatePlacement>,
    pub resource_estimate: ResourceEstimate,
}
```

`WorkKind` includes:

- `Generation`;
- `PreparedSibling`;
- `ChainStage`;
- `PostUpscale`;
- `StandaloneUpscale`;
- `PromptExpansion`;
- `AdminModelLoad`;
- `BatchChild` (phase F).

CPU-only parsing, download, gallery, and catalog work does not acquire a GPU
lease. A multi-phase parent becomes ready work only when its dependencies are
ready.

### 8.2 What the lease covers

A generation lease covers the interval from the first assigned-GPU model
operation through the last assigned-GPU operation. CPU-placed text encoding
or VAE work may happen as part of the unit, but the device remains reserved
unless a later engine-specific optimization explicitly proves that releasing
and reacquiring is safe.

This conservative rule prevents:

- a second job evicting a transformer while CPU encoding is in progress;
- CUDA objects crossing worker ownership;
- a job silently reacquiring a different device with a different execution
  plan;
- false “idle” telemetry while a reserved generation is in a CPU phase.

The queue reports `activity_phase=cpu` so an idle-looking GPU during that
bounded phase is explainable.

### 8.3 Prompt expansion

Prompt expansion currently runs in the HTTP handler before queueing and picks
its own device. Under V2:

1. normalize the generation request;
2. if expansion is needed, create a `PromptExpansion` pre-stage;
3. enqueue the parent immediately and return the normal job identity/stream;
4. schedule expansion on CPU or the parent’s admitted GPU plan;
5. freeze the expanded prompt according to existing prepared-expansion
   invariants;
6. release the utility lease and make the generation stage ready.

No HTTP handler waits for scheduler admission. Expansion model downloads and
prepared-batch route ownership retain their existing host/frozen-route
semantics.

### 8.4 Upscaling

Standalone and post-generation upscalers use leases. A post-upscale is a child
work unit whose source output is private until the parent’s requested final
output is ready. It may run on another eligible device after the generation
lease ends, but never concurrently uses two devices for one stage.

Administrative loads also use leases so they cannot race generation. They
have lower default priority than user-visible generation but honor explicit
administrator hard pins.

## 9. Scheduler and replanning

### 9.1 Event model

The coordinator marks the plan dirty on:

- enqueue or dependency readiness;
- cancel;
- manual reorder;
- priority or hard-pin change;
- pause/resume;
- device enable, drain, disable, start, health, or loss;
- worker ready/completion;
- model residency or cache change;
- chain-stage completion/readiness;
- batch-child readiness/completion;
- material estimate update;
- host-RAM reservation release.

Every mutation increments a monotonic `state_version`.

### 9.2 Timers

Advanced settings:

| Key                             | Default |   Range | Meaning                                        |
| ------------------------------- | ------: | ------: | ---------------------------------------------- |
| `scheduler.replan_debounce_ms`  |    2000 | 0–30000 | Sliding delay after the newest mutation        |
| `scheduler.replan_max_delay_ms` |    5000 | 0–30000 | Cap measured from the first unplanned mutation |
| `scheduler.warm_wait_max_ms`    |    2000 | 0–30000 | Maximum beneficial warm-model hold             |

Validation requires `replan_max_delay_ms >= replan_debounce_ms`. Settings are
profile-scoped user preferences because they alter scheduling behavior, not
device identity. Objective weights remain internal.

On the first dirty event:

1. record `dirty_since`;
2. schedule at `now + debounce`;
3. later events move the deadline to `min(now + debounce,
dirty_since + max_delay)`;
4. a completed plan clears the dirty epoch only if its input
   `state_version` still matches;
5. otherwise discard it and rerun.

### 9.3 Immediate assignment

When a worker becomes ready or ready work appears while a device is idle, run
an immediate deterministic admission/matching pass.

Maintain eligibility indexes incrementally when work, placement, device, or
residency state changes. The immediate pass reads those indexes for every idle
device and the entire ready set; the optimization horizon does not apply.
Correctness must be equivalent to rebuilding the full graph.

Each candidate edge carries its concrete placement and incremental host-RAM
reservation. Visit work in this hard order:

1. starvation-forced status;
2. priority class;
3. manual queue rank;
4. work ID.

For each work unit, tentatively add it to the admitted set and compute a
deterministic minimum-host-RAM matching for the whole admitted set, allowing
augmenting-path rematches. Accept it only if every admitted unit can be
matched to a different device and the aggregate reservation fits current
headroom. Continue until every opening is filled or the ready set is
exhausted. A unit that cannot be added because older reservations consume the
shared headroom is blocked with `aggregate_host_ram_reserved`; younger work
may use only resources left after those older reservations. Mold never drops
an older individually admissible unit merely because two smaller younger
units would increase raw cardinality.

Implement the accepted-set feasibility/minimum-memory step with deterministic
successive augmenting paths (including reverse edges for rematching), not a
greedy device walk or a device-count bitmask. Incremental indexes accelerate
candidate discovery but never change this result.

Global rematching is required so a flexible job cannot occupy the only device
available to a specialist job:

```text
A can use GPU 0 or 1
B can use GPU 0 only

valid maximum matching: A → GPU 1, B → GPU 0
invalid greedy result:  A → GPU 0, GPU 1 idle, B blocked
```

A ready item beyond optimization rank 200 must still participate in this pass.
Valid warm holds are removed only from the immediate edge set they are
intentionally declining; other jobs and devices still match.

For a cold idle-device candidate:

```text
cold_finish = now + cold_setup + predicted_run
warm_finish = predicted_warm_ready + warm_reload + predicted_run
```

Hold for warm only when:

- `warm_finish < cold_finish`;
- the warm device is compatible and has a credible availability estimate;
- the hold stays within `warm_wait_max_ms`;
- the work has fewer than three bypasses.

Otherwise dispatch cold now. Before any grant, atomically reserve the complete
selected matching against the host-RAM ledger and its sample/ledger generation.
A concurrent memory or state change makes the plan stale and restarts
admission; Mold never validates and grants edges one by one. The resulting
ordered admission vector and cardinality are hard floors for the plan:
locality/ETA optimization may rematch devices but may not drop an older
admitted unit, reduce immediate leases, or increase aggregate reservations
beyond the atomic reservation. An immediate lease does not freeze the rest of
the horizon; the debounced optimizer still replans all unstarted work.

The 250 ms p95 issue-to-lease SLO applies to the default queue and acceptance
envelopes through 10,000 ready units on 8 and 64 devices. Correctness and full
ready-set priority still apply above 10,000 when an administrator explicitly
configures a larger `--queue-size`/`MOLD_QUEUE_SIZE`, but latency is best
effort and emits `scheduler_ready_set_oversize` telemetry with size and
duration.

### 9.4 Planner horizon and budget

Default horizon:

```text
min(ready_work_count, max(64, 8 * schedulable_device_count), 200)
```

Include starvation-forced work even if it falls outside that window.

The matching pass always completes. It has no optimization-horizon or
wall-clock early exit. The improvement pass uses a deterministic operation
budget:

```text
clamp(
    64 * schedulable_device_count + 4 * optimization_horizon_length,
    512,
    8192
)
```

The pure planner:

1. computes the full-ready-set, priority-preserving, aggregate-admissible
   immediate matching from §9.3;
2. freezes its ordered admission vector, aggregate reservation, and
   cardinality as hard constraints;
3. selects the ETA/locality horizon and inserts starvation-forced work;
4. builds an earliest-finish deterministic seed consistent with the matching;
5. enumerates move, swap, reorder, and (phase F) split/merge candidates in
   stable order for exactly the operation budget or until candidates exhaust;
6. returns the best deterministic complete plan;
7. falls back to the priority/cardinality-preserving seed on watchdog expiry
   or internal error.

A 200 ms wall-clock watchdog protects the coordinator inside the documented
queue-size envelope, but it may return only the deterministic
priority/cardinality-preserving seed. It must not publish whichever
intermediate plan happened to exist when a preemption-dependent timer fired.
The serialized plan for one input snapshot is therefore independent of CPU
speed, scheduling, and clock injection.

This is the optimality contract:

- immediate safety, strict priority admission, and matching cardinality are
  exact for the observed snapshot;
- the later ETA/locality plan is the best deterministic plan found within the
  stated horizon and operation budget;
- the planner is never described as globally optimal for arbitrary future
  arrivals or unknown runtimes.

Small exhaustive-oracle tests prove the exact guarantees. Trace replay on 8-
and 64-device synthetic fleets compares the predictive plan with FIFO,
round-robin, least-busy, and the current dispatcher. “Optimal” support for an
8×B200 server means exact work conservation and priority under current facts
plus measured improvement against those baselines—not an unprovable
future-workload claim.

No native solver dependency is added.

### 9.5 Objective order

Balanced SLA is neither pure FIFO nor throughput-only. For planning purposes,
each ready user-visible parent gets a soft start deadline:

```text
soft_wait_budget = clamp(0.25 * predicted_parent_runtime, 5 s, 60 s)
soft_start_deadline = ready_at + soft_wait_budget
lateness = max(0, planned_start - soft_start_deadline)
```

Prompt expansion and post-upscale inherit their parent’s deadline. Durable
chain stages and prepared siblings are normal user work; administrative
preloads are lower priority and cannot consume an opening needed by ready user
work. V2 does not expose an arbitrary public priority knob.

The objective is lexicographic:

1. hard constraints and resource safety;
2. starvation-forced work, explicit priority classes, and manual queue order;
3. preserve the priority-admitted immediate matching cardinality;
4. three-bypass accounting for openings not already consumed by harder
   priority;
5. minimize predicted parent SLA lateness;
6. minimize predicted parent completion/makespan;
7. minimize sum of completion times;
8. avoid non-beneficial device idleness, using the warm-finish test above;
9. minimize cold loads, evictions, and model transfers;
10. minimize memory-risk penalty;
11. stable tie-break by queue rank, work ID, and device ID.

“No idle device” is not placed above predicted completion time because that
would force destructive cold loads. Every beneficial idle exception is
bounded and observable.

### 9.6 Queue order and starvation

Manual order changes operate on stable work/parent IDs. For two ready jobs in
the same priority class and compatible with the same opening, the older
queue-rank wins unless:

- it cannot run on that device;
- it is already assigned to an earlier-finishing device;
- it is in a valid warm wait;
- a younger job uses an otherwise idle device without delaying it.

Whenever a younger unit starts while an older ready unit could have used that
opening under already-honored aggregate reservations, increment the older
unit’s bypass count. At three, it must take the next compatible opening.
Incompatible device starts and starts using only host RAM left after older
reservations do not count as bypasses. Cancellation/requeue does not silently
reset the counter; a materially edited request becomes a new work unit.

### 9.7 Plan versioning

A plan contains:

- `plan_version`;
- input `state_version`;
- creation time and next replan deadline;
- per-device ordered lanes;
- immediate lease proposals;
- assignment reasons;
- start/finish estimates and confidence;
- warm-wait deadlines;
- blocked work and typed reasons.

Before granting a lease, the coordinator validates:

- plan and state versions;
- work still ready and uncancelled;
- worker generation still current and ready;
- device still enabled/healthy;
- the complete immediate matching’s atomic host-memory reservation still owns
  the current ledger generation.

Failure triggers an immediate replan; it is not converted into a best-effort
dispatch.

## 10. Placement, VRAM, and host RAM

### 10.1 Resolved execution plan

Normalize placement before generating candidates:

```text
request placement
    > MOLD_PLACE_* environment
    > persisted per-model placement
    > Auto
```

This deliberately makes the server honor `Config::resolved_placement()`;
today it persists placement but validates only request placement during
generation. Record the behavior correction in release notes.

Every non-`Auto` component value is a hard constraint:

- explicit CPU stays CPU even without pressure;
- legacy `{"kind":"gpu","ordinal":N}` resolves once to a stable device ID;
- add `{"kind":"device","id":"cuda:..."}` for durable stable-ID placement;
- an explicitly disabled/unavailable device blocks the request;
- component pins to different GPU IDs are rejected because V2 has no
  cross-GPU component execution;
- request, environment, and persisted conflicts resolve only through the
  precedence order above, never scheduler scoring.

For `Auto`, generate a GPU-resident plan first. A CPU component candidate may
exist only when no tested GPU-resident plan satisfies:

```text
static_peak_vram + configured_safety_headroom <= admissible_free_vram
```

or when a current measured pressure/OOM fact invalidates that plan. Auto never
chooses CPU merely to improve ETA, utilization, or model locality.

Replace model-name-only selection with concrete per-component plans:

```rust
pub struct ComponentExecutionPlan {
    pub role: ComponentRole,
    pub artifact_path: PathBuf,
    pub content_fingerprint: ContentFingerprint,
    pub dtype: Option<DType>,
    pub quantization: Option<QuantizationVariant>,
    pub placement: ResolvedComponentPlacement, // Cpu | assigned DeviceId
    pub load_strategy: ComponentLoadStrategy,
    pub predicted_vram_bytes: u64,
    pub predicted_host_bytes: u64,
}

pub struct EffectivePlacement {
    pub components: BTreeMap<ComponentRole, ResolvedComponentConstraint>,
}
```

`ComponentRole` covers the transformer, every named text encoder, VAE, audio
VAE/vocoder, upscaler, and concrete external LTX VAE/text-projection
companions. `ComponentLoadStrategy` covers resident, drop/reload, parked CPU,
streamed blocks, tiled VAE, and other family-declared strategies.

The request-aware aggregate is:

```rust
pub struct ResolvedExecutionPlan {
    pub device_id: DeviceId,
    pub model_fingerprint: ModelFingerprint,
    pub effective_placement: EffectivePlacement,
    pub components: BTreeMap<ComponentRole, ComponentExecutionPlan>,
    pub attention_backend: AttentionBackend,
    pub offload_mode: OffloadMode,
    pub predicted_vram_peak_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub determinism_class: DeterminismClass,
    pub execution_fingerprint: ExecutionFingerprint,
}
```

Resolve concrete encoder variants, quantization, external companions, and load
strategies before dispatch. `execution_fingerprint` hashes every component
artifact/content fingerprint, dtype/quantization, placement, load strategy,
attention backend, and offload mode.

The engine receives and consumes this exact plan. It may not independently
select another GPU, artifact, encoder variant, quantization, or placement. If
runtime facts invalidate it before CUDA work begins, return typed
`PlanInvalidated` with updated facts and replan. Any safe fallback must be an
explicit alternate plan selected by the coordinator; silent runtime fallback
is removed.

### 10.2 Preserve family behavior

The central planner must preserve:

- lazy model load;
- per-worker model caches and parked residency;
- drop-and-reload text encoders;
- quantized encoder fallback;
- FLUX block-level offload;
- VAE tiling;
- LoRA rebuilding/caching;
- streaming/reconstruction requirements;
- LTX-2 external VAE/text-projection/audio assets through every chain stage;
- LTX-2 Gemma CPU retry without moving the transformer or video VAE;
- quantized Qwen edit split CFG;
- CPU-seeded initial noise.

Each engine family supplies a capability descriptor rather than the scheduler
matching model-name substrings:

```rust
pub struct PlacementCapabilities {
    pub supports_text_encoder_cpu: bool,
    pub supports_vae_cpu: bool,
    pub supports_audio_components_cpu: bool,
    pub supports_block_offload: bool,
    pub supports_tiled_vae: bool,
    pub native_batch_sizes: Vec<u32>,
}
```

Unsupported CPU placement is infeasible, not an invitation to try and OOM.

### 10.3 Reset-free CUDA reclamation

Current main calls `reclaim_gpu_memory()` from worker load/unload/OOM paths,
model management, server cleanup, memory preflight, upscaling, and LTX-2; its
CUDA implementation invokes `cuDevicePrimaryCtxReset_v2`. Memory preflight
also assumes a reset restores total VRAM. Those assumptions must not survive
V2.

Phase B explicitly:

1. inventories and removes every normal runtime call to
   `reclaim_gpu_memory`;
2. removes the CUDA primary-context-reset implementation/export;
3. drops engines, tensors, caches, and device-backed objects on the owning
   worker thread;
4. synchronizes only through supported context/device operations that do not
   invalidate retained handles;
5. re-samples actual free VRAM after drops;
6. treats unrecovered/“ghost” VRAM as unavailable capacity or external
   pressure instead of assuming total capacity;
7. uses normal cooldown/replan for recoverable OOM and process teardown for
   fatal CUDA.

The gate is zero normal-runtime references to
`cuDevicePrimaryCtxReset_v2`, plus model-swap, unload, disable, preflight,
upscale, LTX-2, OOM, and fatal-error tests with live-handle detection.

### 10.4 LTX-2 sibling placement removal

Remove the unreserved sibling walk in `select_ltx2_gemma_device`. The
selection becomes:

1. assigned device if the full resolved plan fits;
2. CPU if the tested Gemma CPU plan fits host-RAM admission;
3. blocked/rejected with a typed memory reason.

This may make prompt encoding slower on a pressured multi-GPU machine. Record
it as an intentional correctness trade-off. A future short auxiliary lease
could restore safe sibling execution, but it is not part of this design.

### 10.5 Host-RAM admission

Use one cross-platform `sysinfo`-backed sampler for telemetry and admission.
Do not retain the current Linux `None` path.

Definitions:

```text
safety_floor = max(8 GiB, 15% of total physical RAM)
admission_headroom = saturating_sub(
    sample.available_bytes,
    safety_floor + bytes_accepted_after_sample_started
)
```

The sampler and reservation ledger use monotonic generations:

```rust
pub struct MemorySample {
    pub generation: u64,
    pub collection_started_sequence: u64,
    pub available_bytes: u64,
}

pub enum ReservationState {
    Reserved,
    CommittedAfterSample,
    ReflectedBySample,
    Released,
}
```

Admission, reserve, allocation-commit, release, and sample publication reduce
through one locked/actor-owned ledger:

1. accepting an immediate matching calls one
   `try_reserve_matching(plan_id, sample_generation, ledger_sequence, items)`
   operation; it either records every work-unit reservation and increments the
   monotonic ledger sequence once, or records none;
2. until allocation commits, its full bytes remain in
   `bytes_accepted_after_sample_started`;
3. commit does not release the charge;
4. a later sample may absorb the committed charge only when the allocation
   committed before that sample’s `collection_started_sequence`;
5. allocations committed during/after collection remain charged until the
   following sample;
6. uncommitted reservations are rebased to the new sample and remain charged;
7. released memory triggers an immediate sample but remains conservatively
   unavailable until a later sample observes it.

The ledger rejects a matching if its aggregate would exceed headroom even when
every item fits individually. A stale plan/sample/ledger generation returns a
typed retry, never a partially accepted subset. If any worker grant fails
after reservation, the coordinator releases every ungranted item from that
matching in the same reducer turn and replans before making another grant.

Track persistent Mold allocations (parked encoders, pinned buffers, chain
carries) by allocation ID for ownership, attribution, and release. Once a
sample is known to include a committed persistent allocation, do not subtract
it again: `available_bytes` already reflects committed Mold and non-Mold
memory. This generation protocol prevents both the commit-before-next-sample
over-admission race and permanent double counting.

Reservations cover:

- CPU text-encoder or VAE weights;
- `MOLD_KEEP_TE_RAM` parked encoders;
- pinned offload buffers;
- reconstructed/streamed transformer blocks;
- chain carry and encode buffers;
- batch staging/encoding buffers;
- download/load overlap where model loading materially allocates RAM.

If sampling is unavailable, use conservative static limits and refuse new
automatic CPU offload that cannot be bounded. Do not assume infinite RAM.

Host-RAM blocking is a first-class queue blocked reason. Releasing a
reservation triggers an immediate matching pass.

### 10.6 Estimates

Static estimates remain the safety floor. Learned data may improve scheduling
and ETAs but may never lower memory admission below a static engine estimate.

Observation key dimensions:

- stable device capability class, not only ordinal;
- model and component fingerprint;
- work kind;
- width/height;
- steps;
- frames/fps/audio;
- source/edit/LoRA flags;
- resolved precision/offload/placement;
- batch partition size.

Record:

- cold load, warm reload, prompt encode, denoise, VAE, upscale, and total time;
- Mold-observed VRAM high-water;
- incremental host-memory high-water when measurable;
- outcome and fallback/invalidated-plan reason.

Defaults:

- EWMA alpha `0.25`;
- winsorize a new sample to `0.25×..4×` the prior estimate;
- confidence `low < 3`, `medium 3..9`, `high >= 10`;
- lookup order: exact bucket → normalized family/device/shape bucket → static;
- keep at most 10,000 buckets;
- prune buckets unused for 180 days;
- under `MOLD_DB_DISABLE=1`, learn in memory for the process lifetime.

Persist in DB migration v13 at this baseline (renumber after rebase if needed):

```sql
CREATE TABLE scheduler_estimates (
    estimate_key                 TEXT PRIMARY KEY,
    device_class                TEXT NOT NULL,
    model_fingerprint           TEXT NOT NULL,
    work_kind                   TEXT NOT NULL,
    shape_bucket                TEXT NOT NULL,
    execution_fingerprint       TEXT NOT NULL,
    sample_count                INTEGER NOT NULL,
    ewma_total_ms               REAL NOT NULL,
    ewma_load_ms                REAL,
    vram_high_water_bytes       INTEGER,
    host_high_water_bytes       INTEGER,
    last_observed_at            INTEGER NOT NULL
);
```

## 11. Chains and chained video

### 11.1 Integration

The durable chain runner becomes a producer of `ChainStage` work units. It no
longer selects `GpuWorker` directly or maintains shadow in-flight guards.

For each chain:

1. validate and persist the normalized request and concrete component paths;
2. create only the next dependency-ready stage work unit;
3. scheduler grants one device lease;
4. render and checkpoint that stage;
5. release the lease;
6. publish the next stage with sticky affinity to the previous device;
7. finalize through admitted CPU/GPU work as applicable.

A chain has at most one active stage. Multiple different chains may run
concurrently on different devices.

### 11.2 Affinity and movement

Sticky affinity is a setup-cost preference, not a pin. A later stage may move
when:

- its concrete model and companion assets are available to the worker;
- the resolved execution plan is compatible;
- persisted carry artifacts can be loaded without changing semantics;
- moving predicts a better completion and does not violate queue order.

All transformer-only LTX-2 companion paths are frozen into the durable chain
plan and reused at every stage. Opaque catalog IDs are never reparsed to infer
pipeline/assets.

### 11.3 Queue compatibility

Internal chain stages enter the unified scheduler registry and new queue-plan
API. The legacy `/api/queue` projection excludes non-generation internal work
units so old clients’ position indices do not shift. Legacy position PATCH
operates on that same projection.

New clients reorder/pin by stable job/work IDs. They can render a chain parent
with its current stage without treating every internal checkpoint as an
independent user job.

### 11.4 Durability and cancellation

Preserve `chain_jobs`, `chain_job_stages`, manifests, checkpoint ordering,
resume, retake, and GC behavior.

Cancellation:

- queued stage: remove immediately;
- active stage: use the unified cooperative cancellation primitive at safe
  denoise/encode boundaries;
- if the engine cannot stop immediately, report `cancelling` and finish at the
  next safe point;
- completed stage artifacts remain resumable according to current durable
  semantics.

Acceptance includes two or more interrupted chains, restart reconcile, and
concurrent resume without cross-chain state corruption.

## 12. Batch behavior

### 12.1 Phases A–E: preserve client siblings

Desktop, web, iPhone, CLI, and TUI keep expanding Batch N into N independent
requests with:

- `batch_size = 1`;
- one frozen normalized route;
- `base_seed.wrapping_add(index)`;
- `batch_id`, one-based `batch_index`, and `batch_count`;
- one reviewed prompt per prepared sibling.

The scheduler recognizes sibling metadata for display and locality but each
sibling remains independently cancellable and independently publishable.
There is no parent atomicity retrofit in these phases.

Raw server `batch_size > 1` remains documented as unsupported until phase F.
Do not build adaptive behavior on the false assumption that engines already
produce N outputs.

### 12.2 Phase F0 — cancellation and transaction substrate

F0 is an independently mergeable prerequisite PR series before adaptive
partitioning:

1. change the engine progress/cancellation contract to allow cooperative
   cancellation at safe points across every inference family;
2. add attempt-generation-scoped staging, a parent manifest, disk-space
   preflight, gallery publication barrier, commit journal, startup recovery,
   and cleanup;
3. define `BatchExecutionCapability` for every engine family;
4. define a serialized parent completion reducer;
5. add attempt generations and late-completion fencing.

Parent state machine:

```text
queued → running → prepared → committing → committed
             │
             ├→ cancelling → fenced → cancelled
             └→ failing    → fenced → retrying | failed
```

Every child lease carries `(parent_id, child_index, attempt_generation)`.
Completion reduction is serialized by the parent actor/lock and accepts only
the current generation. A stale/late completion deletes its private artifact
and cannot mutate counters, publish, retry, or commit.

Cancel/failure closes the attempt to new grants, signals active children, and
waits until every active lease reaches a safe terminal acknowledgement before
cleanup, retry, or commit. Retry increments the attempt generation only after
the old generation is fenced. Every state transition is journaled with the
attempt generation; `prepared`, `committing`, and `committed` transitions use
atomic manifest replacement plus file/directory fsync. Crash recovery
reconstructs this state from the journal before any gallery route is served.

### 12.3 Parent and child model

A server batch parent lazily represents indices `0..N`; it does not
immediately allocate N images or N full requests.

For child index `i`:

```text
seed_i = base_seed.wrapping_add(i)
batch_index = i + 1
batch_count = N
```

The parent resolves model, companion assets, precision, attention backend,
offload strategy, component placement, output format, and determinism class
once. Candidate devices must admit that exact fingerprint.

Initial family capabilities may expose `native_batch_sizes = [1]`. That is
valid: the scheduler can distribute singleton children across devices without
pretending a pipeline supports native tensor batches. Larger partitions are
considered only for families with measured, tested native batch support.

Candidate partition sizes are capability-derived, not hard-coded
`1,2,4,...`. The planner compares parallel singleton distribution, native
microbatching, load duplication, VRAM, and predicted finish.

### 12.4 Determinism

The guaranteed contract is:

- child ordering and seeds match sequential client fan-out;
- output metadata/filenames retain the global child index;
- the same model, code version, resolved execution fingerprint, and
  determinism class preserve the existing exact-output guarantee;
- cross-device execution is allowed only when candidate devices share that
  fingerprint/class;
- otherwise keep the affected children on a compatible device and record why.

Do not claim bit-identical output across different GPU architectures,
attention kernels, quantization variants, or CPU/GPU placement strategies.
Every output records device ID and execution fingerprint for provenance.

### 12.5 Logical atomic publication

Strict filesystem+SQLite multi-file atomicity is impossible without a journal.
Implement logical API atomicity with one `GalleryPublicationGate`:

1. write each child only under
   `<resolved_output_dir>/.mold-batch-transactions/<parent_id>/attempts/<attempt_generation>/staging/`;
2. atomically write an attempt manifest containing normalized request,
   generation, ordered children, checksums, collision-safe reserved final
   names, and state;
3. reserve final names without overwrite; a collision is resolved before
   publication and the chosen names are frozen in the manifest;
4. after every child succeeds and the reducer fences further completions, mark
   and fsync the manifest `prepared`;
5. acquire the exclusive side of `GalleryPublicationGate`;
6. atomically persist and fsync `committing`, move every final file with
   no-replace semantics, and fsync the files and affected directories;
7. when the metadata DB is enabled, commit all gallery rows in one SQLite
   transaction; when `MOLD_DB_DISABLE=1`, the committed manifest plus final
   files is the durable authority and filesystem listing remains supported;
8. atomically persist and fsync `committed`;
9. release the publication gate;
10. only then emit one parent completion and ordered child metadata.

Every gallery observer and mutator participates in the same barrier: DB and
filesystem listings, media-token/path validation, media lookup/open, delete,
background reconciliation, and corruption recovery take the shared/read side
or serialize as a writer. A guessed media URL therefore cannot see a moved
child during commit. The server completes batch-manifest recovery before
binding/serving gallery routes, including when the DB is disabled or empty.
The transaction root is deliberately inside the resolved gallery filesystem
so every final no-replace rename stays on one filesystem; gallery scanners and
reconcile must ignore that reserved directory.

An error after entering `committing` does not release the writer barrier while
the live server keeps serving. The commit handler must either roll forward to
durable `committed`, roll back every final/DB change and durably mark the
attempt failed, or terminate the server so startup recovery runs before
serving. It may not leave a live process exposing an unresolved `committing`
transaction.

Recovery is generation-aware and idempotent:

- `staging`/`prepared` with no final moves may resume or roll back without any
  published result;
- `committing` is completed under the exclusive publication gate from the
  manifest/checksums; it is not exposed halfway;
- `committed` is never rolled back, even if the process died before emitting
  the ephemeral completion event; clients reconcile it from the gallery;
- a stale completion or cleanup may touch only its own
  `attempts/<attempt_generation>/` directory and may never overwrite/delete a
  retry’s files.

External direct filesystem observers remain outside the logical API guarantee.

On failure:

- validation/model errors fail immediately;
- one nonfatal device-local child failure may retry once on another
  fingerprint-compatible device;
- fatal CUDA remains process-fatal;
- terminal parent failure closes the current attempt, cancels queued children,
  requests cooperative cancellation for active children, and reaches `fenced`
  before cleanup;
- retry starts a new attempt generation only after the previous attempt is
  fenced;
- late completions from closed generations are ignored and only their
  generation-scoped private artifacts are removed;
- no gallery rows/events are published;
- staging artifacts are deleted after error reporting or retained only under
  an explicit debug-retention setting.

## 13. Public API

### 13.1 `GET /api/devices`

Add a stable device resource:

```json
{
  "devices": [
    {
      "id": "cuda:0123456789abcdef0123456789abcdef",
      "backend": "cuda",
      "ordinal": 0,
      "device_kind": "full_gpu",
      "nvml_uuid": "GPU-...",
      "physical_uuid": "GPU-...",
      "mig_uuid": null,
      "mig_parent_uuid": null,
      "mig_profile": null,
      "name": "NVIDIA GeForce RTX 3090",
      "pci_bus_id": "00000000:01:00.0",
      "compute_capability": "8.6",
      "memory": {
        "total_bytes": 25769803776,
        "used_bytes": 0,
        "mold_used_bytes": null,
        "other_used_bytes": null
      },
      "telemetry": {
        "utilization_percent": 0,
        "temperature_c": null,
        "power_w": null
      },
      "desired_enabled": true,
      "admin_state": "enabled",
      "health": "healthy",
      "activity": "idle",
      "schedulable": true,
      "unschedulable_reason": null,
      "loaded_models": [],
      "active_work_id": null,
      "planned_work_ids": []
    }
  ],
  "plan_version": 42
}
```

Operational telemetry is nullable. Unsupported values are `null`, not zero.
`mold_used_bytes` and `other_used_bytes` are best effort; do not infer exact
per-process memory when NVML/driver data cannot support it.

### 13.2 `PATCH /api/devices/{id}`

Request:

```json
{ "enabled": false }
```

Semantics:

- idempotent;
- `200` when the requested stable state is already reached;
- `202` with current state when draining or starting asynchronously;
- `404` unknown stable ID;
- `409` startup-excluded device cannot be enabled without restart;
- `503` worker start failed, with device remaining unavailable;
- enabling while draining cancels shutdown if the current worker still exists.

Authentication:

- when `MOLD_API_KEY` is configured, require it from every source, including
  loopback;
- when auth is disabled, the server is already open and loopback needs no new
  exemption;
- add `PATCH` to configured-origin CORS methods; existing `allow_headers=Any`
  must continue to admit `X-Api-Key`;
- classify device PATCH under the existing `RouteTier::Generation` quota
  rather than the unknown-route Read fallback;
- audit request ID, authenticated-key identity where available, stable device
  ID, old/new desired state, result, and remote address without logging keys.

### 13.3 Queue plan

Keep the existing `GET /api/queue` response and legacy `entries` projection,
then add an optional top-level `plan` object. `plan.work_items` contains
internal chain stages, utility work, and future batch children; legacy
`entries` excludes those internal units. The additive plan contains:

- `plan_version`, `state_version`;
- `optimizer_state`, `dirty_since`, `next_replan_at`;
- stable `device_id` plus legacy ordinal `gpu`;
- `hard_pinned_device_id` plus legacy `target_gpu`;
- `planned_device_id`;
- lane and planned order;
- estimated start/finish and confidence;
- assignment/warm-wait/blocked reason;
- priority class and queue rank;
- bypass count;
- parent/work kind;
- chain stage;
- planned batch partitions;
- activity phase.

Preserve legacy `state`, `position`, `gpu`, and `target_gpu`. Preserve
the current omission of `target_gpu` once running for old clients; do not
change it to explicit JSON `null`. Retain the new stable pin field separately
for new UIs.

New queue mutation APIs operate by stable job/work ID. Continue accepting
legacy ordinal pins. If both stable ID and ordinal are supplied and resolve to
different devices, return typed 422.

### 13.4 Events and capabilities

Add:

- `queue_plan_changed`;
- `device_state_changed`;
- optional `device_id`, `execution_fingerprint`, and batch partition metadata
  to progress/completion events;
- capabilities indicating device API, stable pins, planned lanes, lifecycle,
  learned ETA, cooperative cancellation, and server-batch availability.

Old clients ignore additive fields and continue using the legacy queue/status
projection.

### 13.5 Legacy status

Keep `status.gpus` and `gpu_info` shape-compatible:

- derive both from the registry’s 1 Hz telemetry snapshot;
- stop spawning `nvidia-smi` for each status request;
- preserve `gpu_info` as the first visible legacy summary only;
- new clients must never use it for multi-GPU routing or display.

## 14. Client requirements

### 14.1 Shared Studio contract

Move browser-safe device and queue-plan types/fetchers into `studio`:

- `DeviceInfo`;
- `DeviceState`;
- `QueuePlan`;
- `QueueWorkItem`;
- device list and mutation helpers using explicit `ApiTarget`;
- event reducers for plan/device changes.

Web, desktop, and iPhone consume the shared types. Do not import desktop
primary-host state into `studio` or mobile.

### 14.2 Common UX

Every interactive client shows:

- all visible devices;
- name, stable short ID, ordinal as secondary display, backend/MIG badge;
- enabled/draining/disabled/unavailable/degraded state;
- per-device VRAM and utilization;
- loaded model and active stage;
- planned lane and blocked jobs;
- tentative ETA/confidence;
- replan countdown while dirty;
- enable/disable toggle in Advanced Settings and Machine detail.

Rules:

- all toggles start on for newly discovered devices;
- toggling off an active device says “Finishing current work” and disables new
  assignments immediately;
- toggling back on during drain cancels the pending shutdown when possible;
- disabling the last device enters maintenance mode after current work ends;
- hard-pinned blocked jobs name the disabled device and offer unpin/re-enable;
- a single-device host keeps a compact layout;
- ordinals are display hints, never durable selection values;
- tentative plans are labeled as such and update without disruptive toasts.

### 14.3 Web

Update:

- Machines cards and host detail;
- `HostCard`/compact status paths that currently use `gpus[0]`;
- queue cards/lanes;
- Settings Advanced device panel;
- browser-side multi-host routing.

The origin host and every connected host use explicit targets and their own API
keys. Host routing ranks:

1. hosts that have the selected model/companions;
2. hosts with at least one feasible schedulable device;
3. predicted completion using queue plan and device capabilities;
4. setup/load cost;
5. stable host tie-break.

Do not use first-GPU VRAM as host capability.

### 14.4 Desktop

Update:

- `StatusPopover`;
- Machines list and `HostDetailView`;
- host store/capability ranking;
- queue drawer and per-host lanes;
- Advanced Settings.

The embedded This Device server uses the same HTTP/SSE contract. Remote hosts
remain separate entries; device enablement always targets the owning server.
All connected/remembered hosts keep their current reconnect and identity
rules.

### 14.5 iPhone

Add the same device panel to Machines host detail and Advanced Settings using
shared explicit-target helpers. Per-host API keys remain in Keychain and never
enter URLs. Compact queue rows may collapse lane details, but must show device,
blocked reason, ETA confidence, and drain state.

Prepared expansion and missing-model recovery keep their frozen host URL/key/
instance identity. A device replan inside that host does not mutate the frozen
host route.

### 14.6 TUI

Replace `gpu_info`-only summaries with a device list and selected detail. Add:

- all-device utilization/VRAM;
- queue lanes and blocked reason;
- enable/disable action with draining feedback;
- compact replan countdown.

Keep existing key→action contracts, layout const assertions, gallery thumbnail
protocol, and Create Advanced accordion invariants.

### 14.7 CLI

Add:

```text
mold gpu list [--json]
mold gpu disable <stable-id-or-ordinal>
mold gpu enable <stable-id-or-ordinal>
```

`mold ps`, `mold server status`, and MCP status display every device. JSON uses
stable IDs. Human output may show ordinals.

Forced-local multi-item generation uses the shared scheduler adapter. A single
local request remains simple and must not incur a material startup regression.

### 14.8 Discord

Discord is read-only:

- report all devices and aggregate queue status;
- show the actual completion device when useful;
- do not expose enable/disable commands.

## 15. Telemetry

Enable `mold-server/nvml` in official CUDA features and forward it through
`mold-cli`. Add it consistently to Cargo features, Nix, release workflows,
containers, desktop Linux builds, and tests.

One sampler owns:

- per-device memory;
- utilization;
- temperature;
- power;
- health/error observations;
- host CPU/RAM;
- model/lease overlay from the registry.

Default cadence is 1 Hz, independently configurable only if existing telemetry
settings already provide a safe home. API reads snapshots and never shells out
per request.

Fallback:

- CUDA without NVML: UUID/device properties from CUDA; optional low-frequency
  `nvidia-smi` telemetry with explicitly missing fields;
- CUDA UUID success with no NVML/type join: schedulable as `UnknownCuda` from
  CUDA-advertised capabilities, with MIG-only and process telemetry fields
  explicitly missing and never ordinal-joined;
- CUDA UUID failure: visible but unavailable because Mold has no persistable
  stable identity;
- Metal: existing supported telemetry;
- CPU-only: host telemetry and empty devices.

Metrics:

- planner run count/duration/fallback/stale plans;
- ready-set size, eligibility-index update cost, and oversize matching duration;
- assignments by reason;
- eligible-idle milliseconds and exception reason;
- warm waits and cold loads avoided;
- model load/reload/eviction;
- prediction error by confidence;
- blocked time by reason;
- CPU offload and host-RAM admission;
- drain/start/stop duration and failures;
- per-device lease utilization;
- cancellation latency;
- batch staging/retry/cleanup;
- chain stage moves and affinity hits.

Logs use stable IDs and include ordinals only as secondary context.

## 16. Distribution

### 16.1 Targets

Add:

- `cuda-sm86` release tarball and container for RTX 3090-class Ampere;
- `cuda-sm100` release tarball and container for B200/GB200;
- matching checksums, release notes, installer detection, Nix packages/modules,
  applicable Linux desktop packaging, source-build documentation, and AUR
  guidance.

NVIDIA lists RTX 3090 as compute capability 8.6 and B200 as 10.0. Do not route
B200 to the consumer Blackwell sm120 artifact by name.

Do not remove sm80/sm89/sm90/sm120 paths in the same change. First reconcile
which hardware each path supports, add explicit compatibility/JIT notes, and
protect the current sm89-on-3090 path with a regression smoke until sm86
migration is complete.

Keep the existing `mold-ai-bin` package on sm89 so its artifact does not change
silently. Add `mold-ai-bin-sm86` after the real 3090 gate passes. Do not
publish `mold-ai-bin-sm100` until the deferred real B200 gate passes; before
then, document the sm100 release tarball and `mold-ai` source build with
`CUDA_COMPUTE_CAP=100`.

Nix adds `mold-sm86`, `mold-sm100`, and Linux `mold-desktop-sm86`. It does not
add a B200 desktop package: desktop clients manage a B200 server remotely.

### 16.2 Qualification language

Three distinct claims:

- **Designed:** data model/planner has no smaller-device-count assumption.
- **Simulated:** deterministic tests passed for a synthetic capability/topology.
- **Hardware-qualified:** the exact artifact and workload matrix ran on real
  hardware.

Initial implementation may claim:

- hardware-qualified 1× and 2× RTX 3090;
- simulated 12 GiB, 8×B200, and MIG;
- designed/simulated up to 64 devices.

It must not claim hardware-qualified B200, real 12 GiB, or MIG until the
deferred campaigns run.

### 16.3 Deferred Lambda campaign

Do not provision or run Lambda resources as part of this implementation, CI,
or review without a new explicit user instruction.

Later 8×B200 qualification must cover:

- sm100 artifact boots and loads every required CUDA/attention kernel;
- eight stable UUIDs and correct telemetry;
- homogeneous same-model saturation;
- mixed image/video/chain workloads;
- queue replanning and locality;
- drain/re-enable during load;
- long soak and fatal/nonfatal fault behavior;
- no hard-coded 2-device limits;
- packaging/install documentation.

MIG qualification may use a later supported A100/H100/B200 environment and
must obey the driver enumeration constraints in §7.3.

## 17. Rollout and phases

Each phase is a normal, independently reviewable PR series from a fresh
worktree. Do not keep the whole feature alive as a long-running mega-branch.
Every phase begins with failing tests, preserves the rollback state, and
updates affected docs.

### Phase A — identity, telemetry, read-only API

Deliver:

- UUID-first discovery and registry;
- NVML default-on for CUDA;
- CUDA_VISIBLE_DEVICES UUID/MIG joins;
- read-only `GET /api/devices`;
- machine-wide preference table written but not yet mutated by API;
- stable-ID and explicit `all`/`none` startup selection with legacy empty/all
  compatibility;
- legacy status derived from sampler;
- fix all first-GPU-only displays/routing reads.

Gates:

- synthetic numeric/raw stable ID/NVIDIA UUID/reordered/MIG/unknown-kind
  selector tests;
- real 2×3090 stable IDs and telemetry;
- byte/shape-compatible legacy status contracts;
- no per-request `nvidia-smi`;
- all standard CI gates.

### Phase B — scheduler V2 and ordinary work

Deliver:

- `mold-scheduler` crate;
- coordinator, versioned plans, debounce/cap, full-ready-set
  priority-preserving aggregate-admissible immediate matching and incremental
  eligibility indexes;
- bounded warm wait and three-bypass;
- deterministic operation-budget optimization and deterministic fallback;
- rendezvous workers;
- generation-aware host-RAM reservations;
- effective placement normalization and concrete per-component execution plans;
- reset-free CUDA reclamation and preflight;
- generation, expansion, upscale, and admin loads under leases;
- zero/single-device paths through the same core;
- local CLI adapter;
- rollout mode.

Rollout environment:

```text
MOLD_DISPATCH_MODE=legacy|observe|v2
```

`observe` does not own workers or compare impossible end-to-end state. It
records what V2 would assign at legacy dispatch points and checks hard
constraints, eligible-idle reasons, and predicted setup differences. Avoid
`MOLD_SCHEDULER`, which already refers to sampler selection.

Gates:

- synthetic 0/1/2/8/16/64 tests;
- flexible-plus-pinned counterexample, compatible work beyond rank 200, and
  watchdog fallback all preserve strict priority and matching cardinality;
- 8+8 GiB admissions against 12 GiB headroom never overcommit; an older 8 GiB
  job is not replaced by two younger 4 GiB jobs; matching reservation is atomic
  across a concurrent sample/ledger change;
- persisted/request/environment placement precedence and Auto CPU only under
  pressure;
- zero normal-runtime primary-context-reset references;
- two-admission/sample-generation host-RAM race suite;
- single 3090 throughput within 1% of baseline;
- 2×3090 mixed-model throughput no worse than the current dispatcher;
- indexed matching equals full recomputation; immediate dispatch p95 ≤250 ms
  for 200 and 10,000 ready units on synthetic 8/64-device inventories;
- planner p95 <100 ms for 200 jobs/8 devices and <200 ms for 200/64 on a
  pinned local benchmark, not a shared-runner CI timing assertion;
- larger explicitly configured queues stay correct, emit oversize telemetry,
  and have best-effort rather than claimed latency;
- no independent GPU acquisition remains for migrated work kinds;
- legacy rollback verified.

### Phase C — lifecycle and settings

Deliver:

- enable/drain/disable/start state machine;
- persisted desired enablement;
- all-disabled maintenance boot;
- authenticated PATCH API and events;
- configured-origin CORS support for PATCH and `X-Api-Key`;
- device PATCH classified under the existing Generation rate-limit tier;
- CLI controls;
- common settings/Machines toggles.

Gates:

- disable idle/busy, re-enable disabled/draining, and restart persistence;
- startup-excluded and absent device behavior;
- last-device maintenance mode;
- hard-pinned blocked jobs;
- auth matrix with key set/unset and loopback/remote;
- OPTIONS/PATCH CORS and queued/running golden legacy JSON;
- worker start/stop soak with no leaked threads or CUDA reuse.

### Phase D — chains

Deliver:

- chain stages under leases;
- multiple concurrent chains;
- boundary release and sticky affinity;
- legacy queue projection;
- adapt the existing chain-safe cancellation points to leased stages.

Gates:

- two concurrent chains on 2×3090;
- long-video chain interleaved with ordinary image/video work;
- durable restart with at least two interrupted chains;
- retake/resume/cancel;
- external VAE/text-projection/audio paths survive every stage;
- old chain endpoints and legacy queue position PATCH remain compatible.

### Phase E — learned estimates and complete clients

Deliver:

- persistent observations/ETAs/confidence;
- plan lanes and blocked reasons;
- Studio type migration;
- complete web/desktop/iPhone/TUI/CLI/Discord views;
- multi-host routing based on feasible devices and predicted completion.

Gates:

- persistence/pruning/DB-disabled/static-floor tests;
- every client renders 1, 2, 8, disabled, draining, unavailable, and MIG-shaped
  fixtures;
- remote explicit-target/auth tests;
- reactive SSE update tests;
- single-device compact layouts;
- UI/UX review and implementation review.

### Phase F — server-owned adaptive batch

F0 lands first as its own PR series:

- cooperative cancellation across every engine family;
- parent state machine, serialized reducer, attempt generations, and
  late-completion fencing;
- attempt-scoped staging, gallery publication barrier, commit journal, disk
  preflight, DB-enabled/disabled recovery, and cleanup;
- family batch capability declarations.

F1 then delivers:

- lazy parent/child model;
- once-per-parent concrete execution fingerprint;
- adaptive partition planning;
- logical atomic commit/recovery;
- fenced retry and cancellation.

Gates:

- ordered seed/output parity with sequential singleton baseline;
- same-fingerprint cross-device equality where exact determinism is claimed;
- enforced single-device fallback when determinism classes differ;
- disk-full/crash/restart commit recovery;
- no partial gallery/API result;
- crash injection after every state transition, file move, fsync, and DB step;
- concurrent gallery list/media GET/delete/reconcile cannot observe a partial
  commit;
- empty/disabled DB, final-name collision, and stale-attempt cleanup preserve
  atomic visibility;
- no completion event precedes durable `committed`;
- active/queued cancellation propagation;
- failure/cancel racing last completion, retry racing an old completion, and
  late completion after cleanup;
- all engine families explicitly declare batch capability.

### Phase G — sm86/sm100 distribution

May proceed in parallel after identity types stabilize.

Gates:

- sm86 build and real 3090 smoke, including attention backend;
- current sm89/JIT regression smoke on 3090;
- sm100 build, artifact publication, and loader/static checks;
- release assets/checksums/container/Nix/AUR/docs agree;
- no real Lambda launch;
- sm100 remains “simulated, not hardware-qualified” until the deferred
  campaign.

### Cutover

1. ship `observe` for decision telemetry;
2. enable V2 explicitly on development and qualification hosts;
3. make V2 default after phases A–E gates;
4. keep `legacy` for one release as restart-time rollback;
5. remove legacy dispatcher and flag after the rollback window.

Dynamic lifecycle is a V2 feature. A restart into legacy honors the startup
allowlist and enabled preferences at boot but does not promise runtime worker
mutation.

## 18. Test and acceptance strategy

### 18.1 TDD

Every behavior change starts with a failing exported-contract test:

- planner inputs/outputs and properties;
- worker state transitions;
- API serialization and compatibility;
- SSE reducers;
- UI state/rendering;
- packaging/document claims.

Use fake clocks for debounce/warm waits. Do not use sleeps for deterministic
scheduler tests.

### 18.2 Pure planner matrix

Inventories:

- 0, 1, 2, 8, 16, 64 devices;
- homogeneous 2×3090 and 8×B200 capability fixtures;
- one synthetic 12 GiB device;
- heterogeneous speed/VRAM/capability;
- reordered visible ordinals;
- UUID allowlists;
- MIG child/parent topology;
- disabled/draining/degraded/unavailable devices.

Properties:

- no duplicate lease or over-capacity device;
- no hard-pin violation;
- no assignment to disabled/unhealthy/infeasible devices;
- priority-preserving cardinality for flexible-plus-pinned devices and
  compatible work beyond optimization rank 200;
- exact admission matches a small exhaustive oracle;
- aggregate 8+8 GiB work cannot enter 12 GiB headroom, and an older 8 GiB unit
  is not bypassed by two younger 4 GiB units;
- all compatible idle devices used unless a typed exception exists;
- warm holds expire;
- three-bypass bound;
- deterministic result independent of map/hash iteration order, CPU speed,
  preemption, and injected wall clock;
- user order preserved under the defined exceptions;
- stale plans cannot grant;
- planner terminates within the deterministic operation budget;
- device sets use vectors/maps rather than fixed-size pairs or
  capacity-limited bitmasks.

### 18.3 Runtime/concurrency

Test:

- ready/lease/ack/completion generations;
- cancellation races;
- queue mutation during optimization;
- drain during load/generation/CPU phase;
- enable during drain;
- worker start failure/panic/disappearance;
- recoverable OOM cooldown;
- fatal CUDA rejects buffered/unstarted work and stops the process;
- model cache residency per worker;
- host-RAM concurrent reserve/commit/sample/release generations, including
  parked residency and sampler failure;
- whole-matching reservation is all-or-none across concurrent sample/ledger
  changes and failed worker grants;
- persisted-only/request/environment component placement, explicit CPU/GPU
  pins, disabled/conflicting pins, and Auto CPU pressure rules;
- model swaps, ghost VRAM, OOM, unload, disable, upscale, and LTX-2 without
  primary-context reset;
- pause/resume;
- DB disabled/corrupt;
- telemetry unavailable;
- disk-full staging/chain paths.

No test may reset or reuse a fatal CUDA primary context.

### 18.4 Model regression coverage

Avoid an unbounded Cartesian product while covering every engine family:

- **Tier 0, weight-free:** every family declares placement, batch,
  cancellation, determinism, and component capabilities; request
  normalization and concrete per-component execution-plan serialization
  round-trip; any artifact, variant, placement, attention, or load-strategy
  change changes the fingerprint; engines have no runtime variant fallback.
- **Tier 1, family smoke:** at least one runnable checkpoint per existing image
  and video family on applicable hardware, covering BF16/FP16 and one
  quantized path where the family supports both.
- **Tier 2, deep paths:** FLUX covers LoRA, block offload, encoder fallback,
  VAE tiling, source/variation; LTX-2 covers source video/image, audio
  supported/unsupported checkpoints, external assets, ConvRot where
  available, durable chains, chained videos, retake, and prepared batches.
- **Tier 3, deferred hardware:** real 8×B200, real 12 GiB, and real MIG.

Every supported family appears in the matrix with an owner and a concrete
test. “All models” means every engine family and execution contract, not every
third-party checkpoint multiplied by every option.

### 18.5 Real local 2×3090 acceptance

Build the sm86 CUDA artifact and prove:

- `/api/devices`, legacy status, queue plan, and telemetry show both cards;
- two compatible jobs use two cards;
- mixed models balance locality and cold-load cost;
- no sustained eligible-idle time without a typed exception;
- disable busy GPU drains, queued work replans, and re-enable restores it;
- missing selector, legacy empty selector, `all`, `none`, ordinal, stable UUID,
  NVIDIA UUID, ambiguous prefix, and reordered-visible-device cases behave as
  specified;
- all-disabled maintenance mode;
- image, source/edit, LoRA, upscale, video, generated audio where supported,
  durable chain, chained video, and prepared sibling flows;
- every client surface renders both devices;
- restart and legacy rollback.

The synthetic 12 GiB fixture may constrain the planner/device snapshot on a
3090 and prove the selected CPU/offload plan. It is not equivalent to real
12 GiB hardware qualification because physical allocations can still use the
3090’s full memory.

### 18.6 Determinism

Assert exact output only inside the existing guarantee boundary:

- same seed and normalized request;
- same model/component files;
- same execution fingerprint/determinism class;
- same code/backend guarantees.

Video codec outputs use decoded-frame/audio tolerances where byte identity is
not currently guaranteed. Batch parent order and metadata are always exact.

### 18.7 Performance acceptance

- single-device generation throughput regression ≤1%;
- current 2×3090 mixed-model dispatcher throughput is the minimum V2 target;
- idle assignment ≤250 ms p95 outside named exceptions for the default,
  200-ready, and 10,000-ready acceptance envelopes on 8/64 devices;
- planner output obeys the deterministic operation budget; the coordinator
  watchdog may return only the priority/cardinality-preserving seed;
- larger explicitly configured queues preserve correctness, emit oversize
  telemetry, and make no fixed latency claim;
- no queue-growth-proportional per-event client rendering;
- telemetry/status reads do not block model cache or spawn per-request
  processes.

### 18.8 Documentation and build gates

Run all affected:

- Rust fmt/check/clippy/tests/feature-combo;
- root frontend format/type/build/tests;
- desktop and iOS checks;
- website verification/build;
- release asset/package tests;
- `git diff --check`;
- `git ls-files -ci --exclude-standard`.

Update with each user-facing phase:

- `CHANGELOG.md`;
- `README.md`;
- `CLAUDE.md`/`AGENTS.md`;
- `.claude/skills/mold/SKILL.md`;
- `apps/mobile/README.md`;
- desktop docs;
- website installation/configuration/API/deployment pages;
- OpenAPI and CLI help;
- packaging/AUR documentation.

## 19. File impact map

Expected ownership:

| Area                | Primary paths                                                           |
| ------------------- | ----------------------------------------------------------------------- |
| Pure planner        | `crates/mold-scheduler/`                                                |
| Wire/domain types   | `crates/mold-core/src/types.rs`, client/OpenAPI types                   |
| Discovery/placement | `crates/mold-inference/src/device.rs`, engine capability descriptors    |
| Runtime coordinator | `crates/mold-server/src/scheduler/`                                     |
| Workers/cache       | `crates/mold-server/src/gpu_pool.rs`, `gpu_worker.rs`, `model_cache.rs` |
| Queue compatibility | `queue.rs`, `job_registry.rs`, routes/tests                             |
| Chains              | `chain_job_runner.rs`, chain routes/store                               |
| Persistence         | `crates/mold-db/src/migrations.rs`, server adapters                     |
| Telemetry           | `crates/mold-server/src/resources.rs`                                   |
| Shared browser API  | `studio/api/`, `studio/lib/`                                            |
| Web                 | `web/src/components/`, Machines/Settings/queue/routing                  |
| Desktop             | `desktop/src/components/`, stores, `HostDetailView`                     |
| iPhone              | `desktop/src/mobile/`, native API-key boundary unchanged                |
| TUI                 | `crates/mold-tui/src/ui/machines.rs`, chrome/app/backend                |
| CLI/MCP             | `crates/mold-cli/src/commands/`                                         |
| Discord             | `crates/mold-discord/src/format.rs`                                     |
| Distribution        | release/desktop workflows, Dockerfile, `flake.nix`, installer, AUR      |

Before implementation, each phase must replace this broad map with an exact
file/test checklist based on current main. Do not edit all surfaces in one PR
when additive backend contracts can land first.

## 20. Failure policy

Typed blocked/error reasons include:

- `device_disabled`;
- `device_draining`;
- `device_startup_excluded`;
- `device_unavailable`;
- `device_degraded`;
- `hard_pin_unavailable`;
- `backend_unsupported`;
- `model_not_installed`;
- `insufficient_vram`;
- `insufficient_host_ram`;
- `aggregate_host_ram_reserved`;
- `execution_plan_incompatible`;
- `dependency_wait`;
- `warm_wait`;
- `queue_paused`;
- `maintenance_mode`;
- `cancelling`.

Recoverable failure:

- update health/estimate;
- release lease/reservation exactly once;
- retain/replan unstarted work;
- do not silently change a hard pin or execution fingerprint.

Fatal CUDA:

- mark transient poison;
- reject all unstarted/buffered GPU work without touching CUDA;
- stop the server or embedded app;
- rely on service/app restart;
- never attempt in-process primary-context reset.

Device disappearance or worker panic is unavailable unless the error is
classified fatal CUDA, in which case process-fatal wins.

## 21. Definition of done

The feature is complete only when:

1. one registry is the source for workers, scheduling, telemetry, API, and UI;
2. no GPU consumer bypasses leases;
3. no unstarted work is stranded in a device-local buffer;
4. every ready compatible device is used or exposes a typed, bounded reason;
5. dynamic drain/disable/re-enable works and persists;
6. single-device and all-disabled behavior are correct;
7. queues/plans/telemetry render correctly on every client;
8. all existing engine-family and chain contracts pass the layered matrix;
9. real 2×3090 qualification passes;
10. 8×B200, 12 GiB, and MIG simulations pass without 1/2-device assumptions;
11. distribution produces documented sm86/sm100 artifacts;
12. docs distinguish designed, simulated, and hardware-qualified support;
13. legacy rollback is proven for its one-release window;
14. no fatal CUDA invariant, mobile host/auth invariant, or prepared-expansion
    invariant regresses.

Deferred real Lambda B200, real 12 GiB, and real MIG runs are separate
qualification milestones, not hidden completion criteria for the initial
implementation. When they run, the resulting evidence updates the support
claim and this document’s status.

## 22. Research and review provenance

Primary external constraints:

- NVIDIA CUDA Programming Guide,
  [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html):
  indices/UUIDs/MIG identifiers and ordinal remapping.
- NVIDIA CUDA Driver API,
  [`cuDeviceGetUuid_v2`](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-driver-api/group__CUDA__DEVICE.html):
  unique full-GPU/MIG device identity.
- NVIDIA MIG User Guide,
  [device enumeration](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/610/mig-device-names.html):
  current one-CI-per-GI and 64-instance constraints.
- NVIDIA MIG User Guide,
  [supported GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html):
  B200 MIG support and compute capability.
- NVIDIA,
  [CUDA GPU compute capability](https://developer.nvidia.com/cuda/gpus):
  RTX 3090 = 8.6 and B200 = 10.0.

Advisor review:

- Claude Code `2.1.220`;
- primary reviewer `claude-fable-5`, high effort;
- session `0dd16824-c7bf-464d-bd6f-e659ff7bfae4`;
- three read-only exploration sweeps resolved to `claude-opus-5[1m]`;
- the CLI reported a small internal `claude-haiku-4-5` support invocation;
  it did not author the substantive review, but future reviews must avoid that
  fan-out mode to comply with repository model policy;
- review baseline `c33f68de`; load-bearing scheduler files were unchanged and
  revalidated after fast-forward to `7cdb6aee`;
- no builds or tests were run by the advisor.

Independent architecture review:

- reviewer `gpt-5.6-sol`, requested high reasoning effort;
- collaboration task `/root/sol_spec_review`; the runtime exposed no opaque
  session UUID, so none is invented;
- fresh detached review worktree at `7cdb6aeefc5b`, removed after the read-only
  review;
- first-pass artifact SHA-256
  `87d0b93cd479508d67b05380d652b0041feea75e43aefb9e580f70960e19b5bc`;
- verdict `REVISE`: findings covered immediate maximum-cardinality matching,
  hard effective placement, concrete component plans, deterministic planning,
  host-RAM accounting, reset-free CUDA reclamation, F0 completion fencing,
  CORS/rate-limit/wire compatibility, all-disabled selection, and MIG
  identity classification;
- second-pass artifact SHA-256
  `292ba6de372273a32568173e154eb313ab73eb47117d5476f7dfc157a8357dd0`;
- second verdict `REVISE`: the first ten findings were closed; new findings
  required priority-preserving aggregate host-RAM reservation, attempt-isolated
  gallery publication, an explicit large-queue latency envelope, and
  untracked-file whitespace validation;
- final architecture-content SHA-256
  `20201a7829237cc5cca429745c43f78efc3718130f51b3ed4eb0993241d21908`;
- final verdict `APPROVE`: the reviewer confirmed the aggregate reservation,
  attempt-isolated same-filesystem publication/recovery, 10,000-item latency
  envelope, and whitespace gate, with no new contradiction;
- no build, test, service mutation, or hardware qualification was performed by
  this reviewer.

Acceptance review policy:

- every high-risk scheduler, CUDA lifecycle, concurrency, persistence, or
  authority phase requires independent Sol high-effort and Opus 5 high-effort
  reviews;
- UI/API/product boundaries require Opus 5 high-effort review plus
  implementation review;
- major phase and feature completion requires both;
- record exact model, effort, session, baseline SHA, and deterministic test
  evidence in the phase PR;
- Fable is not required again unless a later architectural decision is
  comparably consequential.
