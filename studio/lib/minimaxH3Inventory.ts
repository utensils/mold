import { modelKindLabel } from "./modelMetadata";

export type MiniMaxH3Task = "fl2va" | "ref2va";
export type MiniMaxH3ComponentState =
  "installed" | "missing" | "downloading" | "failed";
export type MiniMaxH3ComponentRole =
  "transformer" | "qwen" | "processor" | "video-vae" | "audio-vae";

export interface MiniMaxH3DownloadCapability {
  job_id: string;
  bytes_done: number;
  bytes_total: number;
}

export interface MiniMaxH3ComponentCapability {
  id: string;
  display_name: string;
  kind: string;
  role: MiniMaxH3ComponentRole;
  scope: "shared" | MiniMaxH3Task;
  size_bytes: number;
  state: MiniMaxH3ComponentState;
  download?: MiniMaxH3DownloadCapability | null;
  error?: string | null;
}

export interface MiniMaxH3PartitionCapability {
  task: MiniMaxH3Task;
  model: string;
  display_name: string;
  runtime_available: boolean;
  tier: string;
  component_ids: string[];
  request?: MiniMaxH3RequestCapability | null;
  /** Reviewed Turbo LoRA variants (additive; absent on older servers). */
  turbo?: MiniMaxH3TurboVariantCapability[] | null;
}

/** One reviewed Turbo LoRA variant of a compact H3 partition: the same base
 * component stack plus one pinned adapter. `request` is present only when the
 * variant is executable (base stack and adapter both landed). */
export interface MiniMaxH3TurboVariantCapability {
  model: string;
  display_name: string;
  tier: string;
  adapter_size_bytes: number;
  installed: boolean;
  request?: MiniMaxH3RequestCapability | null;
}

export interface MiniMaxH3RequestCapability {
  width: number;
  height: number;
  frames: number;
  fps: number;
  steps: number;
  batch_size: number;
  output_format: "mp4";
  required_endpoint: "first";
  generation_profile_sha256: string;
}

export interface MiniMaxH3QualificationCapability {
  backend: "cuda";
  metal_supported: false;
  minimum_host_ram_bytes: number;
  minimum_vram_bytes: number;
  attention_profile: string;
  quantization_profile: string;
}

/**
 * Additive, host-authored presentation facts for a future authorized H3
 * runtime. Current servers omit this field. It is deliberately not a model
 * manifest or catalog recipe and therefore cannot authorize a download.
 */
export interface MiniMaxH3Capability {
  runtime_available: boolean;
  qualification: MiniMaxH3QualificationCapability;
  partitions: MiniMaxH3PartitionCapability[];
  components: MiniMaxH3ComponentCapability[];
}

export interface MiniMaxH3CapabilityRecord {
  model_access?: {
    restrictions?:
      | readonly {
          code: string;
          family: string;
          message: string;
          license_url: string;
          authorization_url: string;
        }[]
      | null;
  } | null;
  minimax_h3?: MiniMaxH3Capability | null;
}

export interface MiniMaxH3HostInput {
  id: string;
  label: string;
  capabilities: MiniMaxH3CapabilityRecord | null | undefined;
}

export interface MiniMaxH3DownloadPresentation {
  hostId: string;
  hostLabel: string;
  jobId: string;
  bytesDone: number;
  bytesTotal: number;
}

export interface MiniMaxH3ComponentPresentation {
  id: string;
  displayName: string;
  kind: string;
  kindLabel: string;
  role: MiniMaxH3ComponentRole;
  scope: "shared" | MiniMaxH3Task;
  sizeBytes: number;
  state: MiniMaxH3ComponentState;
  download: MiniMaxH3DownloadPresentation | null;
  error: string | null;
}

export interface MiniMaxH3TaskPresentation {
  task: MiniMaxH3Task;
  taskLabel: "FL2VA" | "Ref2VA";
  model: string;
  displayName: string;
  tier: string;
  diskBytes: number;
  remainingBytes: number;
  readiness: MiniMaxH3ComponentState;
  components: MiniMaxH3ComponentPresentation[];
  request: MiniMaxH3RequestCapability | null;
  /** Reviewed Turbo LoRA variants advertised additively on this partition. */
  turbo: MiniMaxH3TurboVariantCapability[];
}

export interface MiniMaxH3HostPresentation {
  hostId: string;
  hostLabel: string;
  qualification: MiniMaxH3QualificationCapability;
  tasks: MiniMaxH3TaskPresentation[];
  components: MiniMaxH3ComponentPresentation[];
  sharedComponents: MiniMaxH3ComponentPresentation[];
  advertisedTasksDiskBytes: number;
  remainingBytes: number;
}

export type MiniMaxH3PlanAction =
  "installed" | "install" | "repair" | "waiting";

export interface MiniMaxH3InstallPlan {
  hostId: string;
  hostLabel: string;
  tasks: MiniMaxH3Task[];
  action: MiniMaxH3PlanAction;
  /** Missing or failed components to fetch/repair, unique by component id. */
  components: MiniMaxH3ComponentPresentation[];
  /** Existing work on this exact host; never transferable to another host. */
  downloads: MiniMaxH3DownloadPresentation[];
  bytesToDownload: number;
}

const TASKS: readonly MiniMaxH3Task[] = ["fl2va", "ref2va"];
const SHARED_ROLES: readonly MiniMaxH3ComponentRole[] = [
  "qwen",
  "processor",
  "video-vae",
  "audio-vae",
];
const COMPONENT_STATES: readonly MiniMaxH3ComponentState[] = [
  "installed",
  "missing",
  "downloading",
  "failed",
];
const COMPONENT_ROLES: readonly MiniMaxH3ComponentRole[] = [
  "transformer",
  ...SHARED_ROLES,
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function byteCount(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function taskValue(value: unknown): value is MiniMaxH3Task {
  return TASKS.includes(value as MiniMaxH3Task);
}

function stateValue(value: unknown): value is MiniMaxH3ComponentState {
  return COMPONENT_STATES.includes(value as MiniMaxH3ComponentState);
}

function roleValue(value: unknown): value is MiniMaxH3ComponentRole {
  return COMPONENT_ROLES.includes(value as MiniMaxH3ComponentRole);
}

/** Client-side mirror of the compact FL2VA step DEFAULTS
 * (`mold_core::minimax_h3`): 21 for the base identity, 9/5 for the reviewed
 * Turbo tiers.
 *
 * For a Turbo tier this is still exact — the count is the distilled adapter's
 * own schedule length, so the advertised envelope must agree. For the base
 * identity it is the default within
 * `MINIMAX_H3_COMPACT_MIN_STEPS..=MINIMAX_H3_COMPACT_MAX_STEPS`, not a pin. */
export const MINIMAX_H3_REVIEWED_FL2VA_STEPS: Readonly<Record<string, number>> =
  {
    "minimax-h3-fl2va:comfy-pruned-int8": 21,
    "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step": 9,
    "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p": 5,
  };

/** Canvases the ADVERTISED reference request may name.
 *
 * The server advertises exactly one reference request per model row, and it is
 * always the default canvas; the compact runtime itself admits any 32-aligned
 * canvas inside its area ceiling, which reaches clients through the row's
 * generation profile. So this is a pin on the advertised request, not a
 * statement about what the runtime renders. */
const MINIMAX_H3_REVIEWED_CANVASES: readonly (readonly [number, number])[] = [
  [1344, 768],
  [768, 768],
];

function parseTurboVariant(
  value: unknown,
): MiniMaxH3TurboVariantCapability | null {
  if (
    !isRecord(value) ||
    !nonEmptyString(value.model) ||
    !nonEmptyString(value.display_name) ||
    !nonEmptyString(value.tier) ||
    !byteCount(value.adapter_size_bytes) ||
    typeof value.installed !== "boolean"
  ) {
    return null;
  }
  const request = value.request == null ? null : parseRequest(value.request);
  if (value.request != null && request === null) return null;
  return {
    model: value.model,
    display_name: value.display_name.trim(),
    tier: value.tier.trim(),
    adapter_size_bytes: value.adapter_size_bytes,
    installed: value.installed,
    request,
  };
}

function parseRequest(value: unknown): MiniMaxH3RequestCapability | null {
  if (
    !isRecord(value) ||
    !byteCount(value.width) ||
    value.width === 0 ||
    !byteCount(value.height) ||
    value.height === 0 ||
    !byteCount(value.frames) ||
    value.frames === 0 ||
    !byteCount(value.fps) ||
    value.fps === 0 ||
    !byteCount(value.steps) ||
    value.steps === 0 ||
    value.batch_size !== 1 ||
    value.output_format !== "mp4" ||
    value.required_endpoint !== "first" ||
    typeof value.generation_profile_sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(value.generation_profile_sha256)
  ) {
    return null;
  }
  return {
    width: value.width,
    height: value.height,
    frames: value.frames,
    fps: value.fps,
    steps: value.steps,
    batch_size: 1,
    output_format: "mp4",
    required_endpoint: "first",
    generation_profile_sha256: value.generation_profile_sha256,
  };
}

function parseDownload(
  value: unknown,
  host: MiniMaxH3HostInput,
): MiniMaxH3DownloadPresentation | null {
  if (!isRecord(value)) return null;
  if (
    !nonEmptyString(value.job_id) ||
    !byteCount(value.bytes_done) ||
    !byteCount(value.bytes_total) ||
    value.bytes_done > value.bytes_total
  ) {
    return null;
  }
  return {
    hostId: host.id,
    hostLabel: host.label,
    jobId: value.job_id,
    bytesDone: value.bytes_done,
    bytesTotal: value.bytes_total,
  };
}

function parseComponent(
  value: unknown,
  host: MiniMaxH3HostInput,
): MiniMaxH3ComponentPresentation | null {
  if (
    !isRecord(value) ||
    !nonEmptyString(value.id) ||
    !nonEmptyString(value.display_name) ||
    !nonEmptyString(value.kind) ||
    !roleValue(value.role) ||
    !(value.scope === "shared" || taskValue(value.scope)) ||
    !byteCount(value.size_bytes) ||
    !stateValue(value.state)
  ) {
    return null;
  }
  const download = parseDownload(value.download, host);
  if (value.state === "downloading" && download === null) return null;
  const error = nonEmptyString(value.error) ? value.error.trim() : null;
  if (value.state === "failed" && error === null) return null;
  return {
    id: value.id,
    displayName: value.display_name.trim(),
    kind: value.kind,
    kindLabel: modelKindLabel(value.kind),
    role: value.role,
    scope: value.scope,
    sizeBytes: value.size_bytes,
    state: value.state,
    download: value.state === "downloading" ? download : null,
    error: value.state === "failed" ? error : null,
  };
}

function parseQualification(
  value: unknown,
): MiniMaxH3QualificationCapability | null {
  if (
    !isRecord(value) ||
    value.backend !== "cuda" ||
    value.metal_supported !== false ||
    !byteCount(value.minimum_host_ram_bytes) ||
    value.minimum_host_ram_bytes === 0 ||
    !byteCount(value.minimum_vram_bytes) ||
    value.minimum_vram_bytes === 0 ||
    !nonEmptyString(value.attention_profile) ||
    !nonEmptyString(value.quantization_profile)
  ) {
    return null;
  }
  return {
    backend: "cuda",
    metal_supported: false,
    minimum_host_ram_bytes: value.minimum_host_ram_bytes,
    minimum_vram_bytes: value.minimum_vram_bytes,
    attention_profile: value.attention_profile.trim(),
    quantization_profile: value.quantization_profile.trim(),
  };
}

function readiness(
  components: readonly MiniMaxH3ComponentPresentation[],
): MiniMaxH3ComponentState {
  if (components.some((component) => component.state === "failed"))
    return "failed";
  if (components.some((component) => component.state === "downloading"))
    return "downloading";
  if (components.some((component) => component.state === "missing"))
    return "missing";
  return "installed";
}

function sumUnique(
  components: readonly MiniMaxH3ComponentPresentation[],
  include: (component: MiniMaxH3ComponentPresentation) => boolean = () => true,
): number {
  const seen = new Set<string>();
  return components.reduce((total, component) => {
    if (seen.has(component.id) || !include(component)) return total;
    seen.add(component.id);
    return total + component.sizeBytes;
  }, 0);
}

/**
 * Validate and present one exact host's H3 facts. This is the #831/#841
 * authority boundary: absence, disabled runtime, Metal support, or a malformed
 * component graph all return null, so no surface can tease support. The
 * additive partition is intentionally narrower than the legacy family-wide
 * model-access denial, which remains present for clients that do not understand
 * this exact private capability.
 */
export function presentMiniMaxH3Host(
  host: MiniMaxH3HostInput,
): MiniMaxH3HostPresentation | null {
  const capabilities = host.capabilities;
  if (!capabilities || capabilities.minimax_h3?.runtime_available !== true) {
    return null;
  }
  const raw = capabilities.minimax_h3;
  if (!Array.isArray(raw.components) || !Array.isArray(raw.partitions))
    return null;
  const qualification = parseQualification(raw.qualification);
  if (!qualification) return null;

  const components: MiniMaxH3ComponentPresentation[] = [];
  const componentsById = new Map<string, MiniMaxH3ComponentPresentation>();
  for (const candidate of raw.components) {
    const component = parseComponent(candidate, host);
    if (!component || componentsById.has(component.id)) return null;
    components.push(component);
    componentsById.set(component.id, component);
  }

  const partitionsByTask = new Map<
    MiniMaxH3Task,
    MiniMaxH3PartitionCapability
  >();
  for (const candidate of raw.partitions as unknown[]) {
    if (
      !isRecord(candidate) ||
      !taskValue(candidate.task) ||
      !nonEmptyString(candidate.model) ||
      !nonEmptyString(candidate.display_name) ||
      candidate.runtime_available !== true ||
      !nonEmptyString(candidate.tier) ||
      !Array.isArray(candidate.component_ids) ||
      !candidate.component_ids.every(nonEmptyString) ||
      new Set(candidate.component_ids).size !==
        candidate.component_ids.length ||
      partitionsByTask.has(candidate.task)
    ) {
      return null;
    }
    const turbo: MiniMaxH3TurboVariantCapability[] = [];
    if (candidate.turbo != null) {
      if (!Array.isArray(candidate.turbo)) return null;
      const seen = new Set<string>();
      for (const raw of candidate.turbo as unknown[]) {
        const variant = parseTurboVariant(raw);
        if (!variant || seen.has(variant.model)) return null;
        seen.add(variant.model);
        turbo.push(variant);
      }
    }
    partitionsByTask.set(candidate.task, {
      task: candidate.task,
      model: candidate.model,
      display_name: candidate.display_name.trim(),
      runtime_available: true,
      tier: candidate.tier.trim(),
      component_ids: [...candidate.component_ids],
      request: parseRequest(candidate.request),
      turbo,
    });
  }
  if (partitionsByTask.size === 0) return null;

  const sharedByRole = new Map<
    MiniMaxH3ComponentRole,
    MiniMaxH3ComponentPresentation
  >();
  for (const component of components.filter(
    (candidate) => candidate.scope === "shared",
  )) {
    if (sharedByRole.has(component.role)) return null;
    sharedByRole.set(component.role, component);
  }
  if (SHARED_ROLES.some((role) => !sharedByRole.has(role))) return null;

  const tasks: MiniMaxH3TaskPresentation[] = [];
  for (const task of TASKS) {
    const partition = partitionsByTask.get(task);
    if (!partition) continue;
    const taskComponents = partition.component_ids.map((id) =>
      componentsById.get(id),
    );
    if (taskComponents.some((component) => component === undefined))
      return null;
    const resolved = taskComponents as MiniMaxH3ComponentPresentation[];
    if (
      resolved.some(
        (component) => component.scope !== "shared" && component.scope !== task,
      ) ||
      !resolved.some(
        (component) =>
          component.scope === task && component.role === "transformer",
      ) ||
      SHARED_ROLES.some(
        (role) =>
          !resolved.some(
            (component) =>
              component.scope === "shared" && component.role === role,
          ),
      )
    ) {
      return null;
    }
    tasks.push({
      task,
      taskLabel: task === "fl2va" ? "FL2VA" : "Ref2VA",
      model: partition.model,
      displayName: partition.display_name,
      tier: partition.tier,
      diskBytes: sumUnique(resolved),
      remainingBytes: sumUnique(
        resolved,
        (component) => component.state !== "installed",
      ),
      readiness: readiness(resolved),
      components: resolved,
      request: parseRequest(partition.request),
      turbo: partition.turbo ?? [],
    });
  }

  const referencedIds = new Set(
    tasks.flatMap((task) => task.components.map((component) => component.id)),
  );
  if (referencedIds.size !== components.length) return null;
  return {
    hostId: host.id,
    hostLabel: host.label,
    qualification,
    tasks,
    components,
    sharedComponents: components.filter(
      (component) => component.scope === "shared",
    ),
    advertisedTasksDiskBytes: sumUnique(components),
    remainingBytes: sumUnique(
      components,
      (component) => component.state !== "installed",
    ),
  };
}

/**
 * Return the exact request envelope that may override the legacy family-wide
 * H3 restriction for one model row. The complete capability graph must parse,
 * every referenced component must already be installed, and only the reviewed
 * compact FL2VA identities — the base partition and its reviewed Turbo
 * variants — are eligible. Every axis but steps is pinned to the reviewed
 * canvas; the expected step count is the selected model's own reviewed
 * authority from `MINIMAX_H3_REVIEWED_FL2VA_STEPS`, so a Turbo model passes
 * its tier's steps while a widened envelope still fails closed. Malformed or
 * older capability payloads stay read-only inventory and cannot become
 * generation authority.
 */
export function reviewedMiniMaxH3ModelAccess(
  capabilities: MiniMaxH3CapabilityRecord | null | undefined,
  model: string | null | undefined,
  generationProfileSha256: string | null | undefined,
): MiniMaxH3RequestCapability | null {
  if (typeof model !== "string") return null;
  const expectedSteps = MINIMAX_H3_REVIEWED_FL2VA_STEPS[model];
  if (expectedSteps === undefined) return null;
  const presented = presentMiniMaxH3Host({
    id: "model-access",
    label: "model access",
    capabilities,
  });
  const task = presented?.tasks.find(
    (candidate) =>
      candidate.task === "fl2va" && candidate.readiness === "installed",
  );
  if (!task) return null;
  const request =
    task.model === model
      ? task.request
      : (task.turbo.find(
          (variant) => variant.model === model && variant.installed,
        )?.request ?? null);
  // The advertised reference request is always the DEFAULT canvas — the
  // reviewed set's first entry — never "whichever canvas this host felt
  // like". Every other qualified canvas reaches the form through the row's
  // generation-profile buckets, which the sha256 below already fences.
  const [[defaultWidth, defaultHeight]] = MINIMAX_H3_REVIEWED_CANVASES as [
    readonly [number, number],
    ...(readonly [number, number])[],
  ];
  if (
    !request ||
    request.width !== defaultWidth ||
    request.height !== defaultHeight ||
    request.frames !== 124 ||
    request.fps !== 24 ||
    request.steps !== expectedSteps ||
    request.batch_size !== 1 ||
    request.output_format !== "mp4" ||
    request.required_endpoint !== "first" ||
    generationProfileSha256 !== request.generation_profile_sha256
  ) {
    return null;
  }
  return request;
}

/** Keep mixed-fleet snapshots independent; one denied or stale host vanishes. */
export function presentMiniMaxH3Fleet(
  hosts: readonly MiniMaxH3HostInput[],
): MiniMaxH3HostPresentation[] {
  return hosts.flatMap((host) => {
    const presentation = presentMiniMaxH3Host(host);
    return presentation ? [presentation] : [];
  });
}

/**
 * Describe—not execute—the exact-host component work for one or both tasks.
 * Shared requirements are unioned by id, active downloads stay owned by their
 * source host, and no API path or catalog id is produced here.
 */
export function planMiniMaxH3Install(
  host: MiniMaxH3HostPresentation,
  tasks: readonly MiniMaxH3Task[],
): MiniMaxH3InstallPlan | null {
  const requested = [...new Set(tasks)];
  if (requested.length === 0 || requested.some((task) => !TASKS.includes(task)))
    return null;
  const wanted = new Map<string, MiniMaxH3ComponentPresentation>();
  for (const task of requested) {
    const partition = host.tasks.find((candidate) => candidate.task === task);
    if (!partition) return null;
    for (const component of partition.components)
      wanted.set(component.id, component);
  }
  const all = [...wanted.values()];
  const components = all.filter(
    (component) =>
      component.state === "missing" || component.state === "failed",
  );
  const downloadsByJob = new Map<string, MiniMaxH3DownloadPresentation>();
  for (const component of all) {
    if (!component.download) continue;
    downloadsByJob.set(
      `${component.download.hostId}\u0000${component.download.jobId}`,
      component.download,
    );
  }
  const downloads = [...downloadsByJob.values()];
  const installed = all.filter(
    (component) => component.state === "installed",
  ).length;
  const action: MiniMaxH3PlanAction =
    components.length === 0
      ? downloads.length > 0
        ? "waiting"
        : "installed"
      : installed > 0 ||
          downloads.length > 0 ||
          components.some((component) => component.state === "failed")
        ? "repair"
        : "install";
  return {
    hostId: host.hostId,
    hostLabel: host.hostLabel,
    tasks: requested,
    action,
    components,
    downloads,
    bytesToDownload: sumUnique(components),
  };
}

export function formatMiniMaxH3Bytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  return `${bytes} B`;
}
