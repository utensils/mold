import { modelKindLabel } from "./modelMetadata";

export const MINIMAX_H3_FAMILY = "minimax-h3";

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
    partitionsByTask.set(candidate.task, {
      task: candidate.task,
      model: candidate.model,
      display_name: candidate.display_name.trim(),
      runtime_available: true,
      tier: candidate.tier.trim(),
      component_ids: [...candidate.component_ids],
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
