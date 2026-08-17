/*
 * Generation host routing state for the Create surface — the browser's answer
 * to the desktop app's hosts store (web has no Pinia; module-singleton
 * composables are the state layer).
 *
 * It polls every host in the registry for `/api/status` (queue depth + GPU, the
 * routers' inputs) and `/api/models` (so the model picker reflects where the
 * job will actually run, and so Auto/Most-capable can prefer a host that
 * already holds the weights). All the decisions live in `lib/hostRouting.ts`;
 * this file is only the plumbing that keeps them fed.
 */
import {
  computed,
  onBeforeUnmount,
  ref,
  type ComputedRef,
  type Ref,
} from "vue";
import {
  GENERATE_TARGET_CHANGED_EVENT,
  HOSTS_CHANGED_EVENT,
  ORIGIN_HOST_ID,
  getGenerateTargetId,
  listHosts,
  setGenerateTargetId,
  type HostEntry,
} from "../lib/hostRegistry";
import {
  hostCapabilities,
  hostDevices,
  hostModels,
  hostQueue,
  hostStatus,
} from "../components/machines/hostClient";
import {
  filterRestrictedModels,
  modelAccessRestrictionFor,
} from "@studio/lib/modelAccess";
import type { DeviceInfo } from "@studio/api/devices";
import { ApiError, type ApiTarget } from "@studio/api/client";
import { profileHashConflict } from "@studio/lib/profileFleet";
import {
  predictedCompletionUnixMs,
  type QueueListing,
} from "@studio/api/queuePlan";
import { hostMemoryLevel, type HostMemoryLevel } from "@studio/lib/hostMemory";
import {
  buildQueueStatusIndex,
  type QueueStatusIndex,
} from "@studio/lib/queuePosition";
import {
  comparePlacementPreviews,
  classifyPlacementPreview,
  previewChainPlacement,
  previewGenerationPlacement,
  previewRequestForSiblingFanout,
  requiresAuthoritativePlacement,
  type GenerationPlacementPreview,
  type PlacementMissingComponent,
} from "@studio/api/generationPlacement";
import {
  AUTO_TARGET_ID,
  CAPABLE_TARGET_ID,
  hostIdsForModel,
  normalizeTargetId,
  resolveRoute,
  unionModels,
  type HostRoute,
  type HostRoutingStatus,
  type ModelsByHost,
  type RoutableGpu,
  type RoutableHost,
} from "../lib/hostRouting";
import type {
  ChainRequestWire,
  GenerateRequestWire,
  GpuInfo,
  GpuWorkerStatus,
  ModelInfoExtended,
  ServerCapabilities,
  ServerStatus,
} from "../types";

/** `gpu_info` plus the additive `backend` field newer servers report. */
type GpuInfoWithBackend = GpuInfo & { backend?: string | null };

interface HostTelemetry {
  status: HostRoutingStatus;
  instanceId: string | null;
  queueDepth: number | null;
  gpu: RoutableGpu | null;
  predictedCompletionMs: number | null;
  /** Last good `/api/queue` read, retained through a blip so a live queue
   * position does not flicker away for one poll. */
  queue: QueueListing | null;
}

export interface HostRouting {
  /** Every registry host with its live routing inputs, origin first. */
  hosts: Ref<RoutableHost[]>;
  /** The persisted pick, normalized (a forgotten host reads as Auto). */
  targetId: ComputedRef<string>;
  setTarget: (id: string) => void;
  /** True once the browser knows about a machine other than this server. */
  multiHost: ComputedRef<boolean>;
  /** Models available where a job would land: the target host's list, or the
   * union across ready hosts under Auto / Most capable. */
  targetModels: ComputedRef<ModelInfoExtended[]>;
  /** Every downloaded model across ready machines, regardless of the pinned
   * target — the ⌘K palette searches the whole fleet, not just where the next
   * job would land, because choosing a model may repin the target. */
  installedModels: ComputedRef<ModelInfoExtended[]>;
  /** True once every listed host's `/api/models` has settled — the empty-state
   * gate, so a slow remote can't make Create flash "nothing installed". */
  modelsSettled: Ref<boolean>;
  /** Ids of the machines known to hold `name` on disk. */
  modelOwnerIds: (name: string) => string[];
  /** True once this machine's `/api/models` has actually been read. Anything
   * else must never be described as missing a model — we haven't looked. */
  inventoryKnown: (hostId: string) => boolean;
  /** Last successful capability snapshot per exact host. Presentation-only
   * consumers must still fail closed when a host has no current entry. */
  capabilitiesByHost: Ref<Record<string, ServerCapabilities>>;
  /** Live dispatch order (and any blocked reason) for every listed host's
   * queued work, keyed by host + server job id. The one-shot SSE `Queued`
   * frame never updates; this does, on every poll. */
  queueStatus: ComputedRef<QueueStatusIndex>;
  /** Host-RAM pressure for one machine, or null when it does not report it. */
  hostMemoryPressure: (hostId: string) => HostMemoryLevel | null;
  /** Resolve the concrete dispatch route for a model, or null if unreachable. */
  resolve: (model: string | null) => HostRoute | null;
  /** Resolve through each host's read-only authoritative scheduler preview. */
  resolveFeasible: (
    request: GenerateRequestWire,
    copies?: number,
  ) => Promise<FeasibilityResult>;
  /** Revalidate only a previously selected host; never globally rerank. */
  revalidateFeasible: (
    route: HostRoute,
    request: GenerateRequestWire,
    copies?: number,
  ) => Promise<FeasibilityResult>;
  /** Resolve a complete durable sequence through the same scheduler preview. */
  resolveFeasibleChain: (
    request: ChainRequestWire,
    copies?: number,
  ) => Promise<FeasibilityResult>;
  revalidateFeasibleChain: (
    route: HostRoute,
    request: ChainRequestWire,
    copies?: number,
  ) => Promise<FeasibilityResult>;
  /** Re-read the registry and poll every host once. */
  refresh: () => Promise<void>;
  /** Start/stop the poll loop; ref-counted across consumers. */
  start: () => void;
  stop: () => void;
}

export interface InfeasibleHost {
  hostId: string;
  label: string;
  reason: string;
  missingComponents: PlacementMissingComponent[];
}

export interface UnreachableHost {
  hostId: string;
  label: string;
  error: string;
}

export interface TransientHost {
  hostId: string;
  label: string;
  reason: string;
}

export type FeasibilityResult =
  | { kind: "route"; route: HostRoute }
  | {
      kind: "profile_mismatch";
      perHost: Array<{
        hostId: string;
        label: string;
        profileHash: string | null;
      }>;
    }
  | {
      kind: "infeasible";
      perHost: InfeasibleHost[];
      unreachable?: UnreachableHost[];
    }
  | {
      kind: "unreachable";
      perHost: UnreachableHost[];
      infeasible?: InfeasibleHost[];
    }
  | {
      kind: "transient";
      perHost: TransientHost[];
      infeasible?: InfeasibleHost[];
      unreachable?: UnreachableHost[];
    };

const POLL_INTERVAL_MS = 8000;

const entries = ref<HostEntry[]>([]);
const telemetry = ref<Record<string, HostTelemetry>>({});
const modelsByHost = ref<ModelsByHost>({});
const capabilitiesByHost = ref<Record<string, ServerCapabilities>>({});
const settledHostIds = ref<string[]>([]);
/** Hosts whose `/api/models` actually came back. A blipped host keeps its last
 * good list, but one that has never answered is unknown, not empty. */
const inventoryHostIds = ref<string[]>([]);
const modelsSettled = ref(false);
const rawTargetId = ref<string>("");
const pollGenerations = new Map<string, number>();
let routingAuthorityGeneration = 0;
let targetPolicyGeneration = 0;
let registryAuthoritySignature = "";

function readRegistry(): void {
  const next = typeof localStorage === "undefined" ? [] : listHosts();
  const signature = JSON.stringify(
    next.map(({ id, url, apiKey, instanceId }) => [
      id,
      url,
      apiKey ?? null,
      instanceId ?? null,
    ]),
  );
  if (signature !== registryAuthoritySignature) {
    registryAuthoritySignature = signature;
    routingAuthorityGeneration += 1;
  }
  entries.value = next;
}

function readTarget(): void {
  const next =
    typeof localStorage === "undefined"
      ? ORIGIN_HOST_ID
      : getGenerateTargetId();
  if (next !== rawTargetId.value) {
    rawTargetId.value = next;
    routingAuthorityGeneration += 1;
    targetPolicyGeneration += 1;
  }
}

const hosts = computed<RoutableHost[]>(() =>
  entries.value.map((entry) => {
    const live = telemetry.value[entry.id];
    const host: RoutableHost = {
      id: entry.id,
      label: entry.name,
      url: entry.url,
      status: live?.status ?? "connecting",
      queueDepth: live?.queueDepth ?? null,
      gpu: live?.gpu ?? null,
      predictedCompletionMs: live?.predictedCompletionMs ?? null,
    };
    if (entry.apiKey) host.apiKey = entry.apiKey;
    const instanceId = live?.instanceId ?? entry.instanceId;
    if (instanceId) host.instanceId = instanceId;
    return host;
  }),
);

/**
 * Live queue positions across the whole registry, folded from the `/api/queue`
 * read this poll already performs. A host that has not answered contributes
 * nothing — absence of an entry is never "position 0".
 */
const queueStatus = computed<QueueStatusIndex>(() =>
  buildQueueStatusIndex(
    Object.entries(telemetry.value).flatMap(([hostId, live]) =>
      live.queue
        ? [{ hostId, entries: live.queue.entries, plan: live.queue.plan }]
        : [],
    ),
  ),
);

function hostMemoryPressure(hostId: string): HostMemoryLevel | null {
  return hostMemoryLevel(telemetry.value[hostId]?.queue?.plan?.host_memory);
}

const targetId = computed(() =>
  normalizeTargetId(rawTargetId.value, hosts.value),
);

const readyHostIds = computed(() =>
  hosts.value.filter((h) => h.status === "ready").map((h) => h.id),
);

function accessibleModelsOn(hostId: string): ModelInfoExtended[] {
  return filterRestrictedModels(
    modelsByHost.value[hostId] ?? [],
    capabilitiesByHost.value[hostId],
  );
}

function accessRestrictionForHost(hostId: string, model: string) {
  const entry = modelsByHost.value[hostId]?.find(
    (candidate) => candidate.name === model,
  );
  return modelAccessRestrictionFor(capabilitiesByHost.value[hostId], {
    model,
    family: entry?.family,
    generation_profile_sha256: entry?.generation_profile?.profile_hash ?? null,
  });
}

const accessibleModelsByHost = computed<ModelsByHost>(() =>
  Object.fromEntries(
    Object.keys(modelsByHost.value).map((hostId) => [
      hostId,
      accessibleModelsOn(hostId),
    ]),
  ),
);

const targetModels = computed<ModelInfoExtended[]>(() => {
  const sel = targetId.value;
  if (sel !== AUTO_TARGET_ID && sel !== CAPABLE_TARGET_ID) {
    return accessibleModelsOn(sel);
  }
  // Auto / Most capable can land anywhere, so the picker offers the union.
  // Before the first poll resolves nothing is "ready" yet — fall back to every
  // listed host so the picker isn't empty during boot.
  const ids = readyHostIds.value.length
    ? readyHostIds.value
    : hosts.value.map((h) => h.id);
  return unionModels(accessibleModelsByHost.value, ids);
});

/** Every model any reachable machine holds — the ⌘K palette's search corpus.
 * Unlike `targetModels` this deliberately ignores the pinned target: the
 * palette's whole point is that picking a model can move the target.
 *
 * Before the first poll nothing is ready yet, so machines still connecting
 * stand in — but an errored one never does, even though its last inventory is
 * still cached. Offering a model no reachable machine can run would let the
 * palette repin generation to an unavailable owner. */
const installedModels = computed<ModelInfoExtended[]>(() => {
  const ids = readyHostIds.value.length
    ? readyHostIds.value
    : hosts.value.filter((h) => h.status !== "error").map((h) => h.id);
  return unionModels(accessibleModelsByHost.value, ids).filter(
    (m) => m.downloaded,
  );
});

function gpuFrom(
  info: GpuInfoWithBackend | null | undefined,
): RoutableGpu | null {
  if (!info) return null;
  return {
    backend: info.backend ?? null,
    name: info.name,
    vramTotalMb: info.vram_total_mb,
  };
}

/**
 * Collapse a legacy status response only at the host-routing boundary.
 *
 * A model still has to fit on one worker, so "Most capable" compares the
 * largest usable device rather than aggregate VRAM. `/api/status.gpu_info`
 * historically describes GPU 0; using it when the additive `gpus` array is
 * present would make every other card invisible to routing.
 */
function gpuFromStatus(
  status: ServerStatus,
  devices: DeviceInfo[] | null,
): RoutableGpu | null {
  if (devices !== null) {
    const workers = devices.filter(
      (device) => device.schedulable && device.ordinal !== null,
    );
    if (!workers.length) return null;
    const strongest = workers.reduce((best, device) =>
      (device.memory.total_bytes ?? 0) > (best.memory.total_bytes ?? 0)
        ? device
        : best,
    );
    return {
      backend: strongest.backend,
      name: strongest.name,
      vramTotalMb:
        strongest.memory.total_bytes === null
          ? null
          : strongest.memory.total_bytes / 1024 ** 2,
    };
  }
  const legacy = status.gpu_info as GpuInfoWithBackend | null | undefined;
  const workers = status.gpus?.filter(
    (worker: GpuWorkerStatus) => worker.state !== "degraded",
  );
  if (!workers?.length) {
    return status.gpus != null ? null : gpuFrom(legacy);
  }
  const strongest = workers.reduce((best, worker) =>
    worker.vram_total_bytes > best.vram_total_bytes ? worker : best,
  );
  return {
    backend: legacy?.backend ?? null,
    name: strongest.name,
    vramTotalMb: strongest.vram_total_bytes / 1024 ** 2,
  };
}

async function pollHost(entry: HostEntry): Promise<void> {
  const generation = (pollGenerations.get(entry.id) ?? 0) + 1;
  pollGenerations.set(entry.id, generation);
  const [status, models, devices, queue, capabilities] =
    await Promise.allSettled([
      hostStatus(entry),
      hostModels(entry),
      hostDevices(entry),
      hostQueue(entry),
      hostCapabilities(entry),
    ]);
  const current = entries.value.find((candidate) => candidate.id === entry.id);
  if (
    pollGenerations.get(entry.id) !== generation ||
    !current ||
    current.url !== entry.url ||
    current.apiKey !== entry.apiKey
  ) {
    return;
  }
  if (status.status === "fulfilled") {
    const inventory =
      devices.status === "fulfilled" ? devices.value.devices : null;
    const generationReady =
      inventory !== null
        ? inventory.some((device) => device.schedulable)
        : status.value.gpus == null ||
          status.value.gpus.some((gpu) => gpu.state !== "degraded");
    const previousInstanceId =
      telemetry.value[entry.id]?.instanceId ?? entry.instanceId ?? null;
    const nextInstanceId = status.value.instance_id ?? null;
    if (previousInstanceId !== nextInstanceId) {
      routingAuthorityGeneration += 1;
    }
    telemetry.value = {
      ...telemetry.value,
      [entry.id]: {
        status: generationReady ? "ready" : "error",
        instanceId: nextInstanceId,
        queueDepth: status.value.queue_depth ?? null,
        gpu: gpuFromStatus(status.value, inventory),
        predictedCompletionMs:
          queue.status === "fulfilled" && queue.value.plan
            ? predictedCompletionUnixMs(queue.value.plan)
            : null,
        queue:
          queue.status === "fulfilled"
            ? { entries: queue.value.entries, plan: queue.value.plan ?? null }
            : (telemetry.value[entry.id]?.queue ?? null),
      },
    };
  } else {
    telemetry.value = {
      ...telemetry.value,
      [entry.id]: {
        status: "error",
        instanceId:
          telemetry.value[entry.id]?.instanceId ?? entry.instanceId ?? null,
        queueDepth: null,
        gpu: null,
        predictedCompletionMs: null,
        queue: null,
      },
    };
  }
  if (models.status === "fulfilled") {
    modelsByHost.value = { ...modelsByHost.value, [entry.id]: models.value };
    if (!inventoryHostIds.value.includes(entry.id)) {
      inventoryHostIds.value = [...inventoryHostIds.value, entry.id];
    }
  } else if (!modelsByHost.value[entry.id]) {
    // Keep the last good list for a host that blipped; only seed an empty one.
    modelsByHost.value = { ...modelsByHost.value, [entry.id]: [] };
  }
  if (capabilities.status === "fulfilled") {
    const previous = capabilitiesByHost.value[entry.id];
    if (
      JSON.stringify(previous?.model_access ?? null) !==
      JSON.stringify(capabilities.value.model_access ?? null)
    ) {
      routingAuthorityGeneration += 1;
    }
    capabilitiesByHost.value = {
      ...capabilitiesByHost.value,
      [entry.id]: capabilities.value,
    };
  }
  if (!settledHostIds.value.includes(entry.id)) {
    settledHostIds.value = [...settledHostIds.value, entry.id];
  }
}

async function refresh(): Promise<void> {
  readRegistry();
  await Promise.all(entries.value.map((entry) => pollHost(entry)));
  modelsSettled.value = entries.value.every((e) =>
    settledHostIds.value.includes(e.id),
  );
}

let timer: ReturnType<typeof setTimeout> | null = null;
let consumers = 0;

function onHostsChanged(): void {
  void refresh();
}

function onGenerateTargetChanged(): void {
  readTarget();
}

function tick(): void {
  timer = setTimeout(() => {
    void refresh().finally(() => {
      if (consumers > 0) tick();
    });
  }, POLL_INTERVAL_MS);
}

function start(): void {
  consumers += 1;
  if (consumers > 1) return;
  window.addEventListener(HOSTS_CHANGED_EVENT, onHostsChanged);
  window.addEventListener(
    GENERATE_TARGET_CHANGED_EVENT,
    onGenerateTargetChanged,
  );
  readRegistry();
  readTarget();
  void refresh();
  tick();
}

function stop(): void {
  consumers = Math.max(0, consumers - 1);
  if (consumers > 0) return;
  window.removeEventListener(HOSTS_CHANGED_EVENT, onHostsChanged);
  window.removeEventListener(
    GENERATE_TARGET_CHANGED_EVENT,
    onGenerateTargetChanged,
  );
  if (timer) clearTimeout(timer);
  timer = null;
}

function setTarget(id: string): void {
  if (rawTargetId.value !== id) {
    rawTargetId.value = id;
    routingAuthorityGeneration += 1;
    targetPolicyGeneration += 1;
  }
  setGenerateTargetId(id);
}

/** Hosts that hold `model` downloaded — the model-aware routing input. */
function hostsForModel(model: string | null): string[] {
  return model ? hostIdsForModel(accessibleModelsByHost.value, model) : [];
}

function inventoryKnown(hostId: string): boolean {
  return inventoryHostIds.value.includes(hostId);
}

function withReferenceUploads(route: HostRoute | null): HostRoute | null {
  if (!route) return null;
  return {
    ...route,
    target: { ...route.target },
    referenceUploads:
      capabilitiesByHost.value[route.hostId]?.reference_uploads ?? null,
  };
}

function resolve(model: string | null): HostRoute | null {
  const selection = targetId.value;
  if (
    model &&
    selection !== AUTO_TARGET_ID &&
    selection !== CAPABLE_TARGET_ID &&
    accessRestrictionForHost(selection, model)
  ) {
    return null;
  }
  const eligible = model
    ? hosts.value.filter((host) => !accessRestrictionForHost(host.id, model))
    : hosts.value;
  if (
    model &&
    (selection === AUTO_TARGET_ID || selection === CAPABLE_TARGET_ID) &&
    profileHashConflict(
      accessibleModelsByHost.value,
      model,
      eligible.filter((host) => host.status === "ready").map((host) => host.id),
    )
  ) {
    return null;
  }
  return withReferenceUploads(
    resolveRoute(eligible, selection, hostsForModel(model)),
  );
}

async function resolveFeasibleWithPreview(
  model: string,
  previewFor: (target: ApiTarget) => Promise<GenerationPlacementPreview>,
  requireAuthoritative = false,
  authorityRetry = 0,
): Promise<FeasibilityResult> {
  readRegistry();
  readTarget();
  const authorityGeneration = routingAuthorityGeneration;
  const selection = targetId.value;
  let candidates = hosts.value.filter(
    (candidate) =>
      candidate.id === ORIGIN_HOST_ID || candidate.status === "ready",
  );
  if (selection !== AUTO_TARGET_ID && selection !== CAPABLE_TARGET_ID) {
    candidates = candidates.filter((candidate) => candidate.id === selection);
  }
  const restricted = candidates.flatMap((candidate) => {
    const restriction = accessRestrictionForHost(candidate.id, model);
    return restriction
      ? [
          {
            hostId: candidate.id,
            label: candidate.label,
            reason: restriction.message,
            missingComponents: [],
          },
        ]
      : [];
  });
  candidates = candidates.filter(
    (candidate) => !accessRestrictionForHost(candidate.id, model),
  );
  if (candidates.length === 0 && restricted.length > 0) {
    return { kind: "infeasible", perHost: restricted };
  }
  if (selection === AUTO_TARGET_ID || selection === CAPABLE_TARGET_ID) {
    const conflict = profileHashConflict(
      accessibleModelsByHost.value,
      model,
      candidates.map((candidate) => candidate.id),
    );
    if (conflict) {
      return {
        kind: "profile_mismatch",
        perHost: conflict.hostIds.map((hostId) => ({
          hostId,
          label:
            candidates.find((candidate) => candidate.id === hostId)?.label ??
            hostId,
          profileHash: conflict.hashesByHost[hostId] ?? null,
        })),
      };
    }
  }
  const probes = await Promise.all(
    candidates.map(async (candidate) => {
      const entry = entries.value.find((item) => item.id === candidate.id);
      if (!entry)
        return {
          candidate,
          preview: null,
          legacyUnsupported: false,
          error: "machine disappeared from the host registry",
          roundTripMs: 0,
        };
      const started = performance.now();
      try {
        const target = {
          baseUrl: entry.url,
          apiKey: entry.apiKey ?? null,
        };
        const preview = await previewFor(target);
        return {
          candidate,
          preview,
          legacyUnsupported: false,
          error: null,
          roundTripMs: Math.max(0, performance.now() - started),
        };
      } catch (error) {
        const probeError =
          error instanceof ApiError
            ? `HTTP ${error.status}: ${error.message}`
            : error instanceof Error
              ? error.message
              : String(error);
        return {
          candidate,
          preview: null,
          legacyUnsupported:
            error instanceof ApiError &&
            (error.status === 404 || error.status === 405),
          error: probeError,
          roundTripMs: Math.max(0, performance.now() - started),
        };
      }
    }),
  );

  readRegistry();
  readTarget();
  if (routingAuthorityGeneration !== authorityGeneration) {
    if (authorityRetry === 0) {
      return resolveFeasibleWithPreview(
        model,
        previewFor,
        requireAuthoritative,
        1,
      );
    }
    return {
      kind: "transient",
      perHost: [],
    };
  }

  const currentById = new Map(hosts.value.map((host) => [host.id, host]));
  const planned = probes
    .flatMap((probe) =>
      probe.preview && classifyPlacementPreview(probe.preview) === "planned"
        ? [
            {
              candidate: probe.candidate,
              preview: probe.preview,
              roundTripMs: probe.roundTripMs,
            },
          ]
        : [],
    )
    .filter((probe) => currentById.has(probe.candidate.id))
    .map((probe) => ({
      hostId: probe.candidate.id,
      roundTripMs: probe.roundTripMs,
      preview: probe.preview,
    }))
    .sort(comparePlacementPreviews);
  if (planned.length > 0) {
    const chosen = currentById.get(planned[0].hostId);
    if (chosen) {
      const target: HostRoute["target"] = { baseUrl: chosen.url };
      if (chosen.apiKey) target.apiKey = chosen.apiKey;
      return {
        kind: "route",
        route: {
          hostId: chosen.id,
          label: chosen.label,
          target,
          instanceId: chosen.instanceId ?? null,
          referenceUploads:
            capabilitiesByHost.value[chosen.id]?.reference_uploads ?? null,
        },
      };
    }
  }

  const unsupportedIds = probes
    .filter(
      (probe) =>
        probe.legacyUnsupported ||
        classifyPlacementPreview(probe.preview) === "unsupported",
    )
    .map((probe) => probe.candidate.id);
  const legacy = hosts.value.filter(
    (candidate) =>
      unsupportedIds.includes(candidate.id) &&
      (candidate.id === ORIGIN_HOST_ID || candidate.status === "ready"),
  );
  if (!requireAuthoritative && legacy.length > 0) {
    const originRoutable = legacy.map((host) =>
      host.id === ORIGIN_HOST_ID ? { ...host, status: "ready" as const } : host,
    );
    const legacyModelIds = hostsForModel(model).filter((id) =>
      unsupportedIds.includes(id),
    );
    const route = resolveRoute(originRoutable, selection, legacyModelIds);
    if (route) return { kind: "route", route: withReferenceUploads(route)! };
  }

  const transient = probes.flatMap((probe) => {
    if (
      !probe.preview ||
      classifyPlacementPreview(probe.preview) !== "temporarily_unavailable"
    ) {
      return [];
    }
    return [
      {
        hostId: probe.candidate.id,
        label: probe.candidate.label,
        reason:
          probe.preview.reason ??
          "could not compute a placement plan right now",
      },
    ];
  });
  const infeasible = probes.flatMap((probe) => {
    if (
      !probe.preview ||
      classifyPlacementPreview(probe.preview) !== "infeasible"
    ) {
      return [];
    }
    return [
      {
        hostId: probe.candidate.id,
        label: probe.candidate.label,
        reason: probe.preview.reason!,
        missingComponents: probe.preview.missing_components ?? [],
      },
    ];
  });
  const unreachable = probes
    .filter((probe) => {
      const classification = classifyPlacementPreview(probe.preview);
      return (
        (requireAuthoritative || !probe.legacyUnsupported) &&
        (!probe.preview ||
          classification === "invalid" ||
          (requireAuthoritative && classification === "unsupported"))
      );
    })
    .map((probe) => ({
      hostId: probe.candidate.id,
      label: probe.candidate.label,
      error:
        requireAuthoritative &&
        (probe.legacyUnsupported ||
          classifyPlacementPreview(probe.preview) === "unsupported")
          ? "does not provide the authoritative placement preview required for reference media"
          : (probe.error ??
            "returned an invalid authoritative placement-preview response"),
    }));
  if (transient.length > 0) {
    return {
      kind: "transient",
      perHost: transient,
      ...(infeasible.length > 0 ? { infeasible } : {}),
      ...(unreachable.length > 0 ? { unreachable } : {}),
    };
  }
  if (infeasible.length > 0) {
    return {
      kind: "infeasible",
      perHost: infeasible,
      ...(unreachable.length > 0 ? { unreachable } : {}),
    };
  }
  if (unreachable.length > 0)
    return { kind: "unreachable", perHost: unreachable };

  const selectedHosts =
    selection === AUTO_TARGET_ID || selection === CAPABLE_TARGET_ID
      ? hosts.value
      : hosts.value.filter((host) => host.id === selection);
  return {
    kind: "unreachable",
    perHost: selectedHosts.map((host) => ({
      hostId: host.id,
      label: host.label,
      error:
        host.status === "connecting"
          ? "is still connecting"
          : "is not ready for generation",
    })),
  };
}

async function resolveFeasible(
  request: GenerateRequestWire,
  copies = 1,
): Promise<FeasibilityResult> {
  return resolveFeasibleWithPreview(
    request.model,
    (target) =>
      previewGenerationPlacement(
        target,
        previewRequestForSiblingFanout(
          request as unknown as Record<string, unknown>,
          copies,
        ),
        copies,
      ),
    requiresAuthoritativePlacement(
      request as unknown as Record<string, unknown>,
    ),
  );
}

async function resolveFeasibleChain(
  request: ChainRequestWire,
  copies = 1,
): Promise<FeasibilityResult> {
  return resolveFeasibleWithPreview(request.model, (target) =>
    previewChainPlacement(
      target,
      previewRequestForSiblingFanout(
        request as unknown as Record<string, unknown>,
        copies,
      ),
      copies,
    ),
  );
}

async function revalidateFeasibleWithPreview(
  route: HostRoute,
  model: string,
  previewFor: (target: ApiTarget) => Promise<GenerationPlacementPreview>,
  requireAuthoritative = false,
  authorityRetry = 0,
): Promise<FeasibilityResult> {
  readRegistry();
  readTarget();
  const authorityGeneration = routingAuthorityGeneration;
  const capturedTargetPolicyGeneration = targetPolicyGeneration;
  const captured = hosts.value.find((entry) => entry.id === route.hostId);
  const restriction = accessRestrictionForHost(route.hostId, model);
  if (restriction) {
    return {
      kind: "infeasible",
      perHost: [
        {
          hostId: route.hostId,
          label: route.label,
          reason: restriction.message,
          missingComponents: [],
        },
      ],
    };
  }
  if (
    !captured ||
    captured.url !== route.target.baseUrl ||
    (captured.apiKey ?? undefined) !== route.target.apiKey ||
    (captured.instanceId ?? null) !== (route.instanceId ?? null)
  ) {
    return { kind: "transient", perHost: [] };
  }
  const capturedInstanceId = captured.instanceId ?? null;
  let preview: GenerationPlacementPreview | null = null;
  let legacyUnsupported = false;
  try {
    preview = await previewFor({
      baseUrl: route.target.baseUrl,
      apiKey: route.target.apiKey ?? null,
    });
  } catch (error) {
    if (
      error instanceof ApiError &&
      (error.status === 404 || error.status === 405)
    ) {
      legacyUnsupported = true;
    } else {
      return {
        kind: "unreachable",
        perHost: [
          {
            hostId: route.hostId,
            label: route.label,
            error:
              error instanceof ApiError
                ? `HTTP ${error.status}: ${error.message}`
                : error instanceof Error
                  ? error.message
                  : String(error),
          },
        ],
      };
    }
  }
  readRegistry();
  readTarget();
  if (targetPolicyGeneration !== capturedTargetPolicyGeneration) {
    return { kind: "transient", perHost: [] };
  }
  if (routingAuthorityGeneration !== authorityGeneration) {
    if (authorityRetry === 0) {
      return revalidateFeasibleWithPreview(
        route,
        model,
        previewFor,
        requireAuthoritative,
        1,
      );
    }
    return { kind: "transient", perHost: [] };
  }
  const current = hosts.value.find((entry) => entry.id === route.hostId);
  if (
    !current ||
    current.url !== route.target.baseUrl ||
    (current.apiKey ?? undefined) !== route.target.apiKey ||
    (current.instanceId ?? null) !== capturedInstanceId
  ) {
    return { kind: "transient", perHost: [] };
  }
  const classification = legacyUnsupported
    ? "unsupported"
    : classifyPlacementPreview(preview);
  if (classification === "temporarily_unavailable") {
    return {
      kind: "transient",
      perHost: [
        {
          hostId: route.hostId,
          label: route.label,
          reason:
            preview?.reason ?? "could not compute a placement plan right now",
        },
      ],
    };
  }
  if (classification === "infeasible") {
    return {
      kind: "infeasible",
      perHost: [
        {
          hostId: route.hostId,
          label: route.label,
          reason: preview!.reason!,
          missingComponents: preview!.missing_components ?? [],
        },
      ],
    };
  }
  if (classification === "invalid") {
    return {
      kind: "unreachable",
      perHost: [
        {
          hostId: route.hostId,
          label: route.label,
          error: "returned an invalid authoritative placement-preview response",
        },
      ],
    };
  }
  if (classification === "unsupported" && requireAuthoritative) {
    return {
      kind: "unreachable",
      perHost: [
        {
          hostId: route.hostId,
          label: route.label,
          error:
            "does not provide the authoritative placement preview required for reference media",
        },
      ],
    };
  }
  return {
    kind: "route",
    route: {
      hostId: route.hostId,
      label: route.label,
      target: { ...route.target },
      instanceId: capturedInstanceId,
      referenceUploads:
        capabilitiesByHost.value[route.hostId]?.reference_uploads ?? null,
    },
  };
}

async function revalidateFeasible(
  route: HostRoute,
  request: GenerateRequestWire,
  copies = 1,
): Promise<FeasibilityResult> {
  return revalidateFeasibleWithPreview(
    route,
    request.model,
    (target) =>
      previewGenerationPlacement(
        target,
        previewRequestForSiblingFanout(
          request as unknown as Record<string, unknown>,
          copies,
        ),
        copies,
      ),
    requiresAuthoritativePlacement(
      request as unknown as Record<string, unknown>,
    ),
  );
}

async function revalidateFeasibleChain(
  route: HostRoute,
  request: ChainRequestWire,
  copies = 1,
): Promise<FeasibilityResult> {
  return revalidateFeasibleWithPreview(route, request.model, (target) =>
    previewChainPlacement(
      target,
      previewRequestForSiblingFanout(
        request as unknown as Record<string, unknown>,
        copies,
      ),
      copies,
    ),
  );
}

export function useHostRouting(): HostRouting {
  start();
  onBeforeUnmount(stop);
  return {
    hosts,
    targetId,
    setTarget,
    multiHost: computed(() => hosts.value.length > 1),
    targetModels,
    installedModels,
    modelsSettled,
    modelOwnerIds: hostsForModel,
    inventoryKnown,
    capabilitiesByHost,
    queueStatus,
    hostMemoryPressure,
    resolve,
    resolveFeasible,
    revalidateFeasible,
    resolveFeasibleChain,
    revalidateFeasibleChain,
    refresh,
    start,
    stop,
  };
}

/** Reset the singleton between tests. */
export const __testing__ = {
  POLL_INTERVAL_MS,
  reset(): void {
    window.removeEventListener(HOSTS_CHANGED_EVENT, onHostsChanged);
    window.removeEventListener(
      GENERATE_TARGET_CHANGED_EVENT,
      onGenerateTargetChanged,
    );
    if (timer) clearTimeout(timer);
    timer = null;
    consumers = 0;
    entries.value = [];
    telemetry.value = {};
    modelsByHost.value = {};
    capabilitiesByHost.value = {};
    settledHostIds.value = [];
    inventoryHostIds.value = [];
    modelsSettled.value = false;
    rawTargetId.value = "";
    pollGenerations.clear();
    routingAuthorityGeneration = 0;
    targetPolicyGeneration = 0;
    registryAuthoritySignature = "";
  },
};

export { ORIGIN_HOST_ID };
