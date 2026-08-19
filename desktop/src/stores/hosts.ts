import { defineStore } from "pinia";
import { modelAccessRestrictionFor } from "@studio/lib/modelAccess";
import { profileHashConflict } from "@studio/lib/profileFleet";
import { listDevices, type DeviceInfo } from "@studio/api/devices";
import { listQueue, predictedCompletionUnixMs } from "@studio/api/queuePlan";
import {
  comparePlacementPreviews,
  classifyPlacementPreview,
  previewChainPlacement,
  previewGenerationPlacement,
  previewRequestForSiblingFanout,
  requiresAuthoritativePlacement,
  type PlacementMissingComponent,
} from "@studio/api/generationPlacement";
import { ApiError, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { fetchServerCapabilities } from "../lib/api/serverCapabilities";
import {
  hostIdFromUrl,
  hostnamesCompatible,
  mergeSavedHostsByInstanceId,
  normalizeHostUrl,
  pickAutoHost,
  pickMostCapableHost,
} from "../lib/hosts";
import { ipc, type SavedHost } from "../lib/ipc";
import { PLATFORM_UI } from "../lib/platform";
import type {
  AutoChainRequest,
  ChainRequest,
  GenerateRequest,
  GpuInfo,
  GpuWorkerStatus,
  ServerCapabilities,
  ServerStatus,
} from "../lib/api/types";
import type { ReferenceUploadCapabilities } from "@studio/api/referenceUploads";
import { useAppPrefsStore } from "./appPrefs";
import { useConnectionStore } from "./connection";
import { useDownloadsStore } from "./downloads";
import { useHostModelsStore } from "./hostModels";

const POLL_INTERVAL_MS = 10_000;
const MAX_SAVED_HOSTS = 8;
/** Once Auto has one authoritative route, allow normal LAN peers a brief
 * window to return a better plan, then stop waiting on cold artifact checks.
 * Explicitly selected hosts retain the full placement-preview deadline. */
const AUTO_PLACEMENT_SETTLE_MS = 250;
/** Auto must remain a bounded interaction even when no candidate ever answers.
 * A pinned machine may legitimately spend minutes authenticating cold weights;
 * Auto instead fails closed and tells the user to retry or pin that machine. */
const AUTO_PLACEMENT_DEADLINE_MS = 5_000;

/**
 * Serialize this store's settings read-modify-writes: `app_settings_set`
 * replaces the whole file last-writer-wins, so two overlapping
 * get→mutate→set cycles (a user action racing the 10 s poll's reconcile)
 * would silently drop one writer's change. Every RMW in this store runs
 * through this chain; failures don't wedge it.
 */
let settingsWriteChain: Promise<unknown> = Promise.resolve();
function withSettingsLock<T>(fn: () => Promise<T>): Promise<T> {
  const run = settingsWriteChain.then(fn, fn);
  settingsWriteChain = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

/** An additional live connection beyond the primary one. */
export interface ExtraHost {
  id: string;
  label: string;
  url: string;
  apiKey: string | null;
  status: "connecting" | "ready" | "error";
  error: string | null;
  /** Stable server-installation UUID (from the connect probe / saved entry);
   *  the refresh poll's telemetry value supersedes it once live. */
  instanceId: string | null;
}

/** Unified host row for the sidebar, the selector, and routing. */
export interface HostView {
  id: string;
  label: string;
  kind: "local" | "remote";
  baseUrl: string | null;
  apiKey: string | null;
  status: "connecting" | "ready" | "error";
  primary: boolean;
  queueDepth: number | null;
  queueCapacity: number | null;
  predictedCompletionMs?: number | null;
  version: string | null;
  /** Stable server-installation UUID; null until learned. Dedupe key. */
  instanceId: string | null;
}

export interface HostTelemetry {
  queueDepth: number | null;
  queueCapacity: number | null;
  predictedCompletionMs?: number | null;
  version: string | null;
  /** Loaded-model names from `/api/status` (absent before the first poll). */
  modelsLoaded?: string[];
  /** GPU summary from `/api/status`; the status bar's fallback when a host's
   *  resources stream is unavailable. */
  gpuInfo?: GpuInfo | null;
  /** Every runtime worker from `/api/status.gpus`. Routing uses the largest
   * healthy device because one model must fit on one worker. */
  gpuWorkers?: GpuWorkerStatus[] | null;
  /** Authoritative current-server inventory. Null means `/api/devices` is
   * unsupported and routing must use the legacy status fields. */
  devices?: DeviceInfo[] | null;
  /** Stable server-installation UUID from `/api/status`; absent on older servers. */
  instanceId?: string | null;
  /** Server-reported hostname from `/api/status`; drives the display label. */
  hostname?: string | null;
}

/** Exported so a view can rank an arbitrary subset of hosts (expansion
 *  routing) with the same GPU summary this store's own routers use. */
export function strongestRoutableGpu(telemetry: HostTelemetry | undefined) {
  if (telemetry?.devices != null) {
    const devices = telemetry.devices.filter(
      (device) => device.schedulable && device.ordinal !== null,
    );
    if (!devices.length) return null;
    const strongest = devices.reduce((best, device) =>
      (device.memory.total_bytes ?? 0) > (best.memory.total_bytes ?? 0) ? device : best,
    );
    return {
      backend: strongest.backend,
      name: strongest.name,
      vramTotalMb:
        strongest.memory.total_bytes === null ? null : strongest.memory.total_bytes / 1024 ** 2,
    };
  }
  const workers = telemetry?.gpuWorkers?.filter((worker) => worker.state !== "degraded");
  if (workers?.length) {
    const strongest = workers.reduce((best, worker) =>
      worker.vram_total_bytes > best.vram_total_bytes ? worker : best,
    );
    return {
      backend: telemetry?.gpuInfo?.backend ?? null,
      name: strongest.name,
      vramTotalMb: strongest.vram_total_bytes / 1024 ** 2,
    };
  }
  if (telemetry?.gpuWorkers != null) return null;
  const legacy = telemetry?.gpuInfo;
  return legacy
    ? {
        backend: legacy.backend ?? null,
        name: legacy.name,
        vramTotalMb: legacy.vram_total_mb,
      }
    : null;
}

/** Where a batch will run: resolved just before submit. */
export interface HostRoute {
  hostId: string;
  label: string;
  kind: "local" | "remote";
  target: ApiTarget;
  /** Optional stable server identity used by remote-only clients to detect
   * when one saved endpoint now reaches a different Mold installation. */
  instanceId?: string | null;
  /** Frozen authenticated reference-ingress contract for this exact host. */
  referenceUploads?: ReferenceUploadCapabilities | null;
}

export interface HostPlacementFailure {
  kind: "infeasible";
  hostId: string;
  label: string;
  route: HostRoute;
  reason: string;
  missingComponents: PlacementMissingComponent[];
}

export interface HostProbeFailure {
  kind: "transient" | "unreachable";
  hostId: string;
  label: string;
  error: string;
}

export type HostFeasibilityFailure = HostPlacementFailure | HostProbeFailure;

export type FeasibleRouteResult =
  | { kind: "route"; route: HostRoute }
  | {
      kind: "profile_mismatch";
      perHost: Array<{
        hostId: string;
        label: string;
        profileHash: string | null;
        version: string | null;
      }>;
    }
  | { kind: "infeasible"; perHost: HostPlacementFailure[] }
  | { kind: "unreachable"; perHost: HostProbeFailure[] }
  | { kind: "transient"; perHost: HostProbeFailure[] }
  | { kind: "mixed"; perHost: HostFeasibilityFailure[] };

function hostRoute(host: HostView, capabilities?: ServerCapabilities): HostRoute | null {
  if (!host.baseUrl) return null;
  return {
    hostId: host.id,
    label: host.label,
    kind: host.kind,
    target: { baseUrl: host.baseUrl, apiKey: host.apiKey },
    instanceId: host.instanceId,
    referenceUploads: capabilities?.reference_uploads ?? null,
  };
}

function probeError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail =
      typeof error.body === "string"
        ? error.body.trim()
        : error.body && typeof error.body === "object" && "error" in error.body
          ? String((error.body as { error?: unknown }).error ?? "").trim()
          : "";
    return `placement preview returned HTTP ${error.status}${detail ? ` — ${detail}` : ""}`;
  }
  return error instanceof Error ? error.message : String(error);
}

function accessRestrictionForHost(
  hostId: string,
  model: string,
  capabilities: ServerCapabilities | undefined,
) {
  const entry = useHostModelsStore().byHost[hostId]?.entries.find(
    (candidate) => candidate.name === model,
  );
  return modelAccessRestrictionFor(capabilities, {
    model,
    family: entry?.family,
    generation_profile_sha256: entry?.generation_profile?.profile_hash ?? null,
  });
}

/**
 * Multi-host registry. The primary connection (connection store) stays the
 * app-wide default that Models/Gallery/Settings talk to; this store adds
 * extra live hosts that generation jobs can queue on concurrently. Remote
 * hosts are pure client-side state — a URL and a key — so nothing here
 * touches the Rust connection machinery.
 */
export const useHostsStore = defineStore("hosts", {
  state: () => ({
    extras: [] as ExtraHost[],
    telemetry: {} as Record<string, HostTelemetry>,
    /** Last capability snapshot per live host. A successful older-host
     *  payload may omit `expand`; that absence is deliberately "unknown". */
    capabilities: {} as Record<string, ServerCapabilities>,
    /** Friendly names from savedHosts, so the primary can be renamed too. */
    names: {} as Record<string, string>,
    /** Last-known server-reported hostname per host id. Sticky — unlike
     *  telemetry it survives failed polls, keeping labels stable and letting
     *  instance-id dedupe stay hostname-qualified while a host is down. */
    hostnames: {} as Record<string, string>,
    refreshGenerations: {} as Record<string, number>,
    pollTimer: null as ReturnType<typeof setInterval> | null,
    initializing: false,
    initialized: false,
  }),
  getters: {
    /** The primary connection as a host row. The built-in engine is always the
     *  internal primary (host id "local"); remotes are additive list entries. */
    primaryHost(state): HostView | null {
      const conn = useConnectionStore();
      if (!conn.info?.baseUrl) return null;
      const t = state.telemetry.local;
      return {
        id: "local",
        label: PLATFORM_UI.deviceLabel,
        kind: "local",
        baseUrl: conn.info.baseUrl,
        apiKey: conn.info.apiKey,
        status:
          conn.status === "ready" ? "ready" : conn.status === "starting" ? "connecting" : "error",
        primary: true,
        queueDepth: t?.queueDepth ?? null,
        queueCapacity: t?.queueCapacity ?? null,
        predictedCompletionMs: t?.predictedCompletionMs ?? null,
        version: t?.version ?? null,
        instanceId: t?.instanceId ?? null,
      };
    },
    /** Every host, the local primary first. Dedupe extras by id, loopback URL,
     *  and instance id — the local primary's id is the literal "local", so a
     *  loopback extra pointing at the same server would otherwise be listed
     *  (and routed to) twice. */
    all(state): HostView[] {
      const primary = this.primaryHost;
      const rows: HostView[] = primary ? [primary] : [];
      const hostnameOf = (id: string): string | null =>
        state.hostnames[id] ?? state.telemetry[id]?.hostname ?? null;
      for (const extra of state.extras) {
        const t = state.telemetry[extra.id];
        const instanceId = t?.instanceId ?? extra.instanceId ?? null;
        // Dedupe by id, loopback URL, AND instance id — the same physical
        // server reached by hostname vs IP has one row, not two. An instance-id
        // match only counts when the reported hostnames agree (or one is
        // unknown): distinct servers sharing a MOLD_HOME share a uuid too.
        if (
          rows.some(
            (row) =>
              row.id === extra.id ||
              (row.baseUrl && row.baseUrl === extra.url) ||
              (instanceId !== null &&
                row.instanceId === instanceId &&
                hostnamesCompatible(hostnameOf(row.id), hostnameOf(extra.id))),
          )
        )
          continue;
        rows.push({
          id: extra.id,
          // User rename wins; otherwise the last-known server hostname (sticky
          // across failed polls); otherwise the URL.
          label: this.names[extra.id] ?? hostnameOf(extra.id) ?? extra.label,
          kind: "remote",
          baseUrl: extra.url,
          apiKey: extra.apiKey,
          status: extra.status,
          primary: false,
          queueDepth: t?.queueDepth ?? null,
          queueCapacity: t?.queueCapacity ?? null,
          predictedCompletionMs: t?.predictedCompletionMs ?? null,
          version: t?.version ?? null,
          instanceId,
        });
      }
      return rows;
    },
    /** True when the generation view should surface a host selector. */
    multiHost(): boolean {
      return this.all.length > 1;
    },
  },
  actions: {
    /** Reconnect every explicitly connected extra host. Never blocks or fails the boot. */
    async init() {
      if (this.initialized || this.initializing) return;
      this.initializing = true;
      try {
        const settings = await ipc.appSettingsGet();
        for (const saved of settings.savedHosts) {
          if (saved.name) this.names[saved.id] = saved.name;
        }
        // Remembering and connecting are intentionally separate. A host that
        // was explicitly disconnected stays out of routing, polling, gallery
        // merges, and downloads until the user reconnects it.
        const candidates = [...new Set(settings.connectedHostIds)]
          .map((id) => settings.savedHosts.find((host) => host.id === id))
          .filter((saved): saved is SavedHost => Boolean(saved))
          .filter((saved) => !this.extras.some((host) => host.id === saved.id));
        // Add the connecting rows synchronously so their MRU order is stable
        // even though secret reads and network probes run concurrently.
        for (const saved of candidates) {
          this.extras.push({
            id: saved.id,
            label: saved.name ?? saved.url.replace(/^https?:\/\//, ""),
            url: saved.url,
            apiKey: null,
            status: "connecting",
            error: null,
            instanceId: saved.instanceId ?? null,
          });
        }
        await Promise.all(
          candidates.map(async (saved) => {
            const id = saved.id;
            const key = await ipc.secretGet(`remote-api-key.${id}`);
            const live = this.extras.find((h) => h.id === id)!;
            live.apiKey = key;
            const test = await ipc.testRemoteHost(saved.url, key);
            // Seed the sticky hostname from the probe so the row is labeled by
            // the server's name (not the raw URL) before the first poll.
            if (test.hostname) this.hostnames[id] = test.hostname;
            if (test.ok) {
              live.status = "ready";
            } else {
              live.status = "error";
              live.error = test.error;
              // Boot probes are intentionally quiet. The Machines status row
              // is sufficient; an offline server must not create a startup
              // notification every time the app launches.
            }
          }),
        );
      } catch {
        // Settings unreadable — multi-host simply starts empty.
      } finally {
        this.initializing = false;
        this.initialized = true;
      }
      this.startPolling();
    },
    /** Connect an additional host (validates first) and remember it. */
    async connect(rawUrl: string, apiKey: string | null, name: string | null): Promise<HostView> {
      const url = normalizeHostUrl(rawUrl);
      const id = hostIdFromUrl(url);
      // `https://local` slugs to the literal "local", colliding with the
      // built-in engine's reserved id — refuse it rather than shadow the
      // primary's secret/routing keys.
      if (id === "local") throw new Error("That address is reserved for the built-in engine.");
      const existing = this.all.find((h) => h.id === id);
      if (existing?.status === "ready") return existing;

      const test = await ipc.testRemoteHost(url, apiKey);
      if (!test.ok) throw new Error(test.error ?? "Connection failed.");
      const instanceId = test.instanceId ?? null;

      // Same physical server reached by a different address (hostname vs IP vs
      // mDNS name): return the existing row instead of a duplicate. The probe
      // just succeeded with the caller's credentials, so a freshly supplied key
      // is authoritative for that box (a stale stored key must not block a
      // rotation). When the twin's own address is dead (DHCP re-lease,
      // recreated RunPod pod), adopt the validated address onto the twin's
      // surviving slug — keeping its per-host secret and reconnect entry —
      // instead of returning the dead row untouched. The primary "local" row
      // has no extra, so it is returned as-is.
      if (instanceId !== null) {
        // Hostname-qualified: a shared uuid with a DIFFERENT reported hostname
        // is two servers sharing a MOLD_HOME, not one server on two addresses.
        const twin = this.all.find(
          (h) =>
            h.instanceId === instanceId &&
            hostnamesCompatible(
              this.hostnames[h.id] ?? this.telemetry[h.id]?.hostname,
              test.hostname,
            ),
        );
        if (twin) {
          if (test.hostname) this.hostnames[twin.id] = test.hostname;
          if (apiKey) await ipc.secretSet(`remote-api-key.${twin.id}`, apiKey);
          const live = this.extras.find((h) => h.id === twin.id);
          if (live) {
            if (apiKey) live.apiKey = apiKey;
            if (twin.status !== "ready") {
              live.url = url;
              live.status = "ready";
              live.error = null;
              live.instanceId = instanceId;
              await this.persist(twin.id, url, name, instanceId);
              void this.refresh();
            }
          }
          return this.all.find((h) => h.id === twin.id) ?? twin;
        }
      }

      this.extras = this.extras.filter((h) => h.id !== id);
      this.extras.push({
        id,
        label: name ?? test.hostname ?? url.replace(/^https?:\/\//, ""),
        url,
        apiKey,
        status: "ready",
        error: null,
        instanceId,
      });
      if (test.hostname) this.hostnames[id] = test.hostname;
      if (apiKey) await ipc.secretSet(`remote-api-key.${id}`, apiKey);
      await this.persist(id, url, name, instanceId);
      void this.refresh();
      return this.all.find((h) => h.id === id)!;
    },
    /** Drop a live extra host. Its saved entry and key stay for later. */
    async disconnect(id: string) {
      useDownloadsStore().unsubscribeHost(id);
      this.extras = this.extras.filter((h) => h.id !== id);
      delete this.telemetry[id];
      delete this.capabilities[id];
      let clearedTarget = false;
      await withSettingsLock(async () => {
        const settings = await ipc.appSettingsGet();
        // A sticky generation target pointing at the removed host would only
        // linger as a dead preference (resolveRoute already falls back to
        // Auto) — clear it alongside the reconnect entry.
        clearedTarget = settings.generateTargetHost === id;
        await ipc.appSettingsSet({
          ...settings,
          connectedHostIds: settings.connectedHostIds.filter((h) => h !== id),
          generateTargetHost: clearedTarget ? null : settings.generateTargetHost,
        });
      });
      if (clearedTarget) {
        // Keep the in-memory prefs snapshot coherent (the Create header host menu reads it).
        const prefs = useAppPrefsStore();
        if (prefs.settings) prefs.settings = { ...prefs.settings, generateTargetHost: null };
      }
    },
    /** Give a host a friendly name (sidebar, host selector, Recent hosts). */
    async rename(id: string, name: string) {
      const trimmed = name.trim();
      if (!trimmed) return;
      this.names[id] = trimmed;
      const extra = this.extras.find((h) => h.id === id);
      if (extra) extra.label = trimmed;
      await withSettingsLock(async () => {
        const settings = await ipc.appSettingsGet();
        const known = settings.savedHosts.some((h) => h.id === id);
        // An adopted host whose MRU entry was pruned still deserves a sticky
        // name — re-insert it so the rename survives relaunch.
        const savedHosts = known
          ? settings.savedHosts.map((h) => (h.id === id ? { ...h, name: trimmed } : h))
          : extra
            ? [
                {
                  id,
                  name: trimmed,
                  url: extra.url,
                  lastUsedMs: Date.now(),
                  instanceId: extra.instanceId,
                },
                ...settings.savedHosts,
              ].slice(0, MAX_SAVED_HOSTS)
            : settings.savedHosts;
        await ipc.appSettingsSet({ ...settings, savedHosts });
      });
    },
    /** Retry a failed extra host in place. */
    async reconnect(id: string) {
      const extra = this.extras.find((h) => h.id === id);
      if (!extra) return;
      extra.status = "connecting";
      const test = await ipc.testRemoteHost(extra.url, extra.apiKey);
      extra.status = test.ok ? "ready" : "error";
      extra.error = test.ok ? null : test.error;
      if (test.ok) void this.refresh();
      else delete this.capabilities[id];
    },
    /**
     * Resolve where a batch should run. `null` = Auto (least busy);
     * `"capable"` = strongest GPU (backend > VRAM > queue). Both are
     * model-aware: when `modelName` is given and at least one ready host
     * already has it installed (per the hostModels store), routing is
     * restricted to those hosts — otherwise every ready host stays in play
     * and the winner auto-pulls. An explicit pick that is CONNECTED but not
     * ready resolves to null so the caller reports it instead of silently
     * rerouting; a pick whose host is gone entirely (disconnected,
     * forgotten) falls back to Auto — the selector already displays it as
     * Auto, and a stale persisted id must never wedge every Generate click.
     */
    resolveRoute(selection: string | null, modelName: string | null = null): HostRoute | null {
      const routable = this.all
        .filter((host) => {
          const telemetry = this.telemetry[host.id];
          if (telemetry?.devices != null)
            return telemetry.devices.some((device) => device.schedulable);
          const workers = telemetry?.gpuWorkers;
          return workers == null || workers.some((worker) => worker.state !== "degraded");
        })
        .map((h) => {
          return {
            ...h,
            gpu: strongestRoutableGpu(this.telemetry[h.id]),
          };
        })
        .filter(
          (host) =>
            !modelName || !accessRestrictionForHost(host.id, modelName, this.capabilities[host.id]),
        );
      const modelHostIds = modelName ? useHostModelsStore().hostsFor(modelName) : [];
      const automatic = selection === null || selection === "capable";
      if (
        modelName &&
        automatic &&
        profileHashConflict(
          Object.fromEntries(
            Object.entries(useHostModelsStore().byHost).map(([id, snapshot]) => [
              id,
              snapshot.entries,
            ]),
          ),
          modelName,
          routable.filter((host) => host.status === "ready").map((host) => host.id),
          Object.fromEntries(
            routable.map((host) => [host.id, this.telemetry[host.id]?.version ?? null]),
          ),
        )
      ) {
        return null;
      }

      let chosen: (typeof routable)[number] | null;
      if (selection === "capable") {
        chosen = pickMostCapableHost(routable, modelHostIds.length > 0 ? modelHostIds : null);
      } else if (selection !== null && this.all.some((h) => h.id === selection)) {
        chosen = routable.find((h) => h.id === selection && h.status === "ready") ?? null;
      } else {
        const withModel = routable.filter(
          (h) => h.status === "ready" && modelHostIds.includes(h.id),
        );
        chosen = pickAutoHost(withModel.length > 0 ? withModel : routable);
      }
      if (!chosen?.baseUrl) return null;
      return {
        hostId: chosen.id,
        label: chosen.label,
        kind: chosen.kind,
        target: { baseUrl: chosen.baseUrl, apiKey: chosen.apiKey },
        instanceId: chosen.instanceId,
        referenceUploads: this.capabilities[chosen.id]?.reference_uploads ?? null,
      };
    },
    async resolveFeasible(
      selection: string | null,
      request: GenerateRequest | ChainRequest | AutoChainRequest,
      copies = 1,
    ): Promise<FeasibleRouteResult> {
      const requireAuthoritative = requiresAuthoritativePlacement(
        request as unknown as Record<string, unknown>,
      );
      const intentSignature = () =>
        JSON.stringify({
          requestedSelection: selection,
          activeSelection: useAppPrefsStore().settings?.generateTargetHost ?? null,
        });
      const identitySignature = () =>
        JSON.stringify(
          this.all.map((host) => [host.id, host.baseUrl, host.apiKey, host.instanceId]),
        );
      const availabilitySignature = () =>
        JSON.stringify({
          hosts: this.all.map((host) => [host.id, host.status]),
          modelAccess: this.all.map((host) => [host.id, this.capabilities[host.id]?.model_access]),
        });
      const capturedIntent = intentSignature();
      const capturedIdentity = identitySignature();
      for (let attempt = 0; attempt < 2; attempt += 1) {
        const capturedAvailability = availabilitySignature();
        let candidates = this.all.filter((host) => {
          // The local server is the origin authority. While it is starting,
          // still probe it and permit the legacy path; a genuinely dead
          // listener then reports its transport error instead of being
          // silently discarded before dispatch.
          if (
            (host.status !== "ready" && !(host.kind === "local" && host.status === "connecting")) ||
            !host.baseUrl
          )
            return false;
          const telemetry = this.telemetry[host.id];
          if (telemetry?.devices != null)
            return telemetry.devices.some((device) => device.schedulable);
          return (
            telemetry?.gpuWorkers == null ||
            telemetry.gpuWorkers.some((worker) => worker.state !== "degraded")
          );
        });
        if (
          selection !== null &&
          selection !== "capable" &&
          this.all.some((host) => host.id === selection)
        )
          candidates = candidates.filter((host) => host.id === selection);

        const restricted = candidates.flatMap((host) => {
          const restriction = accessRestrictionForHost(
            host.id,
            request.model,
            this.capabilities[host.id],
          );
          const route = hostRoute(host, this.capabilities[host.id]);
          return restriction && route
            ? [
                {
                  kind: "infeasible" as const,
                  hostId: host.id,
                  label: host.label,
                  route,
                  reason: restriction.message,
                  missingComponents: [],
                },
              ]
            : [];
        });
        candidates = candidates.filter(
          (host) => !accessRestrictionForHost(host.id, request.model, this.capabilities[host.id]),
        );
        if (candidates.length === 0 && restricted.length > 0) {
          return { kind: "infeasible", perHost: restricted };
        }

        const automatic = selection === null || selection === "capable";
        const profileConflict = automatic
          ? profileHashConflict(
              Object.fromEntries(
                Object.entries(useHostModelsStore().byHost).map(([id, snapshot]) => [
                  id,
                  snapshot.entries,
                ]),
              ),
              request.model,
              candidates.map((host) => host.id),
              Object.fromEntries(
                candidates.map((host) => [host.id, this.telemetry[host.id]?.version ?? null]),
              ),
            )
          : null;
        if (profileConflict) {
          return {
            kind: "profile_mismatch",
            perHost: profileConflict.hostIds.map((hostId) => ({
              hostId,
              label: this.all.find((host) => host.id === hostId)?.label ?? hostId,
              profileHash: profileConflict.hashesByHost[hostId] ?? null,
              version: this.telemetry[hostId]?.version ?? null,
            })),
          };
        }

        if (candidates.length === 0) {
          const selected =
            selection && selection !== "capable"
              ? this.all.filter((host) => host.id === selection)
              : this.all;
          return {
            kind: "unreachable",
            perHost: selected.map((host) => ({
              kind: "unreachable" as const,
              hostId: host.id,
              label: host.label,
              error:
                host.status === "connecting"
                  ? "is still connecting"
                  : host.status === "error"
                    ? "is not reachable"
                    : "has no schedulable device",
            })),
          };
        }

        type PlacementProbe = {
          host: HostView;
          preview: Awaited<ReturnType<typeof previewGenerationPlacement>> | null;
          error: unknown;
          legacyUnsupported: boolean;
          roundTripMs: number;
        };
        const probes: PlacementProbe[] = [];
        const controllers = candidates.map(() => new AbortController());
        let pendingProbes = candidates.length;
        let resolveAllProbes!: () => void;
        let resolveFirstPlanned!: () => void;
        const allProbesSettled = new Promise<void>((resolve) => (resolveAllProbes = resolve));
        const firstPlanned = new Promise<void>((resolve) => (resolveFirstPlanned = resolve));
        candidates.forEach((host, index) => {
          void (async () => {
            const started = performance.now();
            try {
              const target = { baseUrl: host.baseUrl!, apiKey: host.apiKey };
              const preview =
                Array.isArray((request as ChainRequest).stages) || "total_frames" in request
                  ? await previewChainPlacement(
                      target,
                      previewRequestForSiblingFanout(
                        request as unknown as Record<string, unknown>,
                        copies,
                      ),
                      copies,
                      { signal: controllers[index]!.signal },
                    )
                  : await previewGenerationPlacement(
                      target,
                      previewRequestForSiblingFanout(
                        request as unknown as Record<string, unknown>,
                        copies,
                      ),
                      copies,
                      { signal: controllers[index]!.signal },
                    );
              probes.push({
                host,
                preview,
                error: null,
                legacyUnsupported: false,
                roundTripMs: Math.max(0, performance.now() - started),
              });
              if (classifyPlacementPreview(preview) === "planned") resolveFirstPlanned();
            } catch (error) {
              probes.push({
                host,
                preview: null,
                error,
                legacyUnsupported:
                  error instanceof ApiError && (error.status === 404 || error.status === 405),
                roundTripMs: Math.max(0, performance.now() - started),
              });
            } finally {
              pendingProbes -= 1;
              if (pendingProbes === 0) resolveAllProbes();
            }
          })();
        });
        const responsiveAuto = selection === null;
        let autoDeadlineReached = false;
        if (responsiveAuto) {
          let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
          const deadline = new Promise<void>((resolve) => {
            deadlineTimer = setTimeout(() => {
              autoDeadlineReached = true;
              resolve();
            }, AUTO_PLACEMENT_DEADLINE_MS);
          });
          const firstRouteWindow = firstPlanned.then(
            () => new Promise<void>((resolve) => setTimeout(resolve, AUTO_PLACEMENT_SETTLE_MS)),
          );
          await Promise.race([
            allProbesSettled,
            deadline,
            ...(candidates.length > 1 ? [firstRouteWindow] : []),
          ]);
          clearTimeout(deadlineTimer);
          if (pendingProbes > 0) controllers.forEach((controller) => controller.abort());
        } else {
          await allProbesSettled;
        }
        // Aborted fetches settle on a later microtask. Route using the stable
        // snapshot that met Auto's response window, not late cancellation rows.
        const settledProbes = probes.slice();
        if (autoDeadlineReached) {
          const settledHostIds = new Set(settledProbes.map((probe) => probe.host.id));
          for (const host of candidates) {
            if (settledHostIds.has(host.id)) continue;
            settledProbes.push({
              host,
              preview: null,
              error: new Error(
                "Auto placement timed out after 5 seconds; retry or select this machine explicitly for a longer cold check",
              ),
              legacyUnsupported: false,
              roundTripMs: AUTO_PLACEMENT_DEADLINE_MS,
            });
          }
        }
        if (intentSignature() !== capturedIntent || identitySignature() !== capturedIdentity) {
          return {
            kind: "transient",
            perHost: candidates.map((host) => ({
              kind: "transient" as const,
              hostId: host.id,
              label: host.label,
              error: "routing identity changed while placement was being checked",
            })),
          };
        }
        if (availabilitySignature() !== capturedAvailability) {
          if (attempt === 0) continue;
          return {
            kind: "transient",
            perHost: candidates.map((host) => ({
              kind: "transient" as const,
              hostId: host.id,
              label: host.label,
              error: "routing state changed while placement was being checked",
            })),
          };
        }

        const planned = settledProbes
          .flatMap((probe) =>
            probe.preview && classifyPlacementPreview(probe.preview) === "planned"
              ? [
                  {
                    host: probe.host,
                    preview: probe.preview,
                    roundTripMs: probe.roundTripMs,
                  },
                ]
              : [],
          )
          .map((probe) => ({
            hostId: probe.host.id,
            roundTripMs: probe.roundTripMs,
            preview: probe.preview,
          }))
          .sort(comparePlacementPreviews);
        if (planned.length > 0) {
          const chosen = candidates.find((host) => host.id === planned[0]!.hostId);
          const route = chosen ? hostRoute(chosen, this.capabilities[chosen.id]) : null;
          if (route) return { kind: "route", route };
        }

        const unsupportedIds = settledProbes
          .filter(
            (probe) =>
              probe.legacyUnsupported || classifyPlacementPreview(probe.preview) === "unsupported",
          )
          .map((probe) => probe.host.id);
        const legacy = candidates
          .filter((host) => unsupportedIds.includes(host.id))
          .map((host) => ({
            ...host,
            gpu: strongestRoutableGpu(this.telemetry[host.id]),
          }));
        if (!requireAuthoritative && legacy.length > 0) {
          const modelHostIds = useHostModelsStore()
            .hostsFor(request.model)
            .filter((id) => unsupportedIds.includes(id));
          let chosen: (typeof legacy)[number] | null;
          if (selection === "capable") {
            chosen = pickMostCapableHost(legacy, modelHostIds.length > 0 ? modelHostIds : null);
          } else if (selection !== null && this.all.some((host) => host.id === selection)) {
            chosen = legacy.find((host) => host.id === selection) ?? null;
          } else {
            const withModel = legacy.filter((host) => modelHostIds.includes(host.id));
            chosen = pickAutoHost(withModel.length > 0 ? withModel : legacy);
          }
          const route = chosen ? hostRoute(chosen, this.capabilities[chosen.id]) : null;
          if (route) return { kind: "route", route };
        }

        const failures = settledProbes.flatMap<HostFeasibilityFailure>((probe) => {
          const classification = classifyPlacementPreview(probe.preview);
          if (
            requireAuthoritative &&
            (probe.legacyUnsupported || classification === "unsupported")
          ) {
            return [
              {
                kind: "unreachable",
                hostId: probe.host.id,
                label: probe.host.label,
                error:
                  "does not provide the authoritative placement preview required for reference media",
              },
            ];
          }
          if (probe.error && !probe.legacyUnsupported) {
            return [
              {
                kind: "unreachable",
                hostId: probe.host.id,
                label: probe.host.label,
                error: probeError(probe.error),
              },
            ];
          }
          if (classification === "infeasible" && probe.preview) {
            const route = hostRoute(probe.host, this.capabilities[probe.host.id]);
            if (!route) return [];
            return [
              {
                kind: "infeasible",
                hostId: probe.host.id,
                label: probe.host.label,
                route,
                reason:
                  typeof probe.preview.reason === "string" && probe.preview.reason.trim()
                    ? probe.preview.reason.trim()
                    : "the server reported that this request is infeasible",
                missingComponents: probe.preview.missing_components ?? [],
              },
            ];
          }
          if (classification === "temporarily_unavailable") {
            return [
              {
                kind: "transient",
                hostId: probe.host.id,
                label: probe.host.label,
                error: probe.preview?.reason ?? "could not compute a placement plan right now",
              },
            ];
          }
          if (classification === "invalid") {
            return [
              {
                kind: "unreachable",
                hostId: probe.host.id,
                label: probe.host.label,
                error: "returned an invalid authoritative placement-preview response",
              },
            ];
          }
          return [];
        });
        if (failures.length === 0) {
          return {
            kind: "transient",
            perHost: candidates.map((host) => ({
              kind: "transient" as const,
              hostId: host.id,
              label: host.label,
              error: "no placement route could be selected",
            })),
          };
        }
        if (failures.every((failure) => failure.kind === "infeasible")) {
          return {
            kind: "infeasible",
            perHost: failures as HostPlacementFailure[],
          };
        }
        if (failures.every((failure) => failure.kind === "unreachable")) {
          return {
            kind: "unreachable",
            perHost: failures as HostProbeFailure[],
          };
        }
        if (failures.every((failure) => failure.kind === "transient")) {
          return {
            kind: "transient",
            perHost: failures as HostProbeFailure[],
          };
        }
        return { kind: "mixed", perHost: failures };
      }
      return { kind: "transient", perHost: [] };
    },
    /** Compatibility wrapper for callers that only need the chosen route. */
    async resolveFeasibleRoute(
      selection: string | null,
      request: GenerateRequest | ChainRequest | AutoChainRequest,
      copies = 1,
    ): Promise<HostRoute | null> {
      const result = await this.resolveFeasible(selection, request, copies);
      return result.kind === "route" ? result.route : null;
    },
    /** Pull queue depth/capacity from every live host. */
    async refresh() {
      /** hostId → instanceId learned this poll, to reconcile saved entries once. */
      const learned = new Map<string, string>();
      await Promise.all(
        this.all.map(async (host) => {
          if (!host.baseUrl || host.status === "connecting") return;
          const generation = (this.refreshGenerations[host.id] ?? 0) + 1;
          this.refreshGenerations[host.id] = generation;
          const target = { baseUrl: host.baseUrl, apiKey: host.apiKey };
          const isCurrent = () => {
            const current = this.all.find((candidate) => candidate.id === host.id);
            return (
              this.refreshGenerations[host.id] === generation &&
              current?.baseUrl === host.baseUrl &&
              current.apiKey === host.apiKey
            );
          };
          try {
            const [status, devices, queue] = await Promise.all([
              apiJsonTo<ServerStatus>(target, "/api/status"),
              listDevices(target).then(
                (snapshot) => snapshot.devices,
                () => null,
              ),
              listQueue(target).then(
                (listing) => listing,
                () => null,
              ),
            ]);
            if (!isCurrent()) return;
            this.telemetry[host.id] = {
              queueDepth: status.queue_depth ?? null,
              queueCapacity: status.queue_capacity ?? null,
              predictedCompletionMs:
                queue?.plan == null ? null : predictedCompletionUnixMs(queue.plan),
              version: status.version ?? null,
              modelsLoaded: status.models_loaded ?? [],
              gpuInfo: status.gpu_info ?? null,
              gpuWorkers: status.gpus ?? null,
              devices,
              instanceId: status.instance_id ?? null,
              hostname: status.hostname ?? null,
            };
            if (status.instance_id) learned.set(host.id, status.instance_id);
            if (status.hostname) this.hostnames[host.id] = status.hostname;
            const extra = this.extras.find((h) => h.id === host.id);
            if (extra && extra.status !== "ready") {
              extra.status = "ready";
              extra.error = null;
            }
            try {
              const capabilities = await fetchServerCapabilities(target);
              // Disconnect/dedupe may remove or re-home a host while the
              // request is in flight. Never resurrect its cache entry with a
              // late response from the old identity or address.
              const current = this.all.find((candidate) => candidate.id === host.id);
              if (isCurrent() && current?.status === "ready" && current.baseUrl === host.baseUrl) {
                this.capabilities[host.id] = capabilities;
              } else {
                delete this.capabilities[host.id];
              }
            } catch {
              // Capability discovery is advisory. A failed/unsupported probe
              // must not mark a healthy host unavailable or imply expand=false.
              delete this.capabilities[host.id];
            }
          } catch (err) {
            if (!isCurrent()) return;
            const extra = this.extras.find((h) => h.id === host.id);
            if (extra) {
              extra.status = "error";
              extra.error = String(err);
            }
            if (host.id === "local") {
              const conn = useConnectionStore();
              conn.localStatus = "error";
              conn.localError = String(err);
              // The native lifecycle distinguishes a dead embedded engine (or
              // vanished external server) and brings this Mac back online.
              void conn.ensureLocal(true);
            }
            delete this.telemetry[host.id];
            delete this.capabilities[host.id];
          }
        }),
      );
      // Stamp the extras with what the poll learned (the "shadowed by primary"
      // row starts unprobed) so `all`'s instance-id dedupe engages immediately.
      for (const extra of this.extras) {
        const uuid = learned.get(extra.id);
        if (uuid && extra.instanceId !== uuid) extra.instanceId = uuid;
      }
      if (learned.size > 0) await this.reconcileSavedInstanceIds(learned);
    },
    /**
     * Persist newly-learned instance ids onto their saved entries and collapse
     * saved hosts that turn out to be the same physical server (same instance
     * id AND a compatible hostname). Writes settings only when something
     * actually changed, so the 10 s poll stays quiet in steady state. Re-homes
     * the loser's connected-id, sticky target, per-host secret, user name, and
     * live extra onto the surviving slug — in memory as well as on disk.
     */
    async reconcileSavedInstanceIds(learned: Map<string, string>) {
      const hostnameOf = (id: string): string | null =>
        this.hostnames[id] ?? this.telemetry[id]?.hostname ?? null;
      const stamp = (saved: SavedHost[]) =>
        saved.map((h) => {
          const uuid = learned.get(h.id);
          const next = uuid && h.instanceId !== uuid ? { ...h, instanceId: uuid } : h;
          const hostname = hostnameOf(h.id);
          return (hostname ? { ...next, hostname } : next) as SavedHost & {
            hostname?: string | null;
          };
        });
      // Carry per-host secrets onto survivors BEFORE the settings
      // read-modify-write, so the get→set window below contains no slow
      // secret IPC — writers outside this store's lock (theme toggles, route
      // memory) must not be clobbered by a stale settings snapshot.
      const probe = mergeSavedHostsByInstanceId(stamp((await ipc.appSettingsGet()).savedHosts));
      for (const { loser, survivor } of probe.dropped) {
        const survivorKey = await ipc.secretGet(`remote-api-key.${survivor}`);
        if (!survivorKey) {
          const loserKey = await ipc.secretGet(`remote-api-key.${loser}`);
          if (loserKey) await ipc.secretSet(`remote-api-key.${survivor}`, loserKey);
        }
      }
      await withSettingsLock(async () => {
        const settings = await ipc.appSettingsGet();
        const stamped = stamp(settings.savedHosts);
        const { hosts: merged, dropped } = mergeSavedHostsByInstanceId(stamped);
        const changed =
          dropped.length > 0 ||
          stamped.some((h, i) => h.instanceId !== settings.savedHosts[i]?.instanceId);
        if (!changed) return;

        let connectedHostIds = settings.connectedHostIds;
        let generateTargetHost = settings.generateTargetHost;
        for (const { loser, survivor } of dropped) {
          connectedHostIds = connectedHostIds.map((id) => (id === loser ? survivor : id));
          if (generateTargetHost === loser) generateTargetHost = survivor;
          // The loser may hold the only working live connection (the
          // survivor's own address might not answer right now) — re-home it
          // onto the surviving slug instead of deleting it.
          const loserLive = this.extras.find((h) => h.id === loser);
          if (loserLive && !this.extras.some((h) => h.id === survivor)) {
            this.extras = this.extras.map((h) => (h.id === loser ? { ...h, id: survivor } : h));
            if (this.telemetry[loser]) this.telemetry[survivor] = this.telemetry[loser];
            if (this.capabilities[loser]) this.capabilities[survivor] = this.capabilities[loser];
            if (this.hostnames[loser] && !this.hostnames[survivor])
              this.hostnames[survivor] = this.hostnames[loser];
          } else {
            this.extras = this.extras.filter((h) => h.id !== loser);
          }
          delete this.telemetry[loser];
          delete this.capabilities[loser];
          delete this.hostnames[loser];
          // Carry the loser's user-assigned name in memory, mirroring the
          // saved-entry merge — labels must not wait for a relaunch.
          if (this.names[loser] && !this.names[survivor]) this.names[survivor] = this.names[loser];
          delete this.names[loser];
        }
        // `hostname` is a merge input, never a persisted field.
        const savedHosts: SavedHost[] = merged.map((h) => {
          const { hostname: _hostname, ...rest } = h;
          return rest;
        });
        connectedHostIds = [...new Set(connectedHostIds)];
        await ipc.appSettingsSet({ ...settings, savedHosts, connectedHostIds, generateTargetHost });
        // Keep the in-memory prefs snapshot coherent the same way disconnect()
        // does — every generateTargetHost reader consumes it.
        const prefs = useAppPrefsStore();
        if (prefs.settings)
          prefs.settings = { ...prefs.settings, savedHosts, connectedHostIds, generateTargetHost };
      });
    },
    startPolling() {
      if (this.pollTimer) return;
      void this.refresh();
      this.pollTimer = setInterval(() => void this.refresh(), POLL_INTERVAL_MS);
    },
    stopPolling() {
      if (this.pollTimer) clearInterval(this.pollTimer);
      this.pollTimer = null;
    },
    /** Remember the host across launches (MRU list + reconnect set). */
    async persist(id: string, url: string, name: string | null, instanceId: string | null = null) {
      await withSettingsLock(async () => {
        const settings = await ipc.appSettingsGet();
        // A nameless reconnect must not wipe a previously discovered name or a
        // previously learned instance id — same rule as Rust's
        // upsert_saved_host.
        const prior = settings.savedHosts.find((h) => h.id === id);
        const upserted: SavedHost[] = [
          {
            id,
            name: name ?? prior?.name ?? null,
            url,
            lastUsedMs: Date.now(),
            instanceId: instanceId ?? prior?.instanceId ?? null,
          },
          ...settings.savedHosts.filter((h) => h.id !== id),
        ].slice(0, MAX_SAVED_HOSTS);
        // Collapse any saved twin that shares this host's instance id (and a
        // compatible last-known hostname).
        const withHostnames = upserted.map((h) => {
          const hostname = this.hostnames[h.id] ?? this.telemetry[h.id]?.hostname ?? null;
          return hostname ? { ...h, hostname } : h;
        });
        const { hosts: mergedHosts, dropped } = mergeSavedHostsByInstanceId(withHostnames);
        const savedHosts: SavedHost[] = mergedHosts.map((h) => {
          const { hostname: _hostname, ...rest } = h as SavedHost & { hostname?: string | null };
          return rest;
        });
        const droppedIds = new Set(dropped.map((d) => d.loser));
        const connectedHostIds = [
          ...new Set(
            [...settings.connectedHostIds, id]
              .filter((h) => !droppedIds.has(h))
              .map((h) => dropped.find((d) => d.loser === h)?.survivor ?? h),
          ),
        ];
        await ipc.appSettingsSet({ ...settings, savedHosts, connectedHostIds });
      });
    },
  },
});
