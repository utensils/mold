import { defineStore } from "pinia";
import { apiJsonTo, type ApiTarget } from "../lib/api/client";
import {
  hostIdFromUrl,
  mergeSavedHostsByInstanceId,
  normalizeHostUrl,
  pickAutoHost,
  pickMostCapableHost,
} from "../lib/hosts";
import { ipc, type SavedHost } from "../lib/ipc";
import { PLATFORM_UI } from "../lib/platform";
import type { GpuInfo, ServerStatus } from "../lib/api/types";
import { useAppPrefsStore } from "./appPrefs";
import { useConnectionStore } from "./connection";
import { useDownloadsStore } from "./downloads";
import { useHostModelsStore } from "./hostModels";
import { useToastStore } from "./toasts";

const POLL_INTERVAL_MS = 10_000;
const MAX_SAVED_HOSTS = 8;

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
  version: string | null;
  /** Stable server-installation UUID; null until learned. Dedupe key. */
  instanceId: string | null;
}

interface HostTelemetry {
  queueDepth: number | null;
  queueCapacity: number | null;
  version: string | null;
  /** Loaded-model names from `/api/status` (absent before the first poll). */
  modelsLoaded?: string[];
  /** GPU summary from `/api/status`; the status bar's fallback when a host's
   *  resources stream is unavailable. */
  gpuInfo?: GpuInfo | null;
  /** Stable server-installation UUID from `/api/status`; absent on older servers. */
  instanceId?: string | null;
  /** Server-reported hostname from `/api/status`; drives the display label. */
  hostname?: string | null;
}

/** Where a batch will run: resolved just before submit. */
export interface HostRoute {
  hostId: string;
  label: string;
  kind: "local" | "remote";
  target: ApiTarget;
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
    /** Friendly names from savedHosts, so the primary can be renamed too. */
    names: {} as Record<string, string>,
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
      for (const extra of state.extras) {
        const t = state.telemetry[extra.id];
        const instanceId = t?.instanceId ?? extra.instanceId ?? null;
        // Dedupe by id, loopback URL, AND instance id — the same physical
        // server reached by hostname vs IP has one row, not two.
        if (
          rows.some(
            (row) =>
              row.id === extra.id ||
              (row.baseUrl && row.baseUrl === extra.url) ||
              (instanceId !== null && row.instanceId === instanceId),
          )
        )
          continue;
        rows.push({
          id: extra.id,
          // User rename wins; otherwise the server hostname; otherwise the URL.
          label: this.names[extra.id] ?? t?.hostname ?? extra.label,
          kind: "remote",
          baseUrl: extra.url,
          apiKey: extra.apiKey,
          status: extra.status,
          primary: false,
          queueDepth: t?.queueDepth ?? null,
          queueCapacity: t?.queueCapacity ?? null,
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
    /** Reconnect remembered extra hosts. Never blocks or fails the boot. */
    async init() {
      if (this.initialized || this.initializing) return;
      this.initializing = true;
      try {
        const settings = await ipc.appSettingsGet();
        for (const saved of settings.savedHosts) {
          if (saved.name) this.names[saved.id] = saved.name;
        }
        for (const id of settings.connectedHostIds) {
          const saved = settings.savedHosts.find((h) => h.id === id);
          if (!saved) continue;
          // Already listed (e.g. adopted after a failed primary reconnect) —
          // don't duplicate the row or re-probe it.
          if (this.extras.some((h) => h.id === id)) continue;
          const key = await ipc.secretGet(`remote-api-key.${id}`);
          const extra: ExtraHost = {
            id,
            label: saved.name ?? saved.url.replace(/^https?:\/\//, ""),
            url: saved.url,
            apiKey: key,
            status: "connecting",
            error: null,
            instanceId: saved.instanceId ?? null,
          };
          this.extras.push(extra);
          const live = this.extras.find((h) => h.id === id)!;
          const test = await ipc.testRemoteHost(saved.url, key);
          if (test.ok) {
            live.status = "ready";
          } else {
            live.status = "error";
            live.error = test.error;
            useToastStore().push(
              `Couldn't reach ${live.label} — it stays listed for reconnect.`,
              "error",
            );
          }
        }
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
      // mDNS name): return the existing row instead of a duplicate. Adopt the
      // caller's key onto that host's slug if it has none stored yet.
      if (instanceId !== null) {
        const twin = this.all.find((h) => h.instanceId === instanceId);
        if (twin) {
          if (apiKey && !(await ipc.secretGet(`remote-api-key.${twin.id}`))) {
            await ipc.secretSet(`remote-api-key.${twin.id}`, apiKey);
          }
          return twin;
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
      const settings = await ipc.appSettingsGet();
      // A sticky generation target pointing at the removed host would only
      // linger as a dead preference (resolveRoute already falls back to
      // Auto) — clear it alongside the reconnect entry.
      const clearedTarget = settings.generateTargetHost === id;
      await ipc.appSettingsSet({
        ...settings,
        connectedHostIds: settings.connectedHostIds.filter((h) => h !== id),
        generateTargetHost: clearedTarget ? null : settings.generateTargetHost,
      });
      if (clearedTarget) {
        // Keep the in-memory prefs snapshot coherent (HostSelector reads it).
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
    },
    /** Retry a failed extra host in place. */
    async reconnect(id: string) {
      const extra = this.extras.find((h) => h.id === id);
      if (!extra) return;
      extra.status = "connecting";
      const test = await ipc.testRemoteHost(extra.url, extra.apiKey);
      extra.status = test.ok ? "ready" : "error";
      extra.error = test.ok ? null : test.error;
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
      const routable = this.all.map((h) => {
        const gpu = this.telemetry[h.id]?.gpuInfo ?? null;
        return {
          ...h,
          gpu: gpu
            ? { backend: gpu.backend ?? null, name: gpu.name, vramTotalMb: gpu.vram_total_mb }
            : null,
        };
      });
      const modelHostIds = modelName ? useHostModelsStore().hostsFor(modelName) : [];

      let chosen: (typeof routable)[number] | null;
      if (selection === "capable") {
        chosen = pickMostCapableHost(routable, modelHostIds.length > 0 ? modelHostIds : null);
      } else if (selection !== null && routable.some((h) => h.id === selection)) {
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
      };
    },
    /** Pull queue depth/capacity from every live host. */
    async refresh() {
      /** hostId → instanceId learned this poll, to reconcile saved entries once. */
      const learned = new Map<string, string>();
      await Promise.all(
        this.all.map(async (host) => {
          if (!host.baseUrl || host.status === "connecting") return;
          try {
            const status = await apiJsonTo<ServerStatus>(
              { baseUrl: host.baseUrl, apiKey: host.apiKey },
              "/api/status",
            );
            this.telemetry[host.id] = {
              queueDepth: status.queue_depth ?? null,
              queueCapacity: status.queue_capacity ?? null,
              version: status.version ?? null,
              modelsLoaded: status.models_loaded ?? [],
              gpuInfo: status.gpu_info ?? null,
              instanceId: status.instance_id ?? null,
              hostname: status.hostname ?? null,
            };
            if (status.instance_id) learned.set(host.id, status.instance_id);
            const extra = this.extras.find((h) => h.id === host.id);
            if (extra && extra.status !== "ready") {
              extra.status = "ready";
              extra.error = null;
            }
          } catch (err) {
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
     * saved hosts that turn out to be the same physical server. Writes settings
     * only when something actually changed, so the 10 s poll stays quiet in
     * steady state. Re-homes the loser's connected-id, sticky target, per-host
     * secret, and live extra onto the surviving slug.
     */
    async reconcileSavedInstanceIds(learned: Map<string, string>) {
      const settings = await ipc.appSettingsGet();
      const stamped: SavedHost[] = settings.savedHosts.map((h) => {
        const uuid = learned.get(h.id);
        return uuid && h.instanceId !== uuid ? { ...h, instanceId: uuid } : h;
      });
      const { hosts: mergedHosts, dropped } = mergeSavedHostsByInstanceId(stamped);
      const changed =
        dropped.length > 0 ||
        stamped.some((h, i) => h.instanceId !== settings.savedHosts[i]?.instanceId);
      if (!changed) return;

      let connectedHostIds = settings.connectedHostIds;
      let generateTargetHost = settings.generateTargetHost;
      for (const { loser, survivor } of dropped) {
        // Carry the loser's key onto the survivor's slug if it lacks one.
        const survivorKey = await ipc.secretGet(`remote-api-key.${survivor}`);
        if (!survivorKey) {
          const loserKey = await ipc.secretGet(`remote-api-key.${loser}`);
          if (loserKey) await ipc.secretSet(`remote-api-key.${survivor}`, loserKey);
        }
        connectedHostIds = connectedHostIds.map((id) => (id === loser ? survivor : id));
        if (generateTargetHost === loser) generateTargetHost = survivor;
        // Drop the loser's live row; the survivor already covers the server.
        this.extras = this.extras.filter((h) => h.id !== loser);
        delete this.telemetry[loser];
      }
      await ipc.appSettingsSet({
        ...settings,
        savedHosts: mergedHosts,
        connectedHostIds: [...new Set(connectedHostIds)],
        generateTargetHost,
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
      const settings = await ipc.appSettingsGet();
      // A nameless reconnect must not wipe a previously discovered name or a
      // previously learned instance id — same rule as Rust's upsert_saved_host.
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
      // Collapse any saved twin that shares this host's instance id.
      const { hosts: savedHosts, dropped } = mergeSavedHostsByInstanceId(upserted);
      const droppedIds = new Set(dropped.map((d) => d.loser));
      const connectedHostIds = [
        ...new Set(
          [...settings.connectedHostIds, id]
            .filter((h) => !droppedIds.has(h))
            .map((h) => dropped.find((d) => d.loser === h)?.survivor ?? h),
        ),
      ];
      await ipc.appSettingsSet({ ...settings, savedHosts, connectedHostIds });
    },
  },
});
