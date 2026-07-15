import { defineStore } from "pinia";
import { apiJsonTo, type ApiTarget } from "../lib/api/client";
import { hostIdFromUrl, normalizeHostUrl, pickAutoHost, pickMostCapableHost } from "../lib/hosts";
import { ipc, type SavedHost } from "../lib/ipc";
import { PLATFORM_UI } from "../lib/platform";
import type { GpuInfo, ServerStatus } from "../lib/api/types";
import { useAppPrefsStore } from "./appPrefs";
import { useConnectionStore } from "./connection";
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
    /** The primary connection as a host row (this device or the remote). */
    primaryHost(state): HostView | null {
      const conn = useConnectionStore();
      if (!conn.info?.baseUrl) return null;
      const remote = conn.info.mode === "remote";
      const id = remote ? hostIdFromUrl(conn.info.baseUrl) : "local";
      const t = state.telemetry[id];
      return {
        id,
        label: remote
          ? (state.names[id] ?? conn.info.baseUrl.replace(/^https?:\/\//, ""))
          : PLATFORM_UI.deviceLabel,
        kind: remote ? "remote" : "local",
        baseUrl: conn.info.baseUrl,
        apiKey: conn.info.apiKey,
        status:
          conn.status === "ready" ? "ready" : conn.status === "starting" ? "connecting" : "error",
        primary: true,
        queueDepth: t?.queueDepth ?? null,
        queueCapacity: t?.queueCapacity ?? null,
        version: t?.version ?? null,
      };
    },
    /** Every host, primary first; an extra shadowed by the primary is hidden.
     *  Dedupe by URL as well as id — the local primary's id is the literal
     *  "local", so a loopback extra pointing at the same server would
     *  otherwise be listed (and routed to) twice. */
    all(state): HostView[] {
      const primary = this.primaryHost;
      const rows: HostView[] = primary ? [primary] : [];
      const conn = useConnectionStore();
      if (primary?.kind !== "local" && conn.localStatus !== "idle") {
        rows.push({
          id: "local",
          label: PLATFORM_UI.deviceLabel,
          kind: "local",
          baseUrl: conn.localInfo?.baseUrl ?? null,
          apiKey: conn.localInfo?.apiKey ?? null,
          status:
            conn.localStatus === "ready"
              ? "ready"
              : conn.localStatus === "starting"
                ? "connecting"
                : "error",
          primary: false,
          queueDepth: state.telemetry.local?.queueDepth ?? null,
          queueCapacity: state.telemetry.local?.queueCapacity ?? null,
          version: state.telemetry.local?.version ?? null,
        });
      }
      for (const extra of state.extras) {
        if (rows.some((row) => row.id === extra.id || (row.baseUrl && row.baseUrl === extra.url)))
          continue;
        const t = state.telemetry[extra.id];
        rows.push({
          id: extra.id,
          label: extra.label,
          kind: "remote",
          baseUrl: extra.url,
          apiKey: extra.apiKey,
          status: extra.status,
          primary: false,
          queueDepth: t?.queueDepth ?? null,
          queueCapacity: t?.queueCapacity ?? null,
          version: t?.version ?? null,
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
          // Shadowed by the live primary: same server, already connected.
          // List it (so switching the primary away mid-session keeps the
          // host around) but skip the redundant probe; `all` hides the row.
          if (id === this.primaryHost?.id) {
            this.extras.push({
              id,
              label: saved.name ?? saved.url.replace(/^https?:\/\//, ""),
              url: saved.url,
              apiKey: key,
              status: "ready",
              error: null,
            });
            continue;
          }
          const extra: ExtraHost = {
            id,
            label: saved.name ?? saved.url.replace(/^https?:\/\//, ""),
            url: saved.url,
            apiKey: key,
            status: "connecting",
            error: null,
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
      const existing = this.all.find((h) => h.id === id);
      if (existing?.status === "ready") return existing;

      const test = await ipc.testRemoteHost(url, apiKey);
      if (!test.ok) throw new Error(test.error ?? "Connection failed.");

      this.extras = this.extras.filter((h) => h.id !== id);
      this.extras.push({
        id,
        label: name ?? url.replace(/^https?:\/\//, ""),
        url,
        apiKey,
        status: "ready",
        error: null,
      });
      if (apiKey) await ipc.secretSet(`remote-api-key.${id}`, apiKey);
      await this.persist(id, url, name);
      void this.refresh();
      return this.all.find((h) => h.id === id)!;
    },
    /** Drop a live extra host. Its saved entry and key stay for later. */
    async disconnect(id: string) {
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
    /**
     * List a host that could not be reached (e.g. the persisted primary at
     * launch) as an errored extra, so it stays visible in the sidebar for
     * one-click reconnect instead of silently vanishing. The regular refresh
     * poll self-heals it the moment the host answers again.
     */
    adopt(id: string, url: string, apiKey: string | null, label?: string | null) {
      if (this.extras.some((h) => h.id === id)) return;
      this.extras.push({
        id,
        label: label ?? this.names[id] ?? url.replace(/^https?:\/\//, ""),
        url,
        apiKey,
        status: "error",
        error: "Unreachable at launch",
      });
    },
    /**
     * Keep a remote primary live as an extra while the app returns to the
     * built-in engine. Order matters: `connect()` deliberately no-ops while
     * the host is still the primary, so the engine switch happens first.
     */
    async demoteToExtra(host: {
      id: string;
      baseUrl: string | null;
      apiKey: string | null;
      label: string;
    }) {
      const conn = useConnectionStore();
      await conn.useLocal();
      if (!host.baseUrl) return;
      try {
        await this.connect(host.baseUrl, host.apiKey, host.label);
      } catch {
        // Unreachable right now — the saved entry still allows reconnect.
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
              { id, name: trimmed, url: extra.url, lastUsedMs: Date.now() },
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
            };
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
    async persist(id: string, url: string, name: string | null) {
      const settings = await ipc.appSettingsGet();
      // A nameless reconnect must not wipe a previously discovered name —
      // same rule as Rust's upsert_saved_host.
      const existingName = settings.savedHosts.find((h) => h.id === id)?.name ?? null;
      const savedHosts: SavedHost[] = [
        { id, name: name ?? existingName, url, lastUsedMs: Date.now() },
        ...settings.savedHosts.filter((h) => h.id !== id),
      ].slice(0, MAX_SAVED_HOSTS);
      const connectedHostIds = settings.connectedHostIds.includes(id)
        ? settings.connectedHostIds
        : [...settings.connectedHostIds, id];
      await ipc.appSettingsSet({ ...settings, savedHosts, connectedHostIds });
    },
  },
});
