import { defineStore } from "pinia";
import { apiJsonTo, type ApiTarget } from "../lib/api/client";
import { hostIdFromUrl, normalizeHostUrl, pickAutoHost } from "../lib/hosts";
import { ipc, type SavedHost } from "../lib/ipc";
import type { ServerStatus } from "../lib/api/types";
import { useConnectionStore } from "./connection";
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
    pollTimer: null as ReturnType<typeof setInterval> | null,
    initialized: false,
  }),
  getters: {
    /** The primary connection as a host row ("This Mac" or the remote). */
    primaryHost(state): HostView | null {
      const conn = useConnectionStore();
      if (!conn.info?.baseUrl) return null;
      const remote = conn.info.mode === "remote";
      const id = remote ? hostIdFromUrl(conn.info.baseUrl) : "local";
      const t = state.telemetry[id];
      return {
        id,
        label: remote ? conn.info.baseUrl.replace(/^https?:\/\//, "") : "This Mac",
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
    /** Every host, primary first; an extra shadowed by the primary is hidden. */
    all(state): HostView[] {
      const primary = this.primaryHost;
      const rows: HostView[] = primary ? [primary] : [];
      for (const extra of state.extras) {
        if (extra.id === primary?.id) continue;
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
      if (this.initialized) return;
      this.initialized = true;
      try {
        const settings = await ipc.appSettingsGet();
        for (const id of settings.connectedHostIds) {
          const saved = settings.savedHosts.find((h) => h.id === id);
          if (!saved) continue;
          const key = await ipc.secretGet(`remote-api-key.${id}`);
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
      await ipc.appSettingsSet({
        ...settings,
        connectedHostIds: settings.connectedHostIds.filter((h) => h !== id),
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
    },
    /**
     * Resolve where a batch should run. `null`/unknown selection = Auto
     * (least busy). An explicit pick that is not ready resolves to null so
     * the caller can say so instead of silently rerouting.
     */
    resolveRoute(selection: string | null): HostRoute | null {
      const routable = this.all.map((h) => ({
        ...h,
        kind: h.kind,
        queueDepth: h.queueDepth,
      }));
      const chosen = selection
        ? (routable.find((h) => h.id === selection && h.status === "ready") ?? null)
        : pickAutoHost(routable);
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
      const savedHosts: SavedHost[] = [
        { id, name, url, lastUsedMs: Date.now() },
        ...settings.savedHosts.filter((h) => h.id !== id),
      ].slice(0, MAX_SAVED_HOSTS);
      const connectedHostIds = settings.connectedHostIds.includes(id)
        ? settings.connectedHostIds
        : [...settings.connectedHostIds, id];
      await ipc.appSettingsSet({ ...settings, savedHosts, connectedHostIds });
    },
  },
});
