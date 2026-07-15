import { defineStore } from "pinia";
import { hostIdFromUrl } from "../lib/hosts";
import { ipc, type ConnectionInfo, type LocalServerInfo } from "../lib/ipc";
import { useToastStore } from "./toasts";

export type ConnectionStatus = "idle" | "starting" | "ready" | "error";

/** Bare host for user-facing messages ("studio.local", not the full URL). */
function hostLabel(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/**
 * Which engine the app is talking to. `init()` runs once at shell mount:
 * saved remote hosts reconnect, everything else brings the local engine
 * online (embedded, or an already-running `mold serve` on :7680). An
 * unreachable saved host falls back to the local engine for this session
 * without rewriting the remote preference.
 */
export const useConnectionStore = defineStore("connection", {
  state: () => ({
    info: null as ConnectionInfo | null,
    status: "idle" as ConnectionStatus,
    error: null as string | null,
    localInfo: null as LocalServerInfo | null,
    localStatus: "idle" as ConnectionStatus,
    localError: null as string | null,
  }),
  getters: {
    baseUrl: (s) => s.info?.baseUrl ?? null,
    apiKey: (s) => s.info?.apiKey ?? null,
    mode: (s) => s.info?.mode ?? "off",
    ready: (s) => s.status === "ready" && !!s.info?.baseUrl,
  },
  actions: {
    async ensureLocal(force = false) {
      if (this.localStatus === "starting" || (this.localStatus === "ready" && !force)) return;
      this.localStatus = "starting";
      this.localError = null;
      try {
        this.localInfo = await ipc.ensureLocalServer();
        this.localStatus = "ready";
      } catch (err) {
        this.localInfo = null;
        this.localStatus = "error";
        this.localError = String(err);
      }
    },
    async init() {
      if (this.status === "starting" || this.status === "ready") return;
      this.status = "starting";
      this.error = null;
      try {
        const settings = await ipc.appSettingsGet();
        await this.ensureLocal();
        // Remote key: secret store first, legacy settings.json field as
        // fallback for files written before secrets landed.
        const remoteKey =
          settings.mode === "remote"
            ? ((await ipc.secretGet("remote-api-key")) ?? settings.remoteApiKey)
            : null;
        if (settings.mode === "remote" && settings.remoteUrl) {
          try {
            this.info = await ipc.setRemoteHost(settings.remoteUrl, remoteKey);
          } catch {
            // Session-only fallback: settings.mode stays "remote" so the
            // preferred host is retried on the next launch.
            useToastStore().push(
              `Couldn't reach ${hostLabel(settings.remoteUrl)} — using the built-in engine.`,
              "error",
            );
            // The unreachable host stays visible as an errored sidebar row;
            // the hosts poll self-heals it when the server answers again.
            // (Dynamic import: the hosts store statically imports this one.)
            const id = hostIdFromUrl(settings.remoteUrl);
            const saved = settings.savedHosts.find((h) => h.id === id);
            const { useHostsStore } = await import("./hosts");
            useHostsStore().adopt(id, settings.remoteUrl, remoteKey, saved?.name ?? null);
            if (this.localStatus !== "ready") {
              throw new Error(this.localError ?? "The local server didn't start.");
            }
            this.info = await ipc.startLocalEngine();
          }
        } else {
          if (this.localStatus !== "ready") {
            throw new Error(this.localError ?? "The local server didn't start.");
          }
          this.info = await ipc.startLocalEngine();
        }
        this.status = "ready";
      } catch (err) {
        this.status = "error";
        this.error = String(err);
      }
    },
    async useLocal() {
      this.status = "starting";
      this.error = null;
      try {
        await this.ensureLocal();
        if (this.localStatus !== "ready") {
          throw new Error(this.localError ?? "The local server didn't start.");
        }
        this.info = await ipc.startLocalEngine();
        this.status = "ready";
        const settings = await ipc.appSettingsGet();
        await ipc.appSettingsSet({ ...settings, mode: "local" });
      } catch (err) {
        this.status = "error";
        this.error = String(err);
      }
    },
    async useRemote(url: string, apiKey: string | null, name?: string | null) {
      this.status = "starting";
      this.error = null;
      try {
        this.info = await ipc.setRemoteHost(url, apiKey, name);
        this.status = "ready";
      } catch (err) {
        this.status = "error";
        this.error = String(err);
        throw err;
      }
    },
    async stopEngine() {
      this.info = await ipc.stopLocalEngine();
      this.localInfo = null;
      this.localStatus = "idle";
      this.localError = null;
      this.status = "idle";
    },
  },
});
