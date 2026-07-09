import { defineStore } from "pinia";
import { ipc, type ConnectionInfo } from "../lib/ipc";

export type ConnectionStatus = "idle" | "starting" | "ready" | "error";

/**
 * Which engine the app is talking to. `init()` runs once at shell mount:
 * saved remote hosts reconnect, everything else brings the local engine
 * online (embedded, or an already-running `mold serve` on :7680).
 */
export const useConnectionStore = defineStore("connection", {
  state: () => ({
    info: null as ConnectionInfo | null,
    status: "idle" as ConnectionStatus,
    error: null as string | null,
  }),
  getters: {
    baseUrl: (s) => s.info?.baseUrl ?? null,
    apiKey: (s) => s.info?.apiKey ?? null,
    mode: (s) => s.info?.mode ?? "off",
    ready: (s) => s.status === "ready" && !!s.info?.baseUrl,
  },
  actions: {
    async init() {
      if (this.status === "starting" || this.status === "ready") return;
      this.status = "starting";
      this.error = null;
      try {
        const settings = await ipc.appSettingsGet();
        this.info =
          settings.mode === "remote" && settings.remoteUrl
            ? await ipc.setRemoteHost(settings.remoteUrl, settings.remoteApiKey)
            : await ipc.startLocalEngine();
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
        this.info = await ipc.startLocalEngine();
        this.status = "ready";
        const settings = await ipc.appSettingsGet();
        await ipc.appSettingsSet({ ...settings, mode: "local" });
      } catch (err) {
        this.status = "error";
        this.error = String(err);
      }
    },
    async useRemote(url: string, apiKey: string | null) {
      this.status = "starting";
      this.error = null;
      try {
        this.info = await ipc.setRemoteHost(url, apiKey);
        this.status = "ready";
      } catch (err) {
        this.status = "error";
        this.error = String(err);
        throw err;
      }
    },
    async stopEngine() {
      this.info = await ipc.stopLocalEngine();
      this.status = "idle";
    },
  },
});
