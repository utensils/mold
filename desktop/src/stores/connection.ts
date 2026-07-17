import { defineStore } from "pinia";
import { ipc, type ConnectionInfo, type LocalServerInfo } from "../lib/ipc";

export type ConnectionStatus = "idle" | "starting" | "ready" | "error";

/**
 * Which engine the app is talking to. The built-in engine is always the
 * internal primary; `init()` runs once at shell mount and brings the local
 * engine online (embedded, or an already-running `mold serve` on :7680).
 * Remote hosts are additive list entries owned by the hosts store, not a
 * primary the connection ever switches to.
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
        await this.ensureLocal();
        if (this.localStatus !== "ready") {
          throw new Error(this.localError ?? "The local server didn't start.");
        }
        this.info = await ipc.startLocalEngine();
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
    async stopEngine() {
      this.info = await ipc.stopLocalEngine();
      this.localInfo = null;
      this.localStatus = "idle";
      this.localError = null;
      this.status = "idle";
    },
  },
});
