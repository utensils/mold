import { defineStore } from "pinia";
import { apiJson } from "../lib/api/client";
import { gpuSnapshotsFromWorkers } from "../lib/api/gpuStatus";
import { sseStream } from "../lib/api/sse";
import type { GpuSnapshot, ResourceSnapshot, ServerStatus } from "../lib/api/types";
import { shouldRestartEmbeddedEngine } from "../lib/connectionRecovery";
import { percent, vramLevel } from "../lib/format";
import { normalizeTargetHost, pickDisplayHost } from "../lib/hosts";
import {
  hostMemoryLevel,
  type HostMemoryLevel,
  type HostMemorySnapshot,
} from "@studio/lib/hostMemory";
import { useAppPrefsStore } from "./appPrefs";
import { useConnectionStore } from "./connection";
import { useGenerationStore } from "./generation";
import { useHostsStore, type HostView } from "./hosts";
import { useJobsStore } from "./jobs";
import { useToastStore } from "./toasts";

export type DisplayConnection = "idle" | "connecting" | "ready" | "error";

let resourceAbort: AbortController | null = null;
let statusTimer: ReturnType<typeof setInterval> | null = null;
let statusFailures = 0;

/**
 * The machine the shell talks about — in the sidebar's machine card and the
 * status bar (README §04: "which machine, how full, how deep is the queue"
 * never costs a view change).
 *
 * The display host mirrors the concrete host selected in the Create header.
 * Auto and Most capable have no concrete host until routing happens, so
 * those modes follow the most recently submitted live job with a routed host
 * id and otherwise show the primary. The embedded-engine recovery poll stays
 * bound to the PRIMARY connection no matter what is displayed.
 */
export const useHostStatusStore = defineStore("hostStatus", {
  state: () => ({
    /** Live VRAM/RAM for the display host; null while its stream is silent. */
    snapshot: null as ResourceSnapshot | null,
    /** The primary's /api/status, polled every 10 s for engine recovery. */
    status: null as ServerStatus | null,
  }),
  getters: {
    displayHost(): HostView | null {
      const appPrefs = useAppPrefsStore();
      const generation = useGenerationStore();
      const hosts = useHostsStore();
      const selected = normalizeTargetHost(appPrefs.settings?.generateTargetHost, hosts.all);
      if (selected && selected !== "capable") {
        return hosts.all.find((host) => host.id === selected) ?? hosts.primaryHost;
      }
      const liveIds = generation.jobs
        .filter((j) => j.status !== "complete" && j.status !== "error")
        .map((j) => j.hostId);
      const primaryId = hosts.primaryHost?.id ?? "local";
      const id = pickDisplayHost(liveIds, primaryId);
      return hosts.all.find((h) => h.id === id) ?? hosts.primaryHost;
    },
    displayingRemote(): boolean {
      return !!this.displayHost && !this.displayHost.primary;
    },
    connection(): DisplayConnection {
      const conn = useConnectionStore();
      if (this.displayingRemote) return this.displayHost!.status;
      return conn.status === "starting" ? "connecting" : conn.status;
    },
    anyJobRunning(): boolean {
      const generation = useGenerationStore();
      return generation.jobs.some((j) => j.status !== "complete" && j.status !== "error");
    },
    /** Every GPU on the display host: the live snapshot, else status-shaped
     * telemetry for a host whose resources stream is silent. */
    gpus(): GpuSnapshot[] {
      if (this.snapshot?.gpus.length) return this.snapshot.gpus;
      const host = this.displayHost;
      if (!host) return [];
      if (this.displayingRemote) {
        const telemetry = useHostsStore().telemetry[host.id];
        return gpuSnapshotsFromWorkers(telemetry?.gpuInfo, telemetry?.gpuWorkers);
      }
      return gpuSnapshotsFromWorkers(this.status?.gpu_info, this.status?.gpus);
    },
    vramUsed(): number {
      return this.gpus.reduce((sum, g) => sum + g.vram_used, 0);
    },
    vramTotal(): number {
      return this.gpus.reduce((sum, g) => sum + g.vram_total, 0);
    },
    vramPct(): number {
      return this.gpus.length ? percent(this.vramUsed, this.vramTotal) : 0;
    },
    vramCritical(): boolean {
      return this.gpus.some((g) => vramLevel(g.vram_used, g.vram_total) === "critical");
    },
    /**
     * Host-RAM pressure from the scheduler's own ledger, not used/total: RAM
     * committed to a reservation that has not allocated yet looks free to the
     * OS, and that gap is what parks a queue. `/api/status` is the source of
     * record for the primary; a remote display host falls back to its queue
     * plan's mirror.
     */
    hostMemory(): HostMemorySnapshot | null {
      const jobs = useJobsStore();
      const host: HostView | null = this.displayHost;
      const plan = host ? (jobs.queues[host.id]?.plan ?? null) : null;
      const remote: boolean = this.displayingRemote;
      return (remote ? null : (this.status?.host_memory ?? null)) ?? plan?.host_memory ?? null;
    },
    hostMemoryPressure(): HostMemoryLevel {
      return hostMemoryLevel(this.hostMemory) ?? "ok";
    },
    /** One plain-English line for the machine card and the status bar. */
    sentence(): string {
      const remote = this.displayingRemote;
      switch (this.connection) {
        case "error":
          return remote ? "Machine is offline." : "The engine hit an error.";
        case "connecting":
          return remote ? "Connecting…" : "Starting the engine…";
        case "idle":
          return "The engine is off.";
      }
      if (this.anyJobRunning) return "Making images now.";
      return this.gpus.length ? "Ready and waiting." : "Ready — no GPU telemetry yet.";
    },
  },
  actions: {
    /** PRIMARY-only status poll — drives embedded-engine recovery and must
     *  never be re-targeted at the display host. */
    async refreshStatus() {
      const conn = useConnectionStore();
      if (!conn.ready) return;
      try {
        this.status = await apiJson<ServerStatus>("/api/status");
        statusFailures = 0;
      } catch {
        statusFailures += 1;
        if (shouldRestartEmbeddedEngine(conn.mode, statusFailures)) {
          statusFailures = 0;
          this.status = null;
          await conn.useLocal();
          if (conn.ready) {
            useToastStore().push("Engine restarted");
            this.start();
          }
        }
      }
    },
    /** Live VRAM/RAM for the DISPLAY host; reopened whenever it changes. */
    startResourceStream() {
      resourceAbort?.abort();
      resourceAbort = new AbortController();
      this.snapshot = null;
      const host = this.displayHost;
      if (this.displayingRemote && host?.status !== "ready") return;
      void sseStream("/api/resources/stream", {
        signal: resourceAbort.signal,
        ...(this.displayingRemote && host?.baseUrl
          ? { target: { baseUrl: host.baseUrl, apiKey: host.apiKey } }
          : {}),
        onEvent: (event, data) => {
          if (event !== "snapshot") return;
          try {
            this.snapshot = JSON.parse(data) as ResourceSnapshot;
          } catch {
            /* skip malformed frame */
          }
        },
      });
    },
    start() {
      this.stop();
      this.startResourceStream();
      void this.refreshStatus();
      statusTimer = setInterval(() => void this.refreshStatus(), 10_000);
    },
    stop() {
      resourceAbort?.abort();
      resourceAbort = null;
      if (statusTimer) clearInterval(statusTimer);
      statusTimer = null;
      this.snapshot = null;
      this.status = null;
    },
  },
});
