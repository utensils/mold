/**
 * Pull-and-resume for missing-model generations. When a generate lands on a
 * host that doesn't have the model (HTTP 404), the Generate view offers to
 * download it there; this store owns the "resume once the pull finishes"
 * half so navigation away from Generate can't orphan it. It watches the
 * downloads store (already streaming per-host job events) for a terminal
 * job matching the armed model, then resubmits the exact original request
 * on the original route.
 */
import { defineStore } from "pinia";
import { watch } from "vue";
import { useDownloadsStore, type DownloadHostState } from "./downloads";
import { useGenerationStore } from "./generation";
import { useToastStore } from "./toasts";
import type { HostRoute } from "./hosts";
import type { DownloadJob, GenerateRequest } from "../lib/api/types";

export interface PendingResume {
  model: string;
  /** Downloads-store bucket to watch: a host id, or null for the primary. */
  hostId: string | null;
  hostLabel: string;
  request: GenerateRequest;
  batch: number;
  route: HostRoute | null;
}

const isTerminal = (job: DownloadJob) =>
  job.status === "completed" || job.status === "failed" || job.status === "cancelled";

export const usePullResumeStore = defineStore("pullResume", {
  state: () => ({
    pending: null as PendingResume | null,
    /** Terminal job ids that existed when arm() ran — never resume off those. */
    seenTerminal: [] as string[],
  }),
  actions: {
    /** Watch for the model's pull to finish, then regenerate. One at a time —
     *  arming again replaces the previous pending resume. */
    arm(pending: PendingResume) {
      this.pending = pending;
      this.seenTerminal = this.jobsFor(pending.hostId)
        .filter(isTerminal)
        .map((job) => job.id);
      this.ensureWatcher();
      this.check();
    },
    cancel() {
      this.pending = null;
    },
    /** All download jobs in the watched bucket (active + queued + history). */
    jobsFor(hostId: string | null): DownloadJob[] {
      const downloads = useDownloadsStore();
      if (hostId) {
        const host: DownloadHostState | undefined = downloads.hostStates[hostId];
        if (!host) return [];
        return [...host.activeJobs, ...host.queued, ...host.history];
      }
      return [...downloads.activeJobs, ...downloads.queued, ...downloads.history];
    },
    /** Inspect the watched bucket; resume or give up on a NEW terminal job. */
    check() {
      const pending = this.pending;
      if (!pending) return;
      const fresh = this.jobsFor(pending.hostId)
        .filter(isTerminal)
        .filter((job) => job.model === pending.model && !this.seenTerminal.includes(job.id));
      const done = fresh.find((job) => job.status === "completed");
      if (done) {
        this.pending = null;
        useToastStore().push(`${pending.model} is ready on ${pending.hostLabel} — generating`);
        void useGenerationStore().submitBatch(pending.request, pending.batch, pending.route)
          .settled;
        return;
      }
      const failed = fresh.find((job) => job.status !== "completed");
      if (failed) {
        this.pending = null;
        useToastStore().push(
          `Download of ${pending.model} ${failed.status === "cancelled" ? "was cancelled" : "failed"}${failed.error ? ` — ${failed.error}` : ""}; generation not resumed.`,
          "error",
        );
      }
    },
    /** One reactive watcher per store instance over the downloads state. */
    ensureWatcher() {
      if ((this as unknown as { _watching?: boolean })._watching) return;
      (this as unknown as { _watching?: boolean })._watching = true;
      const downloads = useDownloadsStore();
      watch(
        () => {
          const pending = this.pending;
          if (!pending) return "";
          return this.jobsFor(pending.hostId)
            .map((job) => `${job.id}:${job.status}`)
            .join("|");
        },
        () => this.check(),
      );
      // Keep TS satisfied about the unused capture in strict builds.
      void downloads;
    },
  },
});
