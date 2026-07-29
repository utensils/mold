import { defineStore } from "pinia";
import { ipc, type UpdateCandidate, type UpdateChannel, type UpdateProgress } from "../lib/ipc";
import { notifyUpdateAvailable } from "../lib/notify";
import { useAppPrefsStore } from "./appPrefs";

export type UpdatePhase =
  | "idle"
  | "checking"
  | "up-to-date"
  | "available"
  | "downloading"
  | "verifying"
  | "staging"
  | "installing"
  | "failed"
  | "unsupported";

export interface UpdateCommandError {
  code: string;
  message: string;
  disposition: "unchanged";
  retryable: boolean;
}

const BUSY_PHASES: UpdatePhase[] = [
  "checking",
  "downloading",
  "verifying",
  "staging",
  "installing",
];
const UPDATE_CHECK_INTERVAL_MS = 60 * 60 * 1_000;

export function normalizeUpdateError(error: unknown): UpdateCommandError {
  if (error && typeof error === "object") {
    const candidate = error as Partial<UpdateCommandError>;
    if (typeof candidate.message === "string") {
      return {
        code: typeof candidate.code === "string" ? candidate.code : "unknown",
        message: candidate.message,
        disposition: "unchanged",
        retryable: candidate.retryable !== false,
      };
    }
  }
  return {
    code: "unknown",
    message: error instanceof Error ? error.message : String(error),
    disposition: "unchanged",
    retryable: true,
  };
}

export const useUpdaterStore = defineStore("updater", {
  state: () => ({
    phase: "idle" as UpdatePhase,
    currentVersion: null as string | null,
    candidate: null as UpdateCandidate | null,
    checkedAt: null as string | null,
    downloadedBytes: 0,
    totalBytes: null as number | null,
    error: null as UpdateCommandError | null,
    dismissedCandidateId: null as string | null,
    initialized: false,
    progressUnlisten: null as (() => void) | null,
    checkTimer: null as ReturnType<typeof setInterval> | null,
  }),
  getters: {
    isBusy: (state): boolean => BUSY_PHASES.includes(state.phase),
    shouldNotify: (state): boolean =>
      state.phase === "available" &&
      state.candidate !== null &&
      state.dismissedCandidateId !== state.candidate.id,
    percent: (state): number | null => {
      if (state.totalBytes === null || state.totalBytes <= 0) return null;
      return Math.min(100, Math.max(0, (state.downloadedBytes / state.totalBytes) * 100));
    },
  },
  actions: {
    async init(): Promise<void> {
      if (this.initialized) return;
      this.initialized = true;

      await this.subscribeToProgress().catch(() => {});
      // The manifest check is background-only and never delays app startup.
      void this.check();
      // A long-running or freshly updated app must keep discovering later
      // releases without requiring a manual check or another relaunch.
      this.checkTimer = setInterval(() => void this.checkAutomatically(), UPDATE_CHECK_INTERVAL_MS);
    },

    async checkAutomatically(): Promise<void> {
      // Keep the native pending update (and a dismissed notification) stable
      // until the user installs it or explicitly requests another check.
      if (this.isBusy || this.candidate || this.phase === "unsupported") return;
      await this.check();
    },

    async subscribeToProgress(): Promise<void> {
      if (this.progressUnlisten) return;
      this.progressUnlisten = await ipc.onUpdaterProgress((event) => this.applyProgress(event));
    },

    applyProgress(event: UpdateProgress): void {
      if (event.candidateId && event.candidateId !== this.candidate?.id) return;
      this.phase = event.phase;
      this.downloadedBytes = Math.max(0, event.downloadedBytes ?? this.downloadedBytes);
      this.totalBytes = event.totalBytes !== null ? Math.max(0, event.totalBytes) : null;
    },

    async check(): Promise<void> {
      if (this.isBusy) return;
      this.phase = "checking";
      this.error = null;
      this.candidate = null;
      this.downloadedBytes = 0;
      this.totalBytes = null;

      try {
        const result = await ipc.checkForUpdates(useAppPrefsStore().updateChannel);
        this.currentVersion = result.currentVersion;
        this.checkedAt = result.checkedAt;
        this.candidate = result.candidate;
        this.phase = !result.supported
          ? "unsupported"
          : result.candidate
            ? "available"
            : "up-to-date";
        if (result.candidate) notifyUpdateAvailable(result.candidate.version);
      } catch (error) {
        const normalized = normalizeUpdateError(error);
        this.error = normalized;
        if (!this.error.retryable) this.candidate = null;
        this.phase = "failed";
      }
    },

    async setChannel(channel: UpdateChannel): Promise<void> {
      if (this.isBusy || useAppPrefsStore().updateChannel === channel) {
        return;
      }
      this.candidate = null;
      this.error = null;
      this.phase = "idle";
      try {
        await useAppPrefsStore().update({ updateChannel: channel });
        await this.check();
      } catch (error) {
        this.error = normalizeUpdateError(error);
        this.phase = "failed";
      }
    },

    async install(): Promise<void> {
      const candidate = this.candidate;
      if (!candidate || this.isBusy) return;
      this.phase = "downloading";
      this.error = null;
      this.downloadedBytes = 0;
      this.totalBytes = null;
      try {
        await ipc.installPendingUpdate(candidate.id);
        // A successful native command restarts the app. Keep the UI truthful if
        // process shutdown takes a moment instead of pretending the old binary
        // is already current.
        this.phase = "installing";
      } catch (error) {
        this.error = normalizeUpdateError(error);
        if (!this.error.retryable) this.candidate = null;
        this.phase = "failed";
      }
    },

    clearError(): void {
      this.error = null;
      this.phase = this.candidate ? "available" : this.checkedAt ? "up-to-date" : "idle";
    },

    dismissCandidate(): void {
      this.dismissedCandidateId = this.candidate?.id ?? null;
    },
  },
});
