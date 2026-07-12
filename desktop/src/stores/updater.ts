import { defineStore } from "pinia";
import {
  ipc,
  type UpdateCandidate,
  type UpdateChannel,
  type UpdateProgress,
  type UpdateRecovery,
} from "../lib/ipc";
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
  | "rolling-back"
  | "failed"
  | "unsupported";

export interface UpdateCommandError {
  code: string;
  message: string;
  disposition: "unchanged" | "rolled-back" | "rollback-failed";
  retryable: boolean;
}

const BUSY_PHASES: UpdatePhase[] = [
  "checking",
  "downloading",
  "verifying",
  "staging",
  "installing",
  "rolling-back",
];

export function normalizeUpdateError(error: unknown): UpdateCommandError {
  if (error && typeof error === "object") {
    const candidate = error as Partial<UpdateCommandError>;
    if (typeof candidate.message === "string") {
      return {
        code: typeof candidate.code === "string" ? candidate.code : "unknown",
        message: candidate.message,
        disposition:
          candidate.disposition === "rolled-back" || candidate.disposition === "rollback-failed"
            ? candidate.disposition
            : "unchanged",
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
    recovery: null as UpdateRecovery | null,
    initialized: false,
    progressUnlisten: null as (() => void) | null,
  }),
  getters: {
    isBusy: (state): boolean => BUSY_PHASES.includes(state.phase),
    hasTerminalRecovery: (state): boolean =>
      state.recovery?.rollbackFailed === true || state.error?.disposition === "rollback-failed",
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
      try {
        this.recovery = await ipc.takeUpdateRecovery();
      } catch (error) {
        this.error = normalizeUpdateError(error);
        this.phase = "failed";
        return;
      }
      if (this.recovery?.rollbackFailed) return;
      // Network discovery is not part of candidate health. Let it continue in
      // the background so a slow manifest cannot delay startup confirmation.
      void this.check();
    },

    async confirmReady(): Promise<void> {
      // App.vue calls this only after preferences, connection startup, and the
      // first visible native-window paint have completed. The backend treats
      // this as the beginning of probation, not immediate success.
      try {
        await ipc.confirmUpdateHealthy();
      } catch (error) {
        this.error = normalizeUpdateError(error);
        if (this.error.disposition === "rollback-failed") this.candidate = null;
        this.phase = "failed";
      }
    },

    async subscribeToProgress(): Promise<void> {
      if (this.progressUnlisten) return;
      this.progressUnlisten = await ipc.onUpdaterProgress((event) => this.applyProgress(event));
    },

    applyProgress(event: UpdateProgress): void {
      if (this.hasTerminalRecovery) return;
      if (event.candidateId && event.candidateId !== this.candidate?.id) return;
      this.phase = event.phase;
      this.downloadedBytes = Math.max(0, event.downloadedBytes ?? this.downloadedBytes);
      this.totalBytes = event.totalBytes !== null ? Math.max(0, event.totalBytes) : null;
    },

    async check(): Promise<void> {
      if (this.isBusy || this.hasTerminalRecovery) return;
      this.phase = "checking";
      this.error = null;
      this.candidate = null;
      this.downloadedBytes = 0;
      this.totalBytes = null;

      try {
        const result = await ipc.checkForUpdates(useAppPrefsStore().updateChannel);
        if (this.hasTerminalRecovery) return;
        this.currentVersion = result.currentVersion;
        this.checkedAt = result.checkedAt;
        this.candidate = result.candidate;
        this.phase = !result.supported
          ? "unsupported"
          : result.candidate
            ? "available"
            : "up-to-date";
      } catch (error) {
        const normalized = normalizeUpdateError(error);
        if (this.hasTerminalRecovery && normalized.disposition !== "rollback-failed") return;
        this.error = normalized;
        if (this.error.disposition === "rollback-failed" || !this.error.retryable) {
          this.candidate = null;
        }
        this.phase = "failed";
      }
    },

    async setChannel(channel: UpdateChannel): Promise<void> {
      if (this.isBusy || this.hasTerminalRecovery || useAppPrefsStore().updateChannel === channel) {
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
        if (this.error.disposition === "rollback-failed") this.candidate = null;
        this.phase = "failed";
      }
    },

    async install(): Promise<void> {
      const candidate = this.candidate;
      if (!candidate || this.isBusy || this.hasTerminalRecovery) return;
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
        if (this.error.disposition === "rollback-failed" || !this.error.retryable) {
          this.candidate = null;
        }
        this.phase = "failed";
      }
    },

    clearError(): void {
      if (this.error?.disposition === "rollback-failed") return;
      this.error = null;
      this.phase = this.candidate ? "available" : this.checkedAt ? "up-to-date" : "idle";
    },

    dismissRecovery(): void {
      if (this.recovery?.rollbackFailed) return;
      this.recovery = null;
    },
  },
});
