import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

const checkForUpdates = vi.fn();
const installPendingUpdate = vi.fn();
const takeUpdateRecovery = vi.fn();
const confirmUpdateHealthy = vi.fn();
const appSettingsSet = vi.fn();
let progressListener: ((event: unknown) => void) | null = null;

vi.mock("../lib/ipc", () => ({
  ipc: {
    checkForUpdates: (...args: unknown[]) => checkForUpdates(...args),
    installPendingUpdate: (...args: unknown[]) => installPendingUpdate(...args),
    takeUpdateRecovery: (...args: unknown[]) => takeUpdateRecovery(...args),
    confirmUpdateHealthy: (...args: unknown[]) => confirmUpdateHealthy(...args),
    onUpdaterProgress: (listener: (event: unknown) => void) => {
      progressListener = listener;
      return Promise.resolve(() => {
        progressListener = null;
      });
    },
    appSettingsSet: (...args: unknown[]) => appSettingsSet(...args),
  },
}));

import type { AppSettings, UpdateCheckResult, UpdateProgress } from "../lib/ipc";
import { useAppPrefsStore } from "./appPrefs";
import { useUpdaterStore } from "./updater";

function settings(updateChannel: AppSettings["updateChannel"] = "stable"): AppSettings {
  return {
    mode: "local",
    remoteUrl: null,
    remoteApiKey: null,
    lastRoute: null,
    engineEnv: {},
    theme: "system",
    themeFamily: "mold",
    notifications: true,
    dockBadge: true,
    restoreLastRoute: false,
    runpodIncludeHfToken: false,
    runpodNetworkVolumeId: null,
    uiScalePercent: 100,
    updateChannel,
    savedHosts: [],
    connectedHostIds: [],
    generateTargetHost: null,
  };
}

function checkResult(overrides: Partial<UpdateCheckResult> = {}): UpdateCheckResult {
  return {
    supported: true,
    channel: "stable",
    currentVersion: "0.16.0",
    checkedAt: "2026-07-12T18:00:00Z",
    candidate: null,
    ...overrides,
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
  useAppPrefsStore().settings = settings();
  progressListener = null;
  checkForUpdates.mockReset();
  checkForUpdates.mockResolvedValue(checkResult());
  installPendingUpdate.mockReset();
  installPendingUpdate.mockResolvedValue(undefined);
  takeUpdateRecovery.mockReset();
  takeUpdateRecovery.mockResolvedValue(null);
  confirmUpdateHealthy.mockReset();
  confirmUpdateHealthy.mockResolvedValue(undefined);
  appSettingsSet.mockReset();
  appSettingsSet.mockResolvedValue(undefined);
});

describe("updater store", () => {
  it("initializes recovery handling and performs a non-installing automatic check", async () => {
    takeUpdateRecovery.mockResolvedValue({
      restoredVersion: "0.16.0",
      failedVersion: "0.17.0",
      message: "The new app did not finish startup.",
    });
    checkForUpdates.mockResolvedValue(
      checkResult({
        candidate: {
          id: "stable-0170",
          version: "0.17.0",
          publishedAt: "2026-07-12T17:00:00Z",
          notes: "Faster previews",
        },
      }),
    );

    const updater = useUpdaterStore();
    await updater.init();

    await vi.waitFor(() => expect(updater.phase).toBe("available"));

    expect(takeUpdateRecovery).toHaveBeenCalledOnce();
    expect(confirmUpdateHealthy).not.toHaveBeenCalled();
    expect(checkForUpdates).toHaveBeenCalledWith("stable");
    expect(installPendingUpdate).not.toHaveBeenCalled();
    expect(updater.phase).toBe("available");
    expect(updater.recovery?.restoredVersion).toBe("0.16.0");
  });

  it("confirms candidate health only when the shell declares critical startup ready", async () => {
    const updater = useUpdaterStore();
    await updater.init();

    expect(confirmUpdateHealthy).not.toHaveBeenCalled();
    await updater.confirmReady();
    expect(confirmUpdateHealthy).toHaveBeenCalledOnce();
  });

  it("surfaces a missing-watchdog health rejection as terminal recovery", async () => {
    confirmUpdateHealthy.mockRejectedValue({
      code: "rollback",
      message: "No rollback watchdog owns this process. Recovery: /tmp/Mold.app",
      disposition: "rollback-failed",
      retryable: false,
    });
    const updater = useUpdaterStore();

    await updater.confirmReady();

    expect(updater.phase).toBe("failed");
    expect(updater.hasTerminalRecovery).toBe(true);
    expect(updater.error?.message).toContain("/tmp/Mold.app");
  });

  it("does not let an in-flight check overwrite terminal recovery", async () => {
    let resolve!: (result: UpdateCheckResult) => void;
    checkForUpdates.mockReturnValue(new Promise<UpdateCheckResult>((done) => (resolve = done)));
    confirmUpdateHealthy.mockRejectedValue({
      code: "rollback",
      message: "No rollback watchdog owns this process.",
      disposition: "rollback-failed",
      retryable: false,
    });
    const updater = useUpdaterStore();

    const check = updater.check();
    await updater.confirmReady();
    resolve(checkResult());
    await check;

    expect(updater.phase).toBe("failed");
    expect(updater.error?.disposition).toBe("rollback-failed");
  });

  it("surfaces recovery reconciliation failures and skips the network check", async () => {
    takeUpdateRecovery.mockRejectedValue({
      code: "rollback",
      message: "The rollback watchdog could not start. Recovery: /tmp/Mold.app",
      disposition: "rollback-failed",
      retryable: false,
    });
    const updater = useUpdaterStore();

    await updater.init();

    expect(updater.phase).toBe("failed");
    expect(updater.error?.disposition).toBe("rollback-failed");
    expect(updater.error?.message).toContain("/tmp/Mold.app");
    expect(checkForUpdates).not.toHaveBeenCalled();
  });

  it("treats rollback-failed recovery as terminal across channel and check controls", async () => {
    takeUpdateRecovery.mockResolvedValue({
      restoredVersion: "0.16.0",
      failedVersion: "0.17.0",
      message: "Manual recovery is required.",
      rollbackFailed: true,
      backupPath: "/tmp/Mold.app",
    });
    const updater = useUpdaterStore();

    await updater.init();
    await updater.check();
    await updater.setChannel("nightly");

    expect(updater.hasTerminalRecovery).toBe(true);
    expect(checkForUpdates).not.toHaveBeenCalled();
    expect(appSettingsSet).not.toHaveBeenCalled();
    expect(useAppPrefsStore().updateChannel).toBe("stable");
  });

  it("clears a candidate after automatic rollback itself fails", async () => {
    installPendingUpdate.mockRejectedValue({
      code: "rollback",
      message: "Automatic rollback failed. Recovery copy: /tmp/Mold.app",
      disposition: "rollback-failed",
      retryable: false,
    });
    const updater = useUpdaterStore();
    updater.candidate = {
      id: "candidate-1",
      version: "0.17.0",
      publishedAt: null,
      notes: null,
    };
    updater.phase = "available";

    await updater.install();

    expect(updater.error?.disposition).toBe("rollback-failed");
    expect(updater.candidate).toBeNull();
  });

  it("transitions from checking to available and preserves release metadata", async () => {
    checkForUpdates.mockResolvedValue(
      checkResult({
        candidate: {
          id: "nightly-0170",
          version: "0.17.0-nightly.20260712",
          publishedAt: "2026-07-12T17:00:00Z",
          notes: "Main branch build",
        },
      }),
    );
    const updater = useUpdaterStore();

    await updater.check();

    expect(updater.phase).toBe("available");
    expect(updater.currentVersion).toBe("0.16.0");
    expect(updater.candidate?.notes).toBe("Main branch build");
  });

  it("does not launch duplicate checks while one is in flight", async () => {
    let resolve!: (result: UpdateCheckResult) => void;
    checkForUpdates.mockReturnValue(new Promise<UpdateCheckResult>((done) => (resolve = done)));
    const updater = useUpdaterStore();

    const first = updater.check();
    const second = updater.check();
    expect(checkForUpdates).toHaveBeenCalledTimes(1);
    resolve(checkResult());
    await Promise.all([first, second]);
    expect(updater.phase).toBe("up-to-date");
  });

  it("persists a channel change, invalidates the candidate, and checks that channel", async () => {
    const updater = useUpdaterStore();
    updater.candidate = {
      id: "old",
      version: "0.17.0",
      publishedAt: null,
      notes: null,
    };

    await updater.setChannel("nightly");

    expect(appSettingsSet).toHaveBeenLastCalledWith(
      expect.objectContaining({ updateChannel: "nightly" }),
    );
    expect(checkForUpdates).toHaveBeenCalledWith("nightly");
    expect(updater.candidate).toBeNull();
  });

  it("tracks determinate download progress only for the active candidate", async () => {
    const updater = useUpdaterStore();
    updater.candidate = {
      id: "candidate-1",
      version: "0.17.0",
      publishedAt: null,
      notes: null,
    };
    updater.phase = "available";
    await updater.subscribeToProgress();

    progressListener?.({
      candidateId: "stale",
      phase: "downloading",
      downloadedBytes: 90,
      totalBytes: 100,
    } satisfies UpdateProgress);
    expect(updater.downloadedBytes).toBe(0);

    progressListener?.({
      candidateId: "candidate-1",
      phase: "downloading",
      downloadedBytes: 25,
      totalBytes: 100,
    } satisfies UpdateProgress);
    expect(updater.phase).toBe("downloading");
    expect(updater.percent).toBe(25);
  });

  it("keeps the candidate retryable and describes an unchanged install after failure", async () => {
    installPendingUpdate.mockRejectedValue({
      code: "signature",
      message: "The update signature is invalid.",
      disposition: "unchanged",
      retryable: true,
    });
    const updater = useUpdaterStore();
    updater.candidate = {
      id: "candidate-1",
      version: "0.17.0",
      publishedAt: null,
      notes: null,
    };
    updater.phase = "available";

    await updater.install();

    expect(installPendingUpdate).toHaveBeenCalledWith("candidate-1");
    expect(updater.phase).toBe("failed");
    expect(updater.error?.disposition).toBe("unchanged");
    expect(updater.candidate?.id).toBe("candidate-1");
    expect(updater.isBusy).toBe(false);
  });
});
