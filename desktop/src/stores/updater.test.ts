import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

const checkForUpdates = vi.fn();
const installPendingUpdate = vi.fn();
const appSettingsSet = vi.fn();
const appSettingsGet = vi.fn();
const notifyUpdateAvailable = vi.fn();
const ONE_HOUR_MS = 60 * 60 * 1_000;
let progressListener: ((event: unknown) => void) | null = null;

vi.mock("../lib/ipc", () => ({
  ipc: {
    checkForUpdates: (...args: unknown[]) => checkForUpdates(...args),
    installPendingUpdate: (...args: unknown[]) => installPendingUpdate(...args),
    onUpdaterProgress: (listener: (event: unknown) => void) => {
      progressListener = listener;
      return Promise.resolve(() => {
        progressListener = null;
      });
    },
    appSettingsSet: (...args: unknown[]) => appSettingsSet(...args),
    appSettingsGet: (...args: unknown[]) => appSettingsGet(...args),
  },
}));

vi.mock("../lib/notify", () => ({
  notifyUpdateAvailable: (...args: unknown[]) => notifyUpdateAvailable(...args),
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
    theme: "mocha",
    matchSystem: false,
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
    saveRemoteOutputs: true,
    navRailWidth: null,
    generateParamsWidth: null,
    sidebarCollapsed: false,
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
  appSettingsGet.mockReset();
  appSettingsGet.mockImplementation(() => Promise.resolve(settings()));
  progressListener = null;
  checkForUpdates.mockReset();
  checkForUpdates.mockResolvedValue(checkResult());
  installPendingUpdate.mockReset();
  installPendingUpdate.mockResolvedValue(undefined);
  appSettingsSet.mockReset();
  appSettingsSet.mockResolvedValue(undefined);
  notifyUpdateAvailable.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("updater store", () => {
  it("performs a non-installing automatic check and exposes an update notification", async () => {
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

    expect(checkForUpdates).toHaveBeenCalledWith("stable");
    expect(installPendingUpdate).not.toHaveBeenCalled();
    expect(updater.phase).toBe("available");
    expect(updater.shouldNotify).toBe(true);
    expect(notifyUpdateAvailable).toHaveBeenCalledWith("0.17.0");
  });

  it("keeps checking after the startup check reports the app is up to date", async () => {
    vi.useFakeTimers();
    const updater = useUpdaterStore();

    await updater.init();
    await vi.advanceTimersByTimeAsync(ONE_HOUR_MS);

    expect(checkForUpdates).toHaveBeenCalledTimes(2);
  });

  it("does not replace an update that is already waiting for the user", async () => {
    vi.useFakeTimers();
    checkForUpdates.mockResolvedValue(
      checkResult({
        candidate: {
          id: "candidate-1",
          version: "0.17.0",
          publishedAt: null,
          notes: null,
        },
      }),
    );
    const updater = useUpdaterStore();

    await updater.init();
    await vi.advanceTimersByTimeAsync(ONE_HOUR_MS);

    expect(checkForUpdates).toHaveBeenCalledTimes(1);
    expect(updater.candidate?.id).toBe("candidate-1");
  });

  it("dismisses only the currently announced update", async () => {
    const updater = useUpdaterStore();
    updater.candidate = {
      id: "candidate-1",
      version: "0.17.0",
      publishedAt: null,
      notes: null,
    };
    updater.phase = "available";
    updater.dismissCandidate();
    expect(updater.shouldNotify).toBe(false);

    updater.candidate = {
      id: "candidate-2",
      version: "0.18.0",
      publishedAt: null,
      notes: null,
    };
    expect(updater.shouldNotify).toBe(true);
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
