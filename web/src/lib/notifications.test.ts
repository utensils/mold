import { beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick, ref } from "vue";
import { flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import {
  installNotifications,
  markGalleryVisited,
  useNotificationSignals,
  __resetNotificationsForTest,
} from "./notifications";
import { resetNotifications, useNotifications } from "./toasts";
import type { Job } from "../composables/useGenerateStream";
import type { UseDownloads } from "../composables/useDownloads";
import type { DownloadJobWire } from "../types";
import { useNotificationsStore } from "@studio/stores/notifications";

const listStoredHosts = vi.hoisted(() => vi.fn());
const hostStatus = vi.hoisted(() => vi.fn());
vi.mock("./hostRegistry", () => ({ listStoredHosts }));
vi.mock("../components/machines/hostClient", () => ({ hostStatus }));

function job(over: Partial<Job> = {}): Job {
  return {
    id: "j1",
    state: "running",
    startedAt: Date.now(),
    ...over,
  } as unknown as Job;
}

function dljob(over: Partial<DownloadJobWire> = {}): DownloadJobWire {
  return {
    id: "d1",
    model: "flux-dev:q4",
    catalog_id: null,
    status: "completed",
    files_done: 1,
    files_total: 1,
    bytes_done: 10,
    bytes_total: 10,
    current_file: null,
    started_at: null,
    completed_at: Date.now(),
    error: null,
    ...over,
  };
}

function fakeDownloads(history: DownloadJobWire[]): UseDownloads {
  return { history: ref(history) } as unknown as UseDownloads;
}

const toasts = () => useNotifications().toasts;

beforeEach(() => {
  setActivePinia(createPinia());
  resetNotifications();
  __resetNotificationsForTest();
  listStoredHosts.mockReset();
  hostStatus.mockReset();
  listStoredHosts.mockReturnValue([]);
});

describe("notifications — generation done (G11a)", () => {
  it("toasts and bumps fresh prints off the Create route", async () => {
    const jobs = ref<Job[]>([job({ state: "running" })]);
    const stop = installNotifications({
      jobs,
      downloads: fakeDownloads([]),
      currentRouteName: () => "models",
      hostPollMs: 1_000_000,
    });

    jobs.value = [job({ state: "done" })];
    await nextTick();

    expect(toasts().some((t) => t.text.includes("generated"))).toBe(true);
    expect(useNotificationSignals().freshPrintCount.value).toBe(1);
    stop();
  });

  it("suppresses the toast on Create but still bumps the gallery dot", async () => {
    const jobs = ref<Job[]>([job({ state: "running" })]);
    const stop = installNotifications({
      jobs,
      downloads: fakeDownloads([]),
      currentRouteName: () => "create",
      hostPollMs: 1_000_000,
    });

    jobs.value = [job({ state: "done" })];
    await nextTick();

    expect(toasts().some((t) => t.text.includes("generated"))).toBe(false);
    expect(useNotificationSignals().freshPrintCount.value).toBe(1);
    stop();
  });

  it("does not bump the dot while already on the gallery", async () => {
    const jobs = ref<Job[]>([job({ state: "running" })]);
    const stop = installNotifications({
      jobs,
      downloads: fakeDownloads([]),
      currentRouteName: () => "gallery",
      hostPollMs: 1_000_000,
    });

    jobs.value = [job({ state: "done" })];
    await nextTick();

    expect(useNotificationSignals().freshPrintCount.value).toBe(0);
    markGalleryVisited();
    expect(useNotificationSignals().freshPrintCount.value).toBe(0);
    stop();
  });
});

describe("notifications — downloads (G11b)", () => {
  it("toasts on install and on failure, silent on cancel", async () => {
    const downloads = fakeDownloads([]);
    const stop = installNotifications({
      jobs: ref<Job[]>([]),
      downloads,
      currentRouteName: () => "models",
      hostPollMs: 1_000_000,
    });

    downloads.history.value = [dljob({ id: "d1", status: "completed" })];
    await nextTick();
    downloads.history.value = [
      ...downloads.history.value,
      dljob({ id: "d2", model: "sdxl:base", status: "failed" }),
      dljob({ id: "d3", model: "z-image:turbo", status: "cancelled" }),
    ];
    await nextTick();

    const texts = toasts().map((t) => t.text);
    expect(texts.some((t) => t.includes("installed flux-dev:q4"))).toBe(true);
    expect(texts.some((t) => t.includes("sdxl:base failed"))).toBe(true);
    expect(texts.some((t) => t.includes("z-image:turbo"))).toBe(false);
    stop();
  });

  it("retries a failed pull from the notification message", async () => {
    const downloads = fakeDownloads([]);
    downloads.enqueue = vi.fn().mockResolvedValue(null);
    const stop = installNotifications({
      jobs: ref<Job[]>([]),
      downloads,
      currentRouteName: () => "models",
      hostPollMs: 1_000_000,
    });

    downloads.history.value = [dljob({ status: "failed" })];
    await nextTick();

    const failedToast = toasts().find((entry) =>
      entry.text.includes("failed to download"),
    );
    expect(failedToast?.text).toBe("flux-dev:q4 failed to download");
    expect(failedToast?.actionLabel).toBeUndefined();
    await useNotificationsStore().entries[0]!.action?.run();

    expect(downloads.enqueue).toHaveBeenCalledOnce();
    expect(downloads.enqueue).toHaveBeenCalledWith("flux-dev:q4");
    stop();
  });

  it("never toasts for history already present at install", async () => {
    const downloads = fakeDownloads([
      dljob({ id: "old", status: "completed" }),
    ]);
    const stop = installNotifications({
      jobs: ref<Job[]>([]),
      downloads,
      currentRouteName: () => "models",
      hostPollMs: 1_000_000,
    });
    // Touch the ref so the watcher runs at least once.
    downloads.history.value = [...downloads.history.value];
    await nextTick();
    expect(toasts().length).toBe(0);
    stop();
  });
});

describe("notifications — host offline (G11c)", () => {
  it("toasts once and lights the machines dot when a host is unreachable", async () => {
    listStoredHosts.mockReturnValue([
      { id: "h1", name: "Studio", url: "http://studio:7680" },
    ]);
    hostStatus.mockRejectedValue(new Error("unreachable"));

    const stop = installNotifications({
      jobs: ref<Job[]>([]),
      downloads: fakeDownloads([]),
      currentRouteName: () => "models",
      hostPollMs: 1_000_000,
    });
    await flushPromises();

    const offline = toasts().filter((t) => t.text === "Can't reach Studio");
    expect(offline).toHaveLength(1);
    expect(offline[0]!.kind).toBe("warning");
    expect(useNotificationSignals().hasOfflineHost.value).toBe(true);
    stop();
  });

  it("withdraws the warning and confirms the automatic reconnect", async () => {
    vi.useFakeTimers();
    listStoredHosts.mockReturnValue([
      { id: "h1", name: "Studio", url: "http://studio:7680" },
    ]);
    hostStatus
      .mockRejectedValueOnce(new Error("unreachable"))
      .mockResolvedValue({});

    const stop = installNotifications({
      jobs: ref<Job[]>([]),
      downloads: fakeDownloads([]),
      currentRouteName: () => "models",
      hostPollMs: 1_000,
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(toasts().some((t) => t.text === "Can't reach Studio")).toBe(true);

    // The poll keeps retrying; the next successful probe is the reconnect.
    await vi.advanceTimersByTimeAsync(1_000);
    expect(toasts().some((t) => t.text === "Can't reach Studio")).toBe(false);
    const back = toasts().filter((t) => t.text === "Reconnected to Studio");
    expect(back).toHaveLength(1);
    expect(back[0]!.kind).toBe("success");
    expect(useNotificationSignals().hasOfflineHost.value).toBe(false);

    // A host that never dropped must not be congratulated on every poll.
    await vi.advanceTimersByTimeAsync(1_000);
    expect(
      toasts().filter((t) => t.text === "Reconnected to Studio"),
    ).toHaveLength(1);
    stop();
    vi.useRealTimers();
  });

  it("discards a probe that settles after a later poll already reported", async () => {
    vi.useFakeTimers();
    listStoredHosts.mockReturnValue([
      { id: "h1", name: "Studio", url: "http://studio:7680" },
    ]);
    // Poll 1 hangs past its own interval; poll 2 succeeds. The stale rejection
    // must not resurrect the offline warning for a host that is up.
    const rejecters: ((reason: Error) => void)[] = [];
    hostStatus
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejecters.push(reject);
          }),
      )
      .mockResolvedValue({});

    const stop = installNotifications({
      jobs: ref<Job[]>([]),
      downloads: fakeDownloads([]),
      currentRouteName: () => "models",
      hostPollMs: 1_000,
    });
    await vi.advanceTimersByTimeAsync(1_000);
    rejecters[0]?.(new Error("timed out"));
    await vi.advanceTimersByTimeAsync(1_000);

    expect(toasts().some((t) => t.text === "Can't reach Studio")).toBe(false);
    expect(useNotificationSignals().hasOfflineHost.value).toBe(false);
    stop();
    vi.useRealTimers();
  });

  it("does nothing when no remote hosts are registered", async () => {
    listStoredHosts.mockReturnValue([]);
    const stop = installNotifications({
      jobs: ref<Job[]>([]),
      downloads: fakeDownloads([]),
      currentRouteName: () => "models",
      hostPollMs: 1_000_000,
    });
    await flushPromises();

    expect(toasts().length).toBe(0);
    expect(useNotificationSignals().hasOfflineHost.value).toBe(false);
    expect(hostStatus).not.toHaveBeenCalled();
    stop();
  });
});
