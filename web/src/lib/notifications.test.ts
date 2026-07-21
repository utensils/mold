import { beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick, ref } from "vue";
import { flushPromises } from "@vue/test-utils";
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

    expect(
      toasts().filter((t) => t.text.includes("can't reach Studio")).length,
    ).toBe(1);
    expect(useNotificationSignals().hasOfflineHost.value).toBe(true);
    stop();
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
