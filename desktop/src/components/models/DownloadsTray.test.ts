import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import DownloadsTray from "./DownloadsTray.vue";
import { useConnectionStore } from "../../stores/connection";
import { useDownloadsStore } from "../../stores/downloads";
import type { DownloadJob } from "../../lib/api/types";

const { apiFetchTo, sseStream, startCatalogDownload } = vi.hoisted(() => ({
  apiFetchTo: vi.fn(),
  sseStream: vi.fn().mockResolvedValue(undefined),
  startCatalogDownload: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/api/client")>()),
  apiFetchTo,
}));
vi.mock("../../lib/api/sse", () => ({ sseStream }));
vi.mock("../../lib/api/catalog", () => ({ startCatalogDownload }));

function job(overrides: Partial<DownloadJob> = {}): DownloadJob {
  return {
    id: "j1",
    model: "flux-dev:q4",
    status: "active",
    files_done: 1,
    files_total: 4,
    bytes_done: 25,
    bytes_total: 100,
    ...overrides,
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
  apiFetchTo.mockReset();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 204 }));
  sseStream.mockReset();
  sseStream.mockImplementation((_path: string, options: { onOpen?: () => void }) => {
    options.onOpen?.();
    return Promise.resolve();
  });
  startCatalogDownload.mockReset();
  startCatalogDownload.mockResolvedValue(undefined);
});

describe("DownloadsTray a11y", () => {
  it("exposes each download as a determinate progressbar", () => {
    const store = useDownloadsStore();
    store.activeJobs = [job()];
    const wrapper = mount(DownloadsTray);

    const bar = wrapper.get('[role="progressbar"]');
    expect(bar.attributes("aria-valuemin")).toBe("0");
    expect(bar.attributes("aria-valuemax")).toBe("100");
    expect(bar.attributes("aria-valuenow")).toBe("25");
    expect(bar.attributes("aria-label")).toContain("flux-dev:q4");
  });

  it("labels the cancel control with the model name and target host", () => {
    const store = useDownloadsStore();
    store.activeJobs = [job()];
    const wrapper = mount(DownloadsTray);

    const cancel = wrapper.get("button");
    expect(cancel.attributes("aria-label")).toBe("Cancel download of flux-dev:q4 on This device");
  });

  it("renders a source glyph for each download row", () => {
    const store = useDownloadsStore();
    store.activeJobs = [job(), job({ id: "j2", model: "cv:8001" })];
    const wrapper = mount(DownloadsTray);

    expect(wrapper.find('[data-source="local"]').exists()).toBe(true);
    expect(wrapper.find('[data-source="civitai"]').exists()).toBe(true);
  });

  it("labels primary rows with the built-in engine's name", () => {
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    const store = useDownloadsStore();
    store.activeJobs = [job()];
    const wrapper = mount(DownloadsTray);

    // The primary is always the built-in engine (host id "local").
    expect(wrapper.get('[data-test="download-host"]').text()).toContain("This device");
  });

  it("shows and cancels downloads on an explicitly selected extra host", async () => {
    const store = useDownloadsStore();
    store.hostStates.hal9000 = {
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      activeJobs: [job()],
      queued: [],
      history: [],
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
    };
    const wrapper = mount(DownloadsTray);

    expect(wrapper.text()).toContain("hal9000");
    const cancel = wrapper.get("button");
    expect(cancel.attributes("aria-label")).toBe("Cancel download of flux-dev:q4 on hal9000");
    await cancel.trigger("click");
    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      "/api/downloads/j1",
      { method: "DELETE" },
    );
  });

  it("restarts an extra-host stream when its API key changes", () => {
    const store = useDownloadsStore();
    const host = {
      id: "hal9000",
      label: "hal9000",
      kind: "remote" as const,
      baseUrl: "http://hal9000:7680",
      apiKey: "old-key",
      status: "ready" as const,
      primary: false,
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      instanceId: null,
    };
    store.subscribeHost(host);
    store.hostStates.hal9000!.activeJobs = [job()];
    store.hostStates.hal9000!.cancelling = ["j1"];
    const oldSignal = store.hostStates.hal9000!.abort!.signal;

    store.subscribeHost({ ...host, apiKey: "new-key" });

    expect(oldSignal.aborted).toBe(true);
    expect(store.hostStates.hal9000!.target.apiKey).toBe("new-key");
    expect(store.hostStates.hal9000!.activeJobs).toEqual([job()]);
    expect(store.hostStates.hal9000!.cancelling).toEqual(["j1"]);
    expect(sseStream).toHaveBeenCalledTimes(2);
  });

  it("rotates the primary stream and cancellation target with the selected host", async () => {
    const store = useDownloadsStore();
    const primary = {
      id: "host-a",
      label: "Host A",
      kind: "remote" as const,
      baseUrl: "http://host-a:7680",
      apiKey: "key-a",
      status: "ready" as const,
      primary: true,
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      instanceId: null,
    };
    await store.subscribe(primary);
    const oldSignal = store.abort!.signal;

    await store.subscribe({
      ...primary,
      id: "host-b",
      label: "Host B",
      baseUrl: "http://host-b:7680",
      apiKey: "key-b",
    });
    store.activeJobs = [job()];
    await store.cancel("j1", "host-b");

    expect(oldSignal.aborted).toBe(true);
    expect(store.primaryHostId).toBe("host-b");
    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://host-b:7680", apiKey: "key-b" },
      "/api/downloads/j1",
      { method: "DELETE" },
    );
  });

  it("rejects readiness and resets state when the stream cannot open", async () => {
    sseStream.mockImplementationOnce(
      (_path: string, options: { onOpenError?: (error: Error) => void }) => {
        options.onOpenError?.(new Error("HTTP 401"));
        return Promise.resolve();
      },
    );
    const store = useDownloadsStore();
    const opening = store.subscribe({
      id: "host-a",
      label: "Host A",
      kind: "remote",
      baseUrl: "http://host-a:7680",
      apiKey: "stale-key",
      status: "ready",
      primary: true,
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      instanceId: null,
    });

    await expect(opening).rejects.toThrow("HTTP 401");
    expect(store.subscribed).toBe(false);
    expect(store.primaryTarget).toBeNull();
  });

  it("shows status chip, current file, file counts, and ETA on active rows", () => {
    const store = useDownloadsStore();
    store.activeJobs = [
      job({ current_file: "unet.safetensors", bytes_done: 10_000_000, bytes_total: 100_000_000 }),
    ];
    store.rateSamples["primary:j1"] = [
      { ts: 0, bytes: 0 },
      { ts: 10_000, bytes: 10_000_000 },
    ];
    const wrapper = mount(DownloadsTray);

    expect(wrapper.get('[data-test="download-status"]').text()).toBe("active");
    expect(wrapper.get('[data-test="download-current-file"]').text()).toBe("unet.safetensors");
    expect(wrapper.get('[data-test="download-files"]').text()).toContain("1/4");
    expect(wrapper.get('[data-test="download-eta"]').text()).toContain("1m 30s");
  });

  it("renders failed jobs in a collapsed history section with their error", async () => {
    const store = useDownloadsStore();
    store.history = [job({ status: "failed", error: "connection reset" })];
    const wrapper = mount(DownloadsTray);

    // Tray is visible on history alone, collapsed by default.
    const toggle = wrapper.get('[data-test="history-toggle"]');
    expect(toggle.text()).toContain("History (1)");
    expect(wrapper.find('[data-test="history-list"]').exists()).toBe(false);

    await toggle.trigger("click");
    const list = wrapper.get('[data-test="history-list"]');
    expect(list.text()).toContain("flux-dev:q4");
    expect(list.text()).toContain("failed");
    expect(list.text()).toContain("connection reset");
  });

  it("retries a failed history job on its own host", async () => {
    const store = useDownloadsStore();
    store.hostStates.hal9000 = {
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      activeJobs: [],
      queued: [],
      history: [job({ status: "failed", error: "boom" })],
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
    };
    const wrapper = mount(DownloadsTray);

    await wrapper.get('[data-test="history-toggle"]').trigger("click");
    await wrapper.get('[data-test="download-retry"]').trigger("click");
    expect(startCatalogDownload).toHaveBeenCalledWith(
      "flux-dev:q4",
      { baseUrl: "http://hal9000:7680", apiKey: "remote-key" },
      true,
    );
  });

  it("does not offer retry for completed or cancelled history entries", async () => {
    const store = useDownloadsStore();
    store.history = [
      job({ id: "c1", status: "completed" }),
      job({ id: "x1", status: "cancelled" }),
    ];
    const wrapper = mount(DownloadsTray);

    await wrapper.get('[data-test="history-toggle"]').trigger("click");
    expect(wrapper.find('[data-test="download-retry"]').exists()).toBe(false);
  });

  it("removes an extra stream when that host becomes primary", async () => {
    const store = useDownloadsStore();
    const extraAbort = new AbortController();
    store.hostStates.hal9000 = {
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: "key" },
      activeJobs: [job()],
      queued: [],
      history: [],
      subscribed: true,
      abort: extraAbort,
      cancelling: [],
      ready: Promise.resolve(),
    };

    await store.subscribe({
      id: "hal9000",
      label: "hal9000",
      kind: "remote",
      baseUrl: "http://hal9000:7680",
      apiKey: "key",
      status: "ready",
      primary: true,
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      instanceId: null,
    });

    expect(extraAbort.signal.aborted).toBe(true);
    expect(store.hostStates.hal9000).toBeUndefined();
  });
});
