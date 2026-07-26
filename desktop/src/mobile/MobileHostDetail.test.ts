import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelEntry, ServerStatus } from "../lib/api/types";
import type { QueueEntry } from "../stores/jobs";
import type { MobileHost } from "./hosts";

interface SseCall {
  path: string;
  options: {
    target: { baseUrl: string; apiKey: string | null };
    signal: AbortSignal;
    retry?: boolean;
    onEvent: (event: string, data: string) => void;
  };
}

const { apiJsonTo, unloadModel, sseCalls } = vi.hoisted(() => ({
  apiJsonTo: vi.fn(),
  unloadModel: vi.fn(),
  sseCalls: [] as SseCall[],
}));

vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJsonTo,
}));

vi.mock("../lib/api/sse", () => ({
  sseStream: (path: string, options: SseCall["options"]) => {
    sseCalls.push({ path, options });
    return new Promise<void>(() => {});
  },
}));

vi.mock("../lib/api/models", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/models")>()),
  unloadModel,
}));

import MobileHostDetail from "./MobileHostDetail.vue";

const studio: MobileHost = {
  id: "studio-id",
  name: "Studio",
  baseUrl: "http://studio.tailnet.ts.net:7680",
  apiKey: "studio-secret",
  hostname: "studio",
  version: "0.18.0",
  online: true,
};

const renderBox: MobileHost = {
  id: "render-id",
  name: "Render Box",
  baseUrl: "https://render.example.com:8443",
  apiKey: "render-secret",
  hostname: "render",
  version: "0.19.0",
  online: true,
};

const studioTarget = { baseUrl: studio.baseUrl, apiKey: studio.apiKey };
const renderTarget = { baseUrl: renderBox.baseUrl, apiKey: renderBox.apiKey };

function serverStatus(overrides: Partial<ServerStatus> = {}): ServerStatus {
  return {
    version: "0.18.0",
    models_loaded: ["flux-dev:q8"],
    uptime_secs: 3_661,
    hostname: "studio",
    gpu_info: {
      name: "NVIDIA GeForce RTX 4090",
      backend: "cuda",
      vram_total_mb: 24_000,
      vram_used_mb: 6_000,
    },
    queue_depth: 2,
    queue_capacity: 8,
    models_disk: { total_bytes: 2_000_000_000_000, free_bytes: 500_000_000_000 },
    ...overrides,
  };
}

function model(name: string, family: string, overrides: Partial<ModelEntry> = {}): ModelEntry {
  return {
    name,
    family,
    size_gb: 12,
    is_loaded: false,
    hf_repo: `example/${name}`,
    default_steps: 30,
    default_guidance: 4,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
    ...overrides,
  };
}

const queueEntries: QueueEntry[] = [
  {
    id: "job-running",
    model: "flux-dev:q8",
    state: "running",
    started_at_unix_ms: Date.now() - 5_000,
    position: 0,
    gpu: 0,
  },
  {
    id: "job-queued",
    model: "z-image:q8",
    state: "queued",
    started_at_unix_ms: Date.now(),
    position: 2,
  },
];

function installApi(): void {
  apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string): Promise<unknown> => {
    if (path === "/api/status") {
      return Promise.resolve(
        target.baseUrl === renderBox.baseUrl
          ? serverStatus({
              version: "0.19.0",
              hostname: "render",
              models_loaded: ["qwen-image:bf16"],
            })
          : serverStatus(),
      );
    }
    if (path === "/api/models") {
      return Promise.resolve(
        target.baseUrl === renderBox.baseUrl
          ? [model("qwen-image:bf16", "qwen-image")]
          : [
              model("flux-dev:q8", "flux", { is_loaded: true }),
              model("z-image:q8", "z-image"),
              model("not-installed", "flux", { downloaded: false }),
            ],
      );
    }
    if (path === "/api/queue") {
      return Promise.resolve({ entries: target.baseUrl === renderBox.baseUrl ? [] : queueEntries });
    }
    return Promise.reject(new Error(`Unexpected API path: ${path}`));
  });
}

function stream(path: string, target = studioTarget): SseCall {
  const call = [...sseCalls]
    .reverse()
    .find(
      (candidate) => candidate.path === path && candidate.options.target.baseUrl === target.baseUrl,
    );
  if (!call) throw new Error(`Missing ${path} stream for ${target.baseUrl}`);
  return call;
}

function buttonWithText(wrapper: VueWrapper, text: string) {
  const button = wrapper.findAll("button").find((candidate) => candidate.text() === text);
  if (!button) throw new Error(`Missing ${text} button`);
  return button;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

let wrapper: VueWrapper | null = null;

async function mountDetail(host: MobileHost = studio, active = false): Promise<VueWrapper> {
  wrapper = mount(MobileHostDetail, { props: { host, active } });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  apiJsonTo.mockReset();
  unloadModel.mockReset().mockResolvedValue(undefined);
  sseCalls.length = 0;
  installApi();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

describe("MobileHostDetail remote host data", () => {
  it("targets the exact remote and renders telemetry, queue, downloads, and installed models", async () => {
    const view = await mountDetail();

    expect(apiJsonTo).toHaveBeenCalledWith(studioTarget, "/api/status");
    expect(apiJsonTo).toHaveBeenCalledWith(studioTarget, "/api/models");
    expect(apiJsonTo).toHaveBeenCalledWith(studioTarget, "/api/queue");
    expect(stream("/api/resources/stream").options).toMatchObject({
      target: studioTarget,
      retry: true,
    });
    expect(stream("/api/downloads/stream").options).toMatchObject({
      target: studioTarget,
      retry: true,
    });

    expect(view.text()).toContain("NVIDIA GeForce RTX 4090");
    expect(view.text()).toContain("6.0 GB/24.0 GB");
    expect(view.text()).toContain("500.0 GB free");
    expect(view.text()).toContain("UP 1h 1m");

    const queueRows = view.get("[data-test='host-detail-queue']").findAll("li");
    expect(queueRows).toHaveLength(2);
    expect(queueRows[0]!.text()).toContain("flux-dev:q8RUNNING · GPU 0");
    expect(queueRows[1]!.text()).toContain("z-image:q8QUEUED #2");

    const modelRows = view.get("[data-test='host-detail-models']").findAll("li");
    expect(modelRows).toHaveLength(2);
    expect(modelRows[0]!.text()).toContain("flux-dev:q8");
    expect(modelRows[0]!.text()).toContain("LOADED");
    expect(view.text()).not.toContain("not-installed");

    stream("/api/resources/stream").options.onEvent(
      "snapshot",
      JSON.stringify({
        hostname: "studio",
        timestamp: 1,
        gpus: [
          {
            ordinal: 0,
            name: "NVIDIA RTX 6000 Ada",
            backend: "cuda",
            vram_total: 48_000_000_000,
            vram_used: 18_000_000_000,
            gpu_utilization: 93,
          },
        ],
        system_ram: {
          total: 128_000_000_000,
          used: 32_000_000_000,
          used_by_mold: 16_000_000_000,
          used_by_other: 16_000_000_000,
        },
        cpu: { cores: 32, usage_percent: 43.2 },
      }),
    );
    stream("/api/downloads/stream").options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: {
          active_jobs: [
            {
              id: "download-1",
              model: "qwen-image:bf16",
              status: "active",
              files_done: 2,
              files_total: 4,
              bytes_done: 250,
              bytes_total: 1_000,
            },
          ],
          queued: [],
          history: [],
        },
      }),
    );
    await flushPromises();

    expect(view.text()).toContain("NVIDIA RTX 6000 Ada");
    expect(view.text()).toContain("18.0 GB/48.0 GB");
    expect(view.text()).toContain("43% · 32 cores");
    expect(view.text()).toContain("32.0 GB/128.0 GB");
    expect(view.text()).toContain("qwen-image:bf16");
    expect(view.text()).toContain("2/4 files");
    expect(
      view
        .get("[role='meter'][aria-label='VRAM usage for NVIDIA RTX 6000 Ada']")
        .attributes("aria-valuenow"),
    ).toBe("38");
    expect(view.get("[role='meter'][aria-label='CPU usage']").attributes("aria-valuenow")).toBe(
      "43",
    );
    expect(view.get("[role='meter'][aria-label='RAM usage']").attributes("aria-valuenow")).toBe(
      "25",
    );
    expect(
      view.get("[role='meter'][aria-label='Models disk usage']").attributes("aria-valuenow"),
    ).toBe("75");
    expect(
      view
        .get("[role='meter'][aria-label='Download progress for qwen-image:bf16']")
        .attributes("aria-valuenow"),
    ).toBe("25");
    expect(view.emitted("status")).toEqual([[{ id: studio.id, status: serverStatus() }]]);
  });

  it("renders every status GPU before the resource stream produces a snapshot", async () => {
    apiJsonTo.mockImplementation(
      (target: { baseUrl: string }, path: string): Promise<unknown> => {
        if (path === "/api/status") {
          return Promise.resolve(
            serverStatus({
              gpus: [
                {
                  ordinal: 0,
                  name: "NVIDIA RTX 3090",
                  vram_total_bytes: 24_000_000_000,
                  vram_used_bytes: 8_000_000_000,
                  state: "generating",
                },
                {
                  ordinal: 1,
                  name: "NVIDIA B200",
                  vram_total_bytes: 80_000_000_000,
                  vram_used_bytes: 20_000_000_000,
                  state: "idle",
                },
              ],
            }),
          );
        }
        if (path === "/api/models") return Promise.resolve([]);
        if (path === "/api/queue") return Promise.resolve({ entries: [] });
        return Promise.reject(
          new Error(`Unexpected API path: ${path} for ${target.baseUrl}`),
        );
      },
    );

    const view = await mountDetail();

    expect(view.text()).toContain("NVIDIA RTX 3090");
    expect(view.text()).toContain("NVIDIA B200");
    expect(view.text()).toContain("8.0 GB/24.0 GB");
    expect(view.text()).toContain("20.0 GB/80.0 GB");
  });

  it("uses the live queue count after the queue API responds", async () => {
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/status") return Promise.resolve(serverStatus({ queue_depth: 7 }));
      if (path === "/api/models") return Promise.resolve([]);
      if (path === "/api/queue") return Promise.resolve({ entries: [queueEntries[0]] });
      return Promise.reject(new Error(`Unexpected API path: ${path} for ${target.baseUrl}`));
    });

    const view = await mountDetail();

    expect(view.get("[aria-labelledby='host-queue-title'] .mobile-section-head span").text()).toBe(
      "1/8",
    );
  });

  it("falls back to the status queue depth when the queue API is unavailable", async () => {
    apiJsonTo.mockImplementation((_target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/status") return Promise.resolve(serverStatus({ queue_depth: 7 }));
      if (path === "/api/models") return Promise.resolve([]);
      if (path === "/api/queue") return Promise.reject(new Error("queue unsupported"));
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    const view = await mountDetail();

    expect(view.get("[aria-labelledby='host-queue-title'] .mobile-section-head span").text()).toBe(
      "7/8",
    );
  });

  it("recovers from a transient initial failure when retrying the same host", async () => {
    let statusAttempts = 0;
    apiJsonTo.mockImplementation((_target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/status") {
        statusAttempts += 1;
        return statusAttempts === 1
          ? Promise.reject(new Error("temporary network failure"))
          : Promise.resolve(serverStatus());
      }
      if (path === "/api/models") {
        return Promise.resolve([model("flux-dev:q8", "flux", { is_loaded: true })]);
      }
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    const view = await mountDetail();
    expect(view.get("[role='alert']").text()).toContain("temporary network failure");
    expect(sseCalls).toHaveLength(0);

    await view.get("[data-test='host-detail-retry']").trigger("click");
    await flushPromises();

    expect(statusAttempts).toBe(2);
    expect(view.find("[data-test='host-detail-retry']").exists()).toBe(false);
    expect(view.text()).toContain("flux-dev:q8");
    expect(sseCalls).toHaveLength(2);
    expect(view.emitted("status")).toEqual([
      [{ id: studio.id, status: null }],
      [{ id: studio.id, status: serverStatus() }],
    ]);
  });

  it("selects an online host and reflects the active state", async () => {
    const view = await mountDetail();
    const select = view.get("[data-test='host-detail-select']");

    await select.trigger("click");
    expect(view.emitted("select")).toEqual([[studio.id]]);

    await view.setProps({ active: true });
    expect(select.text()).toBe("Used for generations");
    expect(select.attributes()).toHaveProperty("disabled");
  });

  it("renames, confirms forget on the second tap, and opens Catalog", async () => {
    const view = await mountDetail();

    await buttonWithText(view, "Rename").trigger("click");
    await view.get(".mobile-inline-form input").setValue("  Main Studio  ");
    await view.get(".mobile-inline-form").trigger("submit");
    expect(view.emitted("rename")).toEqual([[{ id: studio.id, name: "Main Studio" }]]);

    const forget = view.get("[data-test='host-detail-forget']");
    await forget.trigger("click");
    expect(view.emitted("forget")).toBeUndefined();
    expect(forget.text()).toBe("Forget Studio?");
    await forget.trigger("click");
    expect(view.emitted("forget")).toEqual([[studio.id]]);

    await view.get("[data-test='host-detail-catalog']").trigger("click");
    expect(view.emitted("catalog")).toEqual([[studio.id]]);
  });

  it("unloads a resident model on the exact remote and updates the loaded list", async () => {
    const pending = deferred<void>();
    unloadModel.mockReturnValueOnce(pending.promise);
    const view = await mountDetail();
    const unload = view.get("[aria-label='Unload flux-dev:q8']");

    await unload.trigger("click");
    expect(unloadModel).toHaveBeenCalledWith("flux-dev:q8", studioTarget);
    expect(unload.attributes()).toHaveProperty("disabled");
    expect(unload.text()).toBe("…");

    pending.resolve();
    await flushPromises();
    expect(view.find("[aria-label='Unload flux-dev:q8']").exists()).toBe(false);
    expect(view.text()).toContain("No models are loaded on the GPU.");
    expect(view.get("[data-test='host-detail-models']").text()).not.toContain("LOADED");
    expect(view.emitted("status")?.at(-1)).toEqual([
      { id: studio.id, status: serverStatus({ models_loaded: [] }) },
    ]);
  });
});

describe("MobileHostDetail host switching", () => {
  it("aborts old streams, ignores their stale frames, and retargets every live service", async () => {
    const view = await mountDetail();
    const oldResources = stream("/api/resources/stream");
    const oldDownloads = stream("/api/downloads/stream");

    await view.setProps({ host: renderBox });
    await flushPromises();

    expect(oldResources.options.signal.aborted).toBe(true);
    expect(oldDownloads.options.signal.aborted).toBe(true);
    expect(apiJsonTo).toHaveBeenCalledWith(renderTarget, "/api/status");
    expect(apiJsonTo).toHaveBeenCalledWith(renderTarget, "/api/models");
    expect(apiJsonTo).toHaveBeenCalledWith(renderTarget, "/api/queue");
    expect(stream("/api/resources/stream", renderTarget).options.signal.aborted).toBe(false);
    expect(stream("/api/downloads/stream", renderTarget).options.signal.aborted).toBe(false);

    oldResources.options.onEvent(
      "snapshot",
      JSON.stringify({
        hostname: "studio",
        timestamp: 2,
        gpus: [
          {
            ordinal: 0,
            name: "STALE STUDIO GPU",
            backend: "cuda",
            vram_total: 1_000,
            vram_used: 1_000,
          },
        ],
        system_ram: {
          total: 1_000,
          used: 1_000,
          used_by_mold: 1_000,
          used_by_other: 0,
        },
      }),
    );
    await flushPromises();

    expect(view.text()).toContain("Render Box");
    expect(view.text()).toContain("qwen-image:bf16");
    expect(view.text()).not.toContain("STALE STUDIO GPU");

    view.unmount();
    wrapper = null;
    expect(stream("/api/resources/stream", renderTarget).options.signal.aborted).toBe(true);
    expect(stream("/api/downloads/stream", renderTarget).options.signal.aborted).toBe(true);
  });

  it("ignores a late status response from the previously selected host", async () => {
    const oldStatus = deferred<ServerStatus>();
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/status") {
        return target.baseUrl === studio.baseUrl
          ? oldStatus.promise
          : Promise.resolve(
              serverStatus({
                version: "0.19.0",
                hostname: "render",
                models_loaded: ["qwen-image:bf16"],
              }),
            );
      }
      if (path === "/api/models") {
        return Promise.resolve(
          target.baseUrl === studio.baseUrl
            ? [model("stale-model", "flux")]
            : [model("qwen-image:bf16", "qwen-image")],
        );
      }
      if (path === "/api/queue") return Promise.resolve({ entries: [] });
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    wrapper = mount(MobileHostDetail, { props: { host: studio, active: false } });
    await flushPromises();
    await wrapper.setProps({ host: renderBox });
    await flushPromises();

    expect(wrapper.text()).toContain("Render Box");
    expect(wrapper.text()).toContain("qwen-image:bf16");
    expect(wrapper.emitted("status")).toEqual([
      [
        {
          id: renderBox.id,
          status: serverStatus({
            version: "0.19.0",
            hostname: "render",
            models_loaded: ["qwen-image:bf16"],
          }),
        },
      ],
    ]);

    oldStatus.resolve(
      serverStatus({
        version: "0.17.0",
        hostname: "old-studio",
        models_loaded: ["stale-model"],
      }),
    );
    await flushPromises();

    expect(wrapper.text()).not.toContain("stale-model");
    expect(wrapper.emitted("status")).toHaveLength(1);
  });
});
