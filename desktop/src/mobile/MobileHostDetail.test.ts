import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../lib/api/client";
import { authenticatedMiniMaxH3Capabilities } from "@studio/lib/minimaxH3Inventory.testFixtures";
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

const { apiJsonTo, setDeviceEnabled, unloadModel, sseCalls } = vi.hoisted(() => ({
  apiJsonTo: vi.fn(),
  setDeviceEnabled: vi.fn(),
  unloadModel: vi.fn(),
  sseCalls: [] as SseCall[],
}));

vi.mock("@studio/api/devices", () => ({
  setDeviceEnabled,
  parseDeviceListResponse: (value: { devices: unknown[]; plan_version: number }) => value,
}));

vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJsonTo,
}));

vi.mock("@studio/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/client")>()),
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
    if (path === "/api/queue?limit=8") {
      return Promise.resolve({ entries: target.baseUrl === renderBox.baseUrl ? [] : queueEntries });
    }
    if (path === "/api/devices") {
      return Promise.resolve({ devices: [], plan_version: 0 });
    }
    if (path === "/api/capabilities") {
      return Promise.resolve({
        events: { available: true },
        devices: { available: true, lifecycle: true, restart_enable: false },
        dispatch: { active_mode: "v2", v2_authoritative: true },
      });
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

function raceDevice(name: string) {
  return {
    id: `cuda:${name}`,
    backend: "cuda",
    ordinal: 0,
    device_kind: "full_gpu",
    nvml_uuid: name,
    physical_uuid: name,
    mig_uuid: null,
    mig_parent_uuid: null,
    mig_profile: null,
    name,
    pci_bus_id: null,
    compute_capability: "8.6",
    memory: {
      total_bytes: 24_000_000_000,
      used_bytes: 0,
      mold_used_bytes: 0,
      other_used_bytes: 0,
    },
    telemetry: {
      utilization_percent: 0,
      temperature_c: 30,
      power_w: 20,
    },
    desired_enabled: true,
    restart_required: false,
    admin_state: "enabled",
    health: "healthy",
    activity: "idle",
    schedulable: true,
    unschedulable_reason: null,
    loaded_models: [],
    active_work_id: null,
    planned_work_ids: [],
  };
}

let wrapper: VueWrapper | null = null;

async function mountDetail(host: MobileHost = studio, active = false): Promise<VueWrapper> {
  wrapper = mount(MobileHostDetail, { props: { host, active } });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  apiJsonTo.mockReset();
  setDeviceEnabled.mockReset().mockResolvedValue(undefined);
  unloadModel.mockReset().mockResolvedValue(undefined);
  sseCalls.length = 0;
  installApi();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("MobileHostDetail remote host data", () => {
  it("shows dependency preparation component and progress on a queued row", async () => {
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target, path) => {
      if (path === "/api/queue?limit=8") {
        return Promise.resolve({
          entries: [queueEntries[1]],
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "settled",
            dirty_since_unix_ms: null,
            next_replan_at_unix_ms: null,
            work_items: [
              {
                work_id: "job-queued",
                parent_id: "job-queued",
                work_kind: "generation",
                priority_class: "user",
                queue_rank: 0,
                bypass_count: 0,
                estimate_confidence: "low",
                blocked_reason: "preparing",
                preparation_progress: {
                  component: "Verifying model files",
                  bytes_done: 27,
                  bytes_total: 100,
                },
              },
            ],
          },
        });
      }
      return originalApi(target, path);
    });

    const view = await mountDetail();

    expect(view.get("[data-test='host-detail-queue-row-job-queued']").text()).toContain(
      "PREPARING · VERIFYING MODEL FILES 27%",
    );
  });

  it("shows the host's runtime loading stage on a running queue row", async () => {
    const running = { ...queueEntries[1]!, state: "running" };
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target, path) => {
      if (path === "/api/queue?limit=8") {
        return Promise.resolve({
          entries: [running],
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "settled",
            dirty_since_unix_ms: null,
            next_replan_at_unix_ms: null,
            work_items: [
              {
                work_id: running.id,
                parent_id: running.id,
                work_kind: "generation",
                priority_class: "user",
                queue_rank: 0,
                bypass_count: 0,
                estimate_confidence: "low",
                activity_phase: "active",
                runtime_phase: "loading",
                runtime_stage: "Loading Flux.2 transformer",
                runtime_current: 17,
                runtime_total: 20,
              },
            ],
          },
        });
      }
      return originalApi(target, path);
    });

    const view = await mountDetail();
    const row = view.get("[data-test='host-detail-queue-row-job-queued']");
    expect(row.text()).toContain("LOADING FLUX.2 TRANSFORMER · 17/20");
    expect(row.text()).not.toContain("NEXT UP");
  });

  it("renders server information without waiting for model inventory", async () => {
    const pendingModels = deferred<ModelEntry[]>();
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target, path) =>
      path === "/api/models" ? pendingModels.promise : originalApi(target, path),
    );

    const view = await mountDetail();

    expect(view.text()).not.toContain("Reading host…");
    expect(view.text()).toContain("NVIDIA GeForce RTX 4090");
    expect(view.text()).toContain("500.0 GB free");
    expect(view.text()).toContain("Reading installed models…");
    expect(stream("/api/resources/stream").options.target).toEqual(studioTarget);

    pendingModels.resolve([model("flux-dev:q8", "flux")]);
    await flushPromises();

    expect(view.text()).toContain("flux-dev:q8");
    expect(view.text()).not.toContain("Reading installed models…");
  });

  it("shows active sequence stages instead of calling the machine queue empty", async () => {
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target, path) => {
      if (path === "/api/status") return Promise.resolve(serverStatus({ queue_depth: 0 }));
      if (path === "/api/queue?limit=8") {
        return Promise.resolve({
          entries: [],
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "optimized",
            dirty_since_unix_ms: null,
            next_replan_at_unix_ms: null,
            work_items: [
              {
                work_id: "chain:sequence-1:attempt:0:stage:0",
                parent_id: "sequence-1",
                work_kind: "chain_stage",
                chain_stage: 0,
                priority_class: "user",
                queue_rank: 4,
                bypass_count: 0,
                gpu: 2,
                planned_device_id: "cuda:gpu-2",
                planned_lane_kind: "device",
                lane_order: 0,
                estimate_confidence: "low",
                activity_phase: "active",
              },
            ],
          },
        });
      }
      return originalApi(target, path);
    });

    const view = await mountDetail();

    expect(view.get("[data-test='host-detail-queue-total']").text()).toBe("1 total");
    expect(view.get("[data-test='host-detail-queue-scope']").text()).toBe(
      "All 1 job loaded",
    );
    expect(view.get("[data-test='host-detail-planned-work']").text()).toContain(
      "activeChain stage · stage 1sequence-1GPU 2",
    );
    expect(view.text()).not.toContain("Queue is empty.");
  });

  it("shows a CPU utility lane when the host reports no GPUs", async () => {
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target, path) => {
      if (path === "/api/queue?limit=8") {
        return Promise.resolve({
          entries: [],
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "optimized",
            dirty_since_unix_ms: null,
            next_replan_at_unix_ms: null,
            work_items: [
              {
                work_id: "cpu-upscale",
                parent_id: "parent",
                work_kind: "post_upscale",
                priority_class: "user",
                queue_rank: 0,
                bypass_count: 0,
                planned_device_id: null,
                planned_lane_kind: "host_utility",
                lane_order: 0,
                estimate_confidence: "low",
                activity_phase: "cpu",
              },
            ],
          },
        });
      }
      return originalApi(target, path);
    });

    const view = await mountDetail();

    expect(view.get("[data-test='host-detail-devices']").text()).toContain("Host utility");
    expect(view.findAll('[data-test="device-card"]')).toHaveLength(0);
  });

  it("keeps a colliding future typed lane out of the mobile GPU lane", async () => {
    const visible = raceDevice("MOBILE DEVICE");
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target, path) => {
      if (path === "/api/devices") {
        return Promise.resolve({ devices: [visible], plan_version: 1 });
      }
      if (path === "/api/queue?limit=8") {
        return Promise.resolve({
          entries: [],
          plan: {
            plan_version: 1,
            state_version: 1,
            optimizer_state: "optimized",
            dirty_since_unix_ms: null,
            next_replan_at_unix_ms: null,
            work_items: [
              {
                work_id: "mobile-future-collision",
                parent_id: "parent",
                work_kind: "future_utility",
                priority_class: "user",
                queue_rank: 0,
                bypass_count: 0,
                planned_device_id: visible.id,
                planned_lane_kind: "future_accelerator_lane",
                lane_order: 0,
                estimate_confidence: "low",
              },
              {
                work_id: "mobile-typed-device",
                parent_id: "typed-parent",
                work_kind: "generation",
                priority_class: "user",
                queue_rank: 1,
                bypass_count: 0,
                planned_device_id: visible.id,
                planned_lane_kind: "device",
                lane_order: 1,
                estimate_confidence: "low",
              },
            ],
          },
        });
      }
      return originalApi(target, path);
    });

    const view = await mountDetail();

    expect(view.get('[data-test="other-compute-lane"]').text()).toContain("Future utility");
    expect(view.get('[data-test="device-lane"]').text()).toContain("Generation");
  });

  it("targets the exact remote and renders telemetry, queue, downloads, and installed models", async () => {
    const view = await mountDetail();

    expect(apiJsonTo).toHaveBeenCalledWith(studioTarget, "/api/status");
    expect(apiJsonTo).toHaveBeenCalledWith(studioTarget, "/api/models");
    expect(apiJsonTo).toHaveBeenCalledWith(studioTarget, "/api/queue?limit=8");
    expect(stream("/api/resources/stream").options).toMatchObject({
      target: studioTarget,
      retry: true,
    });
    expect(stream("/api/downloads/stream").options).toMatchObject({
      target: studioTarget,
      retry: true,
    });
    expect(stream("/api/events").options).toMatchObject({
      target: studioTarget,
      retry: true,
    });

    expect(view.text()).toContain("NVIDIA GeForce RTX 4090");
    expect(view.text()).toContain("6.0 GB/24.0 GB");
    expect(view.text()).toContain("500.0 GB free");
    expect(view.text()).toContain("UP 1h 1m");

    const queueRows = view.get("[data-test='host-detail-queue']").findAll("li");
    expect(queueRows).toHaveLength(2);
    expect(queueRows[0]!.text()).toContain("z-image:q8StudioQUEUED #2");
    expect(queueRows[1]!.text()).toContain("flux-dev:q8StudioRUNNING · GPU 0");

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

    stream("/api/downloads/stream").options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: {
          active_jobs: [
            {
              id: "download-2",
              model: "ltx-2-19b-distilled:fp8",
              status: "active",
              files_done: 0,
              files_total: 0,
              bytes_done: 0,
              bytes_total: 0,
              current_file: "Verifying file [1/19] tokenizer.json...",
            },
          ],
          queued: [],
          history: [],
        },
      }),
    );
    await flushPromises();
    expect(view.text()).toContain("Preparing");
    expect(view.text()).toContain("Verifying file [1/19]");
    expect(view.text()).not.toContain("0/0 files");
    expect(view.emitted("status")).toEqual([[{ id: studio.id, status: serverStatus() }]]);
  });

  it("renders the machine queue newest-first", async () => {
    const view = await mountDetail();
    expect(
      view
        .findAll("[data-test^='host-detail-queue-row-']")
        .map((row) => row.get(".mobile-generation-job-copy p").text()),
    ).toEqual(["z-image:q8", "flux-dev:q8"]);
  });

  it("confirms and cancels only a queued job on the exact authenticated host", async () => {
    let currentQueue = [...queueEntries];
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/queue?limit=8") return Promise.resolve({ entries: currentQueue });
      return originalApi(requestTarget, path);
    });
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        expect(String(input)).toBe(`${studio.baseUrl}/api/queue/job-queued`);
        expect(init?.method).toBe("DELETE");
        expect((init?.headers as Headers).get("x-api-key")).toBe(studio.apiKey);
        currentQueue = currentQueue.filter((entry) => entry.id !== "job-queued");
        return new Response(null, { status: 204 });
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    const view = await mountDetail();

    // A running row on a host without cooperative cancellation offers nothing.
    expect(
      view
        .find("[data-test='host-detail-queue-row-job-running'] [data-test='swipe-action-cancel']")
        .exists(),
    ).toBe(false);
    const row = view.get("[data-test='host-detail-queue-row-job-queued']");
    const reveal = row.get("[data-test='swipe-row-actions']");
    expect(reveal.attributes("aria-label")).toBe("Actions for z-image:q8 job");

    // Revealing the tray is step one — nothing is cancelled by the gesture.
    await reveal.trigger("click");
    expect(fetchMock).not.toHaveBeenCalled();

    await row.get("[data-test='swipe-action-cancel']").trigger("click");
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(view.get("[data-test='host-detail-queue']").text()).not.toContain("z-image:q8");
    expect(view.get("[data-test='host-detail-queue']").text()).toContain("flux-dev:q8");
  });

  it("pauses one queued machine row without pausing its sibling or host", async () => {
    let currentQueue: QueueEntry[] = [
      { ...queueEntries[0]!, id: "job-held", state: "held", held_reason: "host memory" },
      { ...queueEntries[1]!, position: 0 },
    ];
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          devices: { available: true, lifecycle: true },
          dispatch: { v2_authoritative: true },
          queue: { can_pause: true, can_pause_job: true },
        });
      }
      if (path === "/api/queue?limit=8") return Promise.resolve({ entries: currentQueue });
      return originalApi(requestTarget, path);
    });
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/queue/job-held") && init?.method === "DELETE") {
        currentQueue = currentQueue.filter((entry) => entry.id !== "job-held");
        return new Response(null, { status: 204 });
      }
      if (url.endsWith("/api/queue/job-queued/pause") && init?.method === "POST") {
        currentQueue = currentQueue.map((entry) =>
          entry.id === "job-queued" ? { ...entry, state: "paused" } : entry,
        );
        return new Response(null, { status: 204 });
      }
      return Response.json({ paused: url.endsWith("/pause") });
    });
    vi.stubGlobal("fetch", fetchMock);
    const view = await mountDetail();
    const held = view.get("[data-test='host-detail-queue-row-job-held']");
    const queued = view.get("[data-test='host-detail-queue-row-job-queued']");

    expect(view.get("[data-test='host-detail-queue-pause']").text()).toBe("Pause");
    expect(held.find("[data-test='swipe-action-job-pause']").exists()).toBe(false);
    expect(queued.get("[data-test='swipe-action-job-pause']").text()).toBe("Pause");
    expect(held.get("[data-test='swipe-action-cancel']").text()).toBe("Cancel");

    await queued.get("[data-test='swipe-action-job-pause']").trigger("click");
    await flushPromises();
    expect(fetchMock).toHaveBeenCalledWith(
      `${studio.baseUrl}/api/queue/job-queued/pause`,
      expect.objectContaining({ method: "POST" }),
    );
    expect(held.text()).toContain("HELD");
    expect(view.get("[data-test='host-detail-queue-row-job-queued']").text()).toContain("PAUSED");
    expect(view.find("[data-test='host-detail-queue-paused']").exists()).toBe(false);
    expect(view.get("[data-test='host-detail-queue-pause']").text()).toBe("Pause");

    await held.get("[data-test='swipe-action-cancel']").trigger("click");
    await flushPromises();
    expect(fetchMock).toHaveBeenCalledWith(
      `${studio.baseUrl}/api/queue/job-held`,
      expect.objectContaining({ method: "DELETE" }),
    );
    expect(view.find("[data-test='host-detail-queue-row-job-held']").exists()).toBe(false);
  });

  it("keeps ordinary queued work waiting when only another row is restart-paused", async () => {
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          devices: { available: true, lifecycle: true },
          dispatch: { v2_authoritative: true },
          queue: { can_pause: true, can_pause_job: true },
        });
      }
      if (path === "/api/queue?limit=8") {
        return Promise.resolve({
          entries: [
            { ...queueEntries[0]!, id: "job-paused", state: "paused", position: 0 },
            { ...queueEntries[1]!, position: 1 },
          ],
        });
      }
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();

    expect(view.get("[data-test='host-detail-queue-row-job-paused']").text()).toContain("PAUSED");
    expect(view.get("[data-test='host-detail-queue-row-job-queued']").text()).toContain(
      "QUEUED #1",
    );
    expect(view.get("[data-test='host-detail-queue-row-job-queued']").text()).not.toContain(
      "PAUSED",
    );
    expect(view.find("[data-test='host-detail-queue-paused']").exists()).toBe(false);
    expect(view.get("[data-test='host-detail-queue-pause']").text()).toBe("Pause");
    expect(
      view
        .get("[data-test='host-detail-queue-row-job-paused']")
        .get("[data-test='swipe-action-job-resume']")
        .text(),
    ).toBe("Resume");
  });

  it("refreshes host-wide pause state changed from another surface", async () => {
    vi.useFakeTimers();
    let queuePaused = false;
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/status") {
        return Promise.resolve(serverStatus({ queue_paused: queuePaused }));
      }
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          devices: { available: true, lifecycle: true },
          dispatch: { v2_authoritative: true },
          queue: { can_pause: true },
        });
      }
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();
    expect(view.get("[data-test='host-detail-queue-pause']").text()).toBe("Pause");

    queuePaused = true;
    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();
    expect(view.get("[data-test='host-detail-queue-row-job-queued']").text()).toContain("PAUSED");
    expect(view.get("[data-test='host-detail-queue-pause']").text()).toBe("Resume");

    queuePaused = false;
    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();
    expect(view.get("[data-test='host-detail-queue-row-job-queued']").text()).toContain(
      "QUEUED #2",
    );
    expect(view.get("[data-test='host-detail-queue-pause']").text()).toBe("Pause");
  });

  it("does not let an older status poll overwrite a queue pause action", async () => {
    vi.useFakeTimers();
    const staleStatus = deferred<ServerStatus>();
    let statusCalls = 0;
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/status") {
        statusCalls += 1;
        return statusCalls === 2
          ? staleStatus.promise
          : Promise.resolve(serverStatus({ queue_paused: false }));
      }
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          devices: { available: true, lifecycle: true },
          dispatch: { v2_authoritative: true },
          queue: { can_pause: true },
        });
      }
      return originalApi(requestTarget, path);
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(Response.json({ paused: true }))),
    );

    const view = await mountDetail();
    vi.advanceTimersByTime(5_000);
    await flushPromises();

    await view.get("[data-test='host-detail-queue-pause']").trigger("click");
    await flushPromises();
    expect(view.get("[data-test='host-detail-queue-pause']").text()).toBe("Resume");

    staleStatus.resolve(serverStatus({ queue_paused: false }));
    await flushPromises();
    expect(view.get("[data-test='host-detail-queue-row-job-queued']").text()).toContain("PAUSED");
    expect(view.get("[data-test='host-detail-queue-pause']").text()).toBe("Resume");
  });

  it("confirms and cancels a running job when the host advertises cooperative support", async () => {
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          devices: { available: true, lifecycle: true },
          dispatch: { v2_authoritative: true },
          queue: { cooperative_cancellation: true },
        });
      }
      return originalApi(requestTarget, path);
    });
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const view = await mountDetail();
    const row = view.get("[data-test='host-detail-queue-row-job-running']");

    await row.get("[data-test='swipe-row-actions']").trigger("click");
    expect(fetchMock).not.toHaveBeenCalled();
    await row.get("[data-test='swipe-action-cancel']").trigger("click");
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledWith(
      `${studio.baseUrl}/api/queue/job-running`,
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("keeps a queued job visible and reports the host refusal when cancellation fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json(
          { error: "queue job job-queued is already running; only queued jobs can be cancelled" },
          { status: 409, statusText: "Conflict" },
        ),
      ),
    );
    const view = await mountDetail();
    const row = view.get("[data-test='host-detail-queue-row-job-queued']");

    await row.get("[data-test='swipe-row-actions']").trigger("click");
    await row.get("[data-test='swipe-action-cancel']").trigger("click");
    await flushPromises();

    expect(view.text()).toContain("z-image:q8");
    expect(view.get("[role='alert']").text()).toContain(
      "queue job job-queued is already running; only queued jobs can be cancelled",
    );
    // The row survives the refusal and is actionable again.
    expect(row.find("[data-test='swipe-action-cancel']").exists()).toBe(true);
  });

  it("commits the cancel on a second full swipe from the revealed tray, never on one gesture", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const view = await mountDetail();
    const row = view.get("[data-test='host-detail-queue-row-job-queued']");
    vi.spyOn(row.element, "getBoundingClientRect").mockReturnValue({
      width: 390,
    } as DOMRect);
    const surface = row.get("div:last-child");
    const fullSwipe = async () => {
      await surface.trigger("pointerdown", {
        pointerId: 1,
        pointerType: "touch",
        clientX: 380,
        clientY: 0,
      });
      await surface.trigger("pointermove", {
        pointerId: 1,
        pointerType: "touch",
        clientX: 60,
        clientY: 0,
      });
      await surface.trigger("pointerup", {
        pointerId: 1,
        pointerType: "touch",
        clientX: 60,
        clientY: 0,
      });
      await flushPromises();
    };

    // Step one: a full swipe from a closed row only reveals the tray.
    await fullSwipe();
    expect(fetchMock).not.toHaveBeenCalledWith(
      `${studio.baseUrl}/api/queue/job-queued`,
      expect.objectContaining({ method: "DELETE" }),
    );

    // Step two: the same gesture from the revealed tray commits.
    await fullSwipe();
    expect(fetchMock).toHaveBeenCalledWith(
      `${studio.baseUrl}/api/queue/job-queued`,
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("offers Move to back only where the host advertises reordering", async () => {
    const originalApi = apiJsonTo.getMockImplementation()!;
    const patched: RequestInit[] = [];
    apiJsonTo.mockImplementation((requestTarget, path, init) => {
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          devices: { available: true, lifecycle: true },
          dispatch: { v2_authoritative: true },
          queue: { can_reorder: true },
        });
      }
      if (path === "/api/queue/job-queued") {
        patched.push(init as RequestInit);
        return Promise.resolve({
          id: "job-queued",
          model: "z-image:q8",
          state: "queued",
          started_at_unix_ms: 1,
          position: 9,
        });
      }
      return originalApi(requestTarget, path, init);
    });
    const view = await mountDetail();

    // Reordering is queued-only; a running row never offers it.
    expect(
      view
        .find("[data-test='host-detail-queue-row-job-running'] [data-test='swipe-action-back']")
        .exists(),
    ).toBe(false);
    const row = view.get("[data-test='host-detail-queue-row-job-queued']");
    await row.get("[data-test='swipe-row-actions']").trigger("click");
    await row.get("[data-test='swipe-action-back']").trigger("click");
    await flushPromises();

    expect(patched).toHaveLength(1);
    expect(patched[0]!.method).toBe("PATCH");
    // The server clamps a large index to the tail, so no depth read is needed.
    expect(JSON.parse(String(patched[0]!.body)).position).toBe(Number.MAX_SAFE_INTEGER);
  });

  it("opens one queue row in a detail sheet with the shared vocabulary", async () => {
    const view = await mountDetail();
    expect(view.find("[data-test='queue-entry-detail']").exists()).toBe(false);

    await view.get("[data-test='host-detail-queue-open-job-queued']").trigger("click");
    await flushPromises();

    const sheet = view.get("[data-test='queue-entry-detail']");
    expect(sheet.get("[data-test='queue-detail-facts']").text()).toContain("#2 in line");
    // The gap is the wire's, not an old host's.
    expect(sheet.text()).not.toMatch(/upgrade/i);
    expect(sheet.get("[data-test='queue-detail-settings-notice']").text()).toMatch(
      /once this machine loads the job/i,
    );
  });

  it("arms the sheet's cancel before it touches the host", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const view = await mountDetail();
    await view.get("[data-test='host-detail-queue-open-job-queued']").trigger("click");
    await flushPromises();

    const cancel = view.get("[data-test='queue-detail-cancel']");
    await cancel.trigger("click");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(cancel.text()).toBe("Cancel job?");

    await cancel.trigger("click");
    await flushPromises();
    expect(fetchMock).toHaveBeenCalledWith(
      `${studio.baseUrl}/api/queue/job-queued`,
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("ignores older same-host queue and device refetches after a newer event refresh", async () => {
    const view = await mountDetail();
    const olderQueue = deferred<unknown>();
    const newerQueue = deferred<unknown>();
    const olderDevices = deferred<unknown>();
    const newerDevices = deferred<unknown>();
    let queueCall = 0;
    let deviceCall = 0;
    const existingApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/queue?limit=8")
        return queueCall++ === 0 ? olderQueue.promise : newerQueue.promise;
      if (path === "/api/devices")
        return deviceCall++ === 0 ? olderDevices.promise : newerDevices.promise;
      return existingApi(requestTarget, path);
    });
    const deviceEvents = stream("/api/events");
    const invalidate = () =>
      deviceEvents.options.onEvent("message", JSON.stringify({ type: "queue_plan_changed" }));
    invalidate();
    invalidate();

    newerQueue.resolve({
      entries: [
        {
          id: "newer-job",
          model: "newer-model",
          state: "queued",
          started_at_unix_ms: 2,
          position: 0,
        },
      ],
    });
    newerDevices.resolve({
      devices: [raceDevice("NEW DEVICE")],
      plan_version: 3,
    });
    await flushPromises();
    olderQueue.resolve({
      entries: [
        {
          id: "older-job",
          model: "older-model",
          state: "queued",
          started_at_unix_ms: 1,
          position: 0,
        },
      ],
    });
    olderDevices.resolve({
      devices: [raceDevice("OLD DEVICE")],
      plan_version: 2,
    });
    await flushPromises();

    expect(view.text()).toContain("newer-model");
    expect(view.text()).not.toContain("older-model");
    expect(view.text()).toContain("NEW DEVICE");
    expect(view.text()).not.toContain("OLD DEVICE");
  });

  it("polls through a failed event bootstrap, preserves the last device snapshot, and clears its warning after recovery", async () => {
    vi.useFakeTimers();
    let deviceCalls = 0;
    let capabilityCalls = 0;
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        if (deviceCalls === 2) {
          return Promise.reject(new ApiError("device inventory temporarily unavailable", 503));
        }
        return Promise.resolve({
          devices: [raceDevice(deviceCalls === 1 ? "LAST GOOD DEVICE" : "RECOVERED DEVICE")],
          plan_version: deviceCalls,
        });
      }
      if (path === "/api/capabilities") {
        capabilityCalls += 1;
        if (capabilityCalls === 2) {
          return Promise.reject(new TypeError("Failed to fetch"));
        }
      }
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();
    expect(view.text()).toContain("LAST GOOD DEVICE");
    expect(sseCalls.some((call) => call.path === "/api/events")).toBe(false);

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();

    const alert = view.get("[data-test='host-detail-device-error']");
    expect(alert.text()).toContain("Couldn’t refresh compute devices");
    expect(alert.text()).toContain("device inventory temporarily unavailable");
    expect(alert.text()).toContain("try again automatically");
    expect(view.text()).toContain("LAST GOOD DEVICE");

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();

    expect(deviceCalls).toBe(3);
    expect(view.find("[data-test='host-detail-device-error']").exists()).toBe(false);
    expect(view.text()).toContain("RECOVERED DEVICE");
    expect(view.text()).not.toContain("LAST GOOD DEVICE");
  });

  it("treats failed capabilities as uncertainty rather than proof of a legacy host", async () => {
    vi.useFakeTimers();
    let deviceCalls = 0;
    let capabilityCalls = 0;
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        return deviceCalls === 1
          ? Promise.reject(new ApiError("device inventory unavailable", 500))
          : Promise.resolve({
              devices: [raceDevice("RECOVERED AFTER CAPABILITIES")],
              plan_version: 2,
            });
      }
      if (path === "/api/capabilities") {
        capabilityCalls += 1;
        if (capabilityCalls <= 2) {
          return Promise.reject(new TypeError("Failed to fetch"));
        }
      }
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();
    expect(view.get("[data-test='host-detail-device-error']").text()).toContain(
      "device inventory unavailable",
    );

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();

    expect(view.find("[data-test='host-detail-device-error']").exists()).toBe(false);
    expect(view.text()).toContain("RECOVERED AFTER CAPABILITIES");
  });

  it("keeps a capability-confirmed legacy host quiet when the device endpoint is absent", async () => {
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/devices") {
        return Promise.reject(new ApiError("Not Found", 404));
      }
      if (path === "/api/capabilities") return Promise.resolve({});
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();

    expect(view.find("[data-test='host-detail-device-error']").exists()).toBe(false);
    expect(view.find("[data-test='host-detail-devices']").exists()).toBe(false);
  });

  it("keeps device telemetry but disables lifecycle mutations while capability authority is uncertain", async () => {
    vi.useFakeTimers();
    let deviceCalls = 0;
    let capabilityCalls = 0;
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        return Promise.resolve({
          devices: [raceDevice(deviceCalls === 1 ? "INITIAL DEVICE" : "CURRENT DEVICE")],
          plan_version: deviceCalls,
        });
      }
      if (path === "/api/capabilities") {
        capabilityCalls += 1;
        if (capabilityCalls === 3) {
          return Promise.reject(new TypeError("Failed to fetch"));
        }
      }
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();
    expect(view.get("[data-test='device-toggle-0']").attributes("disabled")).toBeUndefined();

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();

    expect(view.text()).toContain("CURRENT DEVICE");
    expect(view.get("[data-test='host-detail-device-error']").text()).toContain(
      "Couldn’t verify compute device controls",
    );
    expect(view.get("[data-test='device-toggle-0']").attributes("disabled")).toBeDefined();

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();

    expect(view.find("[data-test='host-detail-device-error']").exists()).toBe(false);
    expect(view.get("[data-test='device-toggle-0']").attributes("disabled")).toBeUndefined();
  });

  it("coalesces an arbitrary burst of device invalidations behind one in-flight poll", async () => {
    vi.useFakeTimers();
    const slowPoll = deferred<unknown>();
    let deviceCalls = 0;
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        if (deviceCalls === 1) {
          return Promise.resolve({
            devices: [raceDevice("INITIAL DEVICE")],
            plan_version: 1,
          });
        }
        if (deviceCalls === 2) return slowPoll.promise;
        return Promise.resolve({
          devices: [raceDevice("COALESCED DEVICE")],
          plan_version: 3,
        });
      }
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();
    const events = stream("/api/events");
    await vi.advanceTimersByTimeAsync(5_000);
    for (let index = 0; index < 20; index += 1) {
      events.options.onEvent("message", JSON.stringify({ type: "device_state_changed" }));
    }
    await flushPromises();
    expect(deviceCalls).toBe(2);

    slowPoll.resolve({
      devices: [raceDevice("STALE SLOW DEVICE")],
      plan_version: 2,
    });
    await flushPromises();

    expect(deviceCalls).toBe(3);
    expect(view.text()).toContain("COALESCED DEVICE");
    expect(view.text()).not.toContain("STALE SLOW DEVICE");
  });

  it("retargets a pending device poll and suppresses its stale host result", async () => {
    vi.useFakeTimers();
    const staleStudioPoll = deferred<unknown>();
    let studioDeviceCalls = 0;
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/devices") {
        if (requestTarget.baseUrl === studio.baseUrl) {
          studioDeviceCalls += 1;
          return studioDeviceCalls === 1
            ? Promise.resolve({
                devices: [raceDevice("STUDIO DEVICE")],
                plan_version: 1,
              })
            : staleStudioPoll.promise;
        }
        return Promise.resolve({
          devices: [raceDevice("RENDER DEVICE")],
          plan_version: 1,
        });
      }
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();
    await vi.advanceTimersByTimeAsync(5_000);
    expect(studioDeviceCalls).toBe(2);

    await view.setProps({ host: renderBox });
    await flushPromises();
    expect(view.text()).toContain("RENDER DEVICE");

    staleStudioPoll.resolve({
      devices: [raceDevice("STALE STUDIO DEVICE")],
      plan_version: 99,
    });
    await flushPromises();
    expect(view.text()).not.toContain("STALE STUDIO DEVICE");

    await vi.advanceTimersByTimeAsync(5_000);
    await flushPromises();
    const lastDeviceTarget = apiJsonTo.mock.calls
      .filter(([, path]) => path === "/api/devices")
      .at(-1)?.[0];
    expect(lastDeviceTarget).toEqual(renderTarget);
  });

  it("stops device polling and suppresses an in-flight result after unmount", async () => {
    vi.useFakeTimers();
    const pendingPoll = deferred<unknown>();
    let deviceCalls = 0;
    const originalApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((requestTarget, path) => {
      if (path === "/api/devices") {
        deviceCalls += 1;
        return deviceCalls === 1
          ? Promise.resolve({
              devices: [raceDevice("MOUNTED DEVICE")],
              plan_version: 1,
            })
          : pendingPoll.promise;
      }
      return originalApi(requestTarget, path);
    });

    const view = await mountDetail();
    const events = stream("/api/events");
    await vi.advanceTimersByTimeAsync(5_000);
    expect(deviceCalls).toBe(2);

    view.unmount();
    wrapper = null;
    expect(events.options.signal.aborted).toBe(true);

    pendingPoll.resolve({
      devices: [raceDevice("UNMOUNTED DEVICE")],
      plan_version: 2,
    });
    await vi.advanceTimersByTimeAsync(20_000);
    await flushPromises();
    expect(deviceCalls).toBe(2);
  });

  it("renders the H3 capability panel below the primary instrument sections", async () => {
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/capabilities") {
        return Promise.resolve({
          events: { available: true },
          devices: { available: true, lifecycle: true, restart_enable: false },
          dispatch: { active_mode: "v2", v2_authoritative: true },
          ...authenticatedMiniMaxH3Capabilities(),
        });
      }
      return base(target, path) as Promise<unknown>;
    });
    const view = await mountDetail();

    const html = view.html();
    const h3At = html.indexOf('data-test="h3-inventory"');
    expect(h3At).toBeGreaterThan(-1);
    expect(html.indexOf("host-telemetry-title")).toBeLessThan(h3At);
    expect(html.indexOf("host-models-title")).toBeLessThan(h3At);
  });

  it("collapses VRAM and RAM into one Memory row on a unified-memory Metal host", async () => {
    const view = await mountDetail();
    stream("/api/resources/stream").options.onEvent(
      "snapshot",
      JSON.stringify({
        hostname: "studio",
        timestamp: 1,
        gpus: [
          {
            ordinal: 0,
            name: "Apple Metal GPU",
            backend: "metal",
            vram_total: 51_500_000_000,
            vram_used: 46_900_000_000,
            gpu_utilization: null,
          },
        ],
        system_ram: { total: 51_500_000_000, used: 46_900_000_000 },
        cpu: { cores: 16, usage_percent: 44 },
      }),
    );
    await flushPromises();

    expect(
      view.find("[role='meter'][aria-label='Unified memory usage for Apple Metal GPU']").exists(),
    ).toBe(true);
    expect(view.text()).toContain("MEMORY");
    expect(view.text()).not.toContain("VRAM");
    expect(view.find("[role='meter'][aria-label='RAM usage']").exists()).toBe(false);
    expect(view.find("[role='meter'][aria-label='CPU usage']").exists()).toBe(true);
  });

  it("renders every status GPU before the resource stream produces a snapshot", async () => {
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string): Promise<unknown> => {
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
      if (path === "/api/queue?limit=8") return Promise.resolve({ entries: [] });
      return Promise.reject(new Error(`Unexpected API path: ${path} for ${target.baseUrl}`));
    });

    const view = await mountDetail();

    expect(view.text()).toContain("NVIDIA RTX 3090");
    expect(view.text()).toContain("NVIDIA B200");
    expect(view.text()).toContain("8.0 GB/24.0 GB");
    expect(view.text()).toContain("20.0 GB/80.0 GB");
  });

  it("mutates one device on the exact remote and reloads the lifecycle state", async () => {
    const device = {
      id: "cuda:GPU-3090",
      backend: "cuda",
      ordinal: 1,
      device_kind: "full_gpu",
      nvml_uuid: "GPU-3090",
      physical_uuid: "GPU-3090",
      mig_uuid: null,
      mig_parent_uuid: null,
      mig_profile: null,
      name: "NVIDIA RTX 3090",
      pci_bus_id: "0000:02:00.0",
      compute_capability: "8.6",
      memory: {
        total_bytes: 24_000_000_000,
        used_bytes: 4_000_000_000,
        mold_used_bytes: null,
        other_used_bytes: null,
      },
      telemetry: {
        utilization_percent: 10,
        temperature_c: 45,
        power_w: 80,
      },
      desired_enabled: true,
      admin_state: "enabled",
      health: "healthy",
      activity: "idle",
      schedulable: true,
      unschedulable_reason: null,
      loaded_models: [],
      active_work_id: null,
      planned_work_ids: [],
    };
    let deviceLoads = 0;
    const existingApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target, path) => {
      if (path !== "/api/devices") return existingApi(target, path);
      deviceLoads += 1;
      return Promise.resolve({
        devices:
          deviceLoads === 1
            ? [device]
            : [
                {
                  ...device,
                  desired_enabled: false,
                  admin_state: "disabled",
                  schedulable: false,
                  unschedulable_reason: "device_disabled",
                },
              ],
        plan_version: deviceLoads + 1,
      });
    });

    const view = await mountDetail();
    expect(view.get("[data-test='host-detail-devices']").text()).toContain("NVIDIA RTX 3090");

    await view.get("[data-test='device-toggle-1']").trigger("click");
    await flushPromises();

    expect(setDeviceEnabled).toHaveBeenCalledWith(studioTarget, "cuda:GPU-3090", false);
    expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/devices")).toHaveLength(2);
    expect(view.get("[data-test='device-toggle-1']").text()).toBe("Enable");
  });

  it("shows persisted restart recovery instead of an unavailable disable action", async () => {
    const pendingRestart = {
      id: "cuda:GPU-3090",
      backend: "cuda",
      ordinal: 1,
      device_kind: "full_gpu",
      nvml_uuid: "GPU-3090",
      physical_uuid: "GPU-3090",
      mig_uuid: null,
      mig_parent_uuid: null,
      mig_profile: null,
      name: "NVIDIA RTX 3090",
      pci_bus_id: "0000:02:00.0",
      compute_capability: "8.6",
      memory: {
        total_bytes: 24_000_000_000,
        used_bytes: 4_000_000_000,
        mold_used_bytes: null,
        other_used_bytes: null,
      },
      telemetry: {
        utilization_percent: 10,
        temperature_c: 45,
        power_w: 80,
      },
      desired_enabled: true,
      restart_required: true,
      admin_state: "enabled",
      health: "unavailable",
      activity: "idle",
      schedulable: false,
      unschedulable_reason: "device_unavailable",
      loaded_models: [],
      active_work_id: null,
      planned_work_ids: [],
    };
    let deviceLoads = 0;
    const existingApi = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation((target, path) => {
      if (path === "/api/devices") {
        deviceLoads += 1;
        return Promise.resolve({
          devices: [
            deviceLoads === 1
              ? {
                  ...pendingRestart,
                  desired_enabled: false,
                  restart_required: false,
                  admin_state: "disabled",
                  health: "healthy",
                  unschedulable_reason: "device_disabled",
                }
              : pendingRestart,
          ],
          plan_version: deviceLoads,
        });
      }
      if (path === "/api/capabilities")
        return Promise.resolve({
          devices: { available: true, lifecycle: true, restart_enable: true },
          dispatch: { active_mode: "observe", v2_authoritative: false },
        });
      return existingApi(target, path);
    });

    const view = await mountDetail();
    const action = view.get("[data-test='device-toggle-1']");
    expect(action.text()).toBe("Enable on restart");
    await action.trigger("click");
    await flushPromises();
    expect(setDeviceEnabled).toHaveBeenCalledWith(studioTarget, "cuda:GPU-3090", true);

    const pending = view.get("[data-test='device-toggle-1']");
    expect(pending.text()).toBe("Enabled on restart");
    expect(pending.attributes("disabled")).toBeDefined();
    const row = view.get("[data-test='device-card']");
    expect(row.text()).toContain("Restart required");
    expect(row.text()).not.toContain("unavailable");
  });

  it("keeps durable backlog, loaded page, and runtime window as separate quantities", async () => {
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/status") return Promise.resolve(serverStatus({ queue_depth: 7 }));
      if (path === "/api/models") return Promise.resolve([]);
      if (path === "/api/queue?limit=8") return Promise.resolve({ entries: [queueEntries[0]] });
      return Promise.reject(new Error(`Unexpected API path: ${path} for ${target.baseUrl}`));
    });

    const view = await mountDetail();

    expect(view.get("[data-test='host-detail-queue-total']").text()).toBe("7 total");
    expect(view.get("[data-test='host-detail-queue-scope']").text()).toBe(
      "Showing 1 of 7 jobs",
    );
  });

  it("does not infer total backlog from an older host's loaded page or runtime window", async () => {
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/status") {
        return Promise.resolve(serverStatus({ queue_depth: null, queue_capacity: 2 }));
      }
      if (path === "/api/models") return Promise.resolve([]);
      if (path === "/api/queue?limit=2") return Promise.resolve({ entries: [queueEntries[0]] });
      return Promise.reject(new Error(`Unexpected API path: ${path} for ${target.baseUrl}`));
    });

    const view = await mountDetail();

    expect(view.get("[data-test='host-detail-queue-total']").text()).toBe("1 loaded");
    expect(view.get("[data-test='host-detail-queue-scope']").text()).toBe("1 job loaded");
  });

  it("pages by host capacity and loads the durable queue tail on demand", async () => {
    vi.useFakeTimers();
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/status") {
        return Promise.resolve(serverStatus({ queue_depth: 3, queue_capacity: 2 }));
      }
      if (path === "/api/models") return Promise.resolve([]);
      if (path === "/api/queue?limit=2") {
        return Promise.resolve({
          entries: [queueEntries[0], queueEntries[1]],
          page: { limit: 2, offset: 0, returned: 2, next_cursor: "tail" },
        });
      }
      if (path === "/api/queue?limit=2&cursor=tail") {
        return Promise.resolve({
          entries: [
            {
              ...queueEntries[1],
              id: "job-tail",
              position: 3,
            },
          ],
          page: { limit: 2, offset: 2, returned: 1 },
        });
      }
      return Promise.reject(new Error(`Unexpected API path: ${path} for ${target.baseUrl}`));
    });

    const view = await mountDetail();
    expect(view.get("[data-test='host-detail-queue-total']").text()).toBe("3 total");
    expect(view.get("[data-test='host-detail-queue-scope']").text()).toBe(
      "Showing 2 of 3 jobs",
    );
    expect(view.get("[data-test='host-detail-queue']").findAll("li")).toHaveLength(2);

    await view.get("[data-test='host-detail-queue-load-more']").trigger("click");
    await flushPromises();

    expect(view.get("[data-test='host-detail-queue']").findAll("li")).toHaveLength(3);
    expect(view.find("[data-test='host-detail-queue-load-more']").exists()).toBe(false);

    await vi.advanceTimersByTimeAsync(5_001);
    expect(view.get("[data-test='host-detail-queue']").findAll("li")).toHaveLength(2);
    expect(view.find("[data-test='host-detail-queue-load-more']").exists()).toBe(true);
    vi.useRealTimers();
  });

  it("falls back to the status queue depth when the queue API is unavailable", async () => {
    apiJsonTo.mockImplementation((_target: { baseUrl: string }, path: string): Promise<unknown> => {
      if (path === "/api/status") return Promise.resolve(serverStatus({ queue_depth: 7 }));
      if (path === "/api/models") return Promise.resolve([]);
      if (path === "/api/queue?limit=8") return Promise.reject(new Error("queue unsupported"));
      return Promise.reject(new Error(`Unexpected API path: ${path}`));
    });

    const view = await mountDetail();

    expect(view.get("[data-test='host-detail-queue-total']").text()).toBe("7 total");
    expect(view.get("[data-test='host-detail-queue-scope']").text()).toBe(
      "Queue details unavailable",
    );
  });

  it.each([
    ["404", () => new ApiError("queue unsupported", 404)],
    ["transport failure", () => new TypeError("Failed to fetch")],
    ["500", () => new ApiError("server failure", 500)],
  ])(
    "clears stale queue entries and plan after %s, then restores authority",
    async (_label, failure) => {
      vi.useFakeTimers();
      try {
        let queueCalls = 0;
        const originalApi = apiJsonTo.getMockImplementation()!;
        apiJsonTo.mockImplementation((target, path) => {
          if (path === "/api/status") {
            return Promise.resolve(serverStatus({ queue_depth: 7 }));
          }
          if (path === "/api/queue?limit=8") {
            queueCalls += 1;
            if (queueCalls === 1) {
              return Promise.resolve({
                entries: [queueEntries[0]],
                plan: {
                  plan_version: 1,
                  state_version: 1,
                  optimizer_state: "optimized",
                  dirty_since_unix_ms: null,
                  next_replan_at_unix_ms: null,
                  work_items: [
                    {
                      work_id: "stale-cpu-work",
                      parent_id: "parent",
                      work_kind: "post_upscale",
                      priority_class: "user",
                      queue_rank: 0,
                      bypass_count: 0,
                      planned_device_id: null,
                      planned_lane_kind: "host_utility",
                      lane_order: 0,
                      estimate_confidence: "low",
                    },
                  ],
                },
              });
            }
            if (queueCalls === 2) {
              return Promise.reject(failure());
            }
            return Promise.resolve({
              entries: [queueEntries[0]],
              plan: {
                plan_version: 2,
                state_version: 2,
                optimizer_state: "optimized",
                dirty_since_unix_ms: null,
                next_replan_at_unix_ms: null,
                work_items: [
                  {
                    work_id: "restored-cpu-work",
                    parent_id: "parent",
                    work_kind: "post_upscale",
                    priority_class: "user",
                    queue_rank: 0,
                    bypass_count: 0,
                    planned_device_id: null,
                    planned_lane_kind: "host_utility",
                    lane_order: 0,
                    estimate_confidence: "low",
                  },
                ],
              },
            });
          }
          return originalApi(target, path);
        });

        const view = await mountDetail();
        expect(view.text()).toContain("Post upscale");
        expect(view.find("[data-test='host-detail-queue']").exists()).toBe(true);
        expect(view.get("[data-test='host-detail-queue-total']").text()).toBe("7 total");
        expect(view.get("[data-test='host-detail-queue-scope']").text()).toBe(
          "Showing 2 of 7 jobs",
        );

        await vi.advanceTimersByTimeAsync(5_000);
        await flushPromises();

        expect(view.text()).not.toContain("Post upscale");
        expect(view.find("[data-test='host-detail-queue']").exists()).toBe(false);
        expect(view.get("[data-test='host-detail-queue-total']").text()).toBe("7 total");
        expect(view.get("[data-test='host-detail-queue-scope']").text()).toBe(
          "Queue details unavailable",
        );

        await vi.advanceTimersByTimeAsync(5_000);
        await flushPromises();

        expect(view.text()).toContain("Post upscale");
        expect(view.get("[data-test='host-detail-queue-total']").text()).toBe("7 total");
        expect(view.get("[data-test='host-detail-queue-scope']").text()).toBe(
          "Showing 2 of 7 jobs",
        );
      } finally {
        vi.useRealTimers();
      }
    },
  );

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
      if (path === "/api/queue?limit=8") return Promise.resolve({ entries: [] });
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
    const oldDeviceEvents = stream("/api/events");

    await view.setProps({ host: renderBox });
    await flushPromises();

    expect(oldResources.options.signal.aborted).toBe(true);
    expect(oldDownloads.options.signal.aborted).toBe(true);
    expect(oldDeviceEvents.options.signal.aborted).toBe(true);
    expect(apiJsonTo).toHaveBeenCalledWith(renderTarget, "/api/status");
    expect(apiJsonTo).toHaveBeenCalledWith(renderTarget, "/api/models");
    expect(apiJsonTo).toHaveBeenCalledWith(renderTarget, "/api/queue?limit=8");
    expect(stream("/api/resources/stream", renderTarget).options.signal.aborted).toBe(false);
    expect(stream("/api/downloads/stream", renderTarget).options.signal.aborted).toBe(false);
    expect(stream("/api/events", renderTarget).options.signal.aborted).toBe(false);

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
    expect(stream("/api/events", renderTarget).options.signal.aborted).toBe(true);
  });

  it("restarts every authoritative source when the same host rotates credentials", async () => {
    const view = await mountDetail();
    const oldDeviceEvents = stream("/api/events");
    const rotated = { ...studio, apiKey: "rotated-secret" };

    await view.setProps({ host: rotated });
    await flushPromises();

    expect(oldDeviceEvents.options.signal.aborted).toBe(true);
    expect(stream("/api/events").options.target).toEqual({
      baseUrl: studio.baseUrl,
      apiKey: "rotated-secret",
    });
    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: studio.baseUrl, apiKey: "rotated-secret" },
      "/api/devices",
    );
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
      if (path === "/api/queue?limit=8") return Promise.resolve({ entries: [] });
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

describe("MobileHostDetail Library card", () => {
  function installLibraryApi(options: { locked?: boolean; retention?: number } = {}): void {
    const base = apiJsonTo.getMockImplementation()!;
    apiJsonTo.mockImplementation(
      (target: { baseUrl: string }, path: string, init?: RequestInit): Promise<unknown> => {
        if (path === "/api/capabilities") {
          return Promise.resolve({
            events: { available: true },
            devices: { available: true, lifecycle: true, restart_enable: false },
            dispatch: { active_mode: "v2", v2_authoritative: true },
            gallery: {
              can_delete: true,
              organize: true,
              trash: { enabled: true, retention_days: options.retention ?? 30 },
            },
          });
        }
        if (path === "/api/config/gallery.trash_retention_days") {
          if (init?.method === "PUT") {
            const body = JSON.parse(String(init.body)) as { value: number };
            return Promise.resolve({
              key: "gallery.trash_retention_days",
              value: body.value,
              source: "db",
            });
          }
          return Promise.resolve({
            key: "gallery.trash_retention_days",
            value: options.retention ?? 30,
            source: options.locked ? "env" : "db",
            env_var: options.locked ? "MOLD_GALLERY_TRASH_RETENTION_DAYS" : null,
          });
        }
        if (path === "/api/gallery?view=trash") {
          return Promise.resolve([
            {
              filename: "old print.png",
              timestamp: 1_700_000_000,
              format: "png",
              metadata: { prompt: "", model: "flux-dev:q8" },
              trashed_at: 1_700_000_000,
              purge_at: 1_702_592_000,
            },
          ]);
        }
        return base(target, path, init);
      },
    );
  }

  it("hides the card when the host does not advertise a trash", async () => {
    const view = await mountDetail();
    expect(view.find("[data-test='host-detail-library']").exists()).toBe(false);
  });

  it("reads the host's retention and trash count into the card", async () => {
    installLibraryApi();
    const view = await mountDetail();
    await flushPromises();

    const card = view.get("[data-test='host-detail-library']");
    const select = card.get("[data-test='host-detail-retention']").element as HTMLSelectElement;
    expect(select.value).toBe("30");
    expect(card.get("[data-test='host-detail-trash-row']").text()).toContain("Prints in trash: 1");
    expect(apiJsonTo).toHaveBeenCalledWith(
      studioTarget,
      "/api/config/gallery.trash_retention_days",
      expect.anything(),
    );
  });

  it("writes a new retention through PUT on the exact authenticated host", async () => {
    installLibraryApi();
    const view = await mountDetail();
    await flushPromises();

    await view.get("[data-test='host-detail-retention']").setValue("7");
    await flushPromises();

    const put = apiJsonTo.mock.calls.find(
      ([, path, init]) =>
        path === "/api/config/gallery.trash_retention_days" &&
        (init as RequestInit | undefined)?.method === "PUT",
    );
    expect(put?.[0]).toEqual(studioTarget);
    expect(JSON.parse(String((put?.[2] as RequestInit).body))).toEqual({ value: 7 });
    expect(
      (view.get("[data-test='host-detail-retention']").element as HTMLSelectElement).value,
    ).toBe("7");
  });

  it("keeps the selector read-only with a Retry when the retention probe fails", async () => {
    installLibraryApi();
    const base = apiJsonTo.getMockImplementation()!;
    let failConfig = true;
    apiJsonTo.mockImplementation(
      (target: { baseUrl: string }, path: string, init?: RequestInit): Promise<unknown> => {
        if (
          path === "/api/config/gallery.trash_retention_days" &&
          init?.method !== "PUT" &&
          failConfig
        ) {
          return Promise.reject(new Error("config unavailable"));
        }
        return base(target, path, init);
      },
    );
    const view = await mountDetail();
    await flushPromises();

    // Unknown authority = read-only: an env-pinned key must never become
    // editable because its probe failed transiently.
    expect(view.get("[data-test='host-detail-retention']").attributes("disabled")).toBeDefined();
    const failure = view.get("[data-test='host-detail-retention-error']");
    expect(failure.text()).toContain("retention");

    failConfig = false;
    await view.get("[data-test='host-detail-retention-retry']").trigger("click");
    await flushPromises();

    expect(view.find("[data-test='host-detail-retention-error']").exists()).toBe(false);
    expect(view.get("[data-test='host-detail-retention']").attributes("disabled")).toBeUndefined();
    expect(
      (view.get("[data-test='host-detail-retention']").element as HTMLSelectElement).value,
    ).toBe("30");
  });

  it("keeps an env-pinned retention read-only and says which variable owns it", async () => {
    installLibraryApi({ locked: true });
    const view = await mountDetail();
    await flushPromises();

    expect(view.get("[data-test='host-detail-retention']").attributes("disabled")).toBeDefined();
    expect(view.get("[data-test='host-detail-retention-locked']").text()).toContain(
      "MOLD_GALLERY_TRASH_RETENTION_DAYS",
    );
  });

  it("empties the host's trash only after the two-step confirm", async () => {
    installLibraryApi();
    const view = await mountDetail();
    await flushPromises();

    const emptyCalls = () =>
      apiJsonTo.mock.calls.filter(
        ([, path, init]) =>
          path === "/api/gallery/trash" && (init as RequestInit | undefined)?.method === "DELETE",
      );

    const button = view.get("[data-test='host-detail-empty-trash']");
    await button.trigger("click");
    expect(emptyCalls()).toHaveLength(0);
    expect(view.get("[data-test='host-detail-empty-prompt']").text()).toContain(
      "Delete everything in this host's trash forever?",
    );

    apiJsonTo.mockImplementation((_target: { baseUrl: string }, path: string) =>
      path === "/api/gallery/trash"
        ? Promise.resolve({ purged: 1 })
        : Promise.reject(new Error(`Unexpected API path: ${path}`)),
    );
    await button.trigger("click");
    await flushPromises();
    expect(view.get("[data-test='host-detail-trash-row']").text()).toContain("Prints in trash: 0");
  });
});
