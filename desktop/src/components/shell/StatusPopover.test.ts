import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import StatusPopover from "./StatusPopover.vue";
import { useConnectionStore } from "../../stores/connection";
import { useGenerationStore } from "../../stores/generation";
import { useHostsStore } from "../../stores/hosts";

const apiJson = vi.fn();
vi.mock("../../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiJson: (...a: unknown[]) => apiJson(...a),
  apiJsonTo: vi.fn().mockRejectedValue(new Error("unused")),
  apiFetchTo: vi.fn().mockRejectedValue(new Error("unused")),
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null }),
}));

interface StreamCall {
  path: string;
  target?: { baseUrl: string } | null;
}
const streamCalls: StreamCall[] = [];
vi.mock("../../lib/api/sse", () => ({
  sseStream: (path: string, opts: { target?: { baseUrl: string } | null }) => {
    streamCalls.push({ path, target: opts.target ?? null });
    return Promise.resolve();
  },
}));

function mountPopover() {
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  return mount(StatusPopover, { global: { plugins: [pinia] } });
}

/** The telemetry lives in the popover now; the trigger only shows a mini bar. */
async function openPopover(wrapper: ReturnType<typeof mountPopover>) {
  await wrapper.get("[data-test='status-trigger']").trigger("click");
  await flushPromises();
}

function addRemoteHost() {
  const hosts = useHostsStore();
  hosts.extras.push({
    id: "hal9000-7680",
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: null,
  });
  hosts.telemetry["hal9000-7680"] = {
    queueDepth: 2,
    queueCapacity: 200,
    version: "0.16.0",
    modelsLoaded: ["flux2-klein:q4"],
    gpuInfo: { name: "NVIDIA GeForce RTX 4090", vram_total_mb: 24564, vram_used_mb: 8192 },
  };
  return hosts;
}

beforeEach(() => {
  vi.clearAllMocks();
  streamCalls.length = 0;
  apiJson.mockResolvedValue({
    version: "0.16.0",
    models_loaded: [],
    uptime_secs: 1,
    queue_depth: 0,
    queue_capacity: 200,
  });
});

describe("StatusPopover host-aware display", () => {
  it("shows the primary engine when nothing is generating remotely", async () => {
    const wrapper = mountPopover();
    await flushPromises();
    await openPopover(wrapper);
    expect(wrapper.text()).toContain("local");
    expect(wrapper.text()).toContain("queue 0/200");
  });

  it("follows a live remote job: chip, queue, and models come from that host", async () => {
    const wrapper = mountPopover();
    addRemoteHost();
    const generation = useGenerationStore();
    generation.jobs.push({
      clientId: 1,
      status: "denoising",
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
    } as never);
    await flushPromises();
    await openPopover(wrapper);
    expect(wrapper.text()).toContain("hal9000");
    expect(wrapper.text()).toContain("queue 2/200");
    expect(wrapper.text()).toContain("flux2-klein:q4");
  });

  it("re-targets the resources stream at the display host", async () => {
    mountPopover();
    addRemoteHost();
    const generation = useGenerationStore();
    generation.jobs.push({
      clientId: 1,
      status: "denoising",
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
    } as never);
    await flushPromises();
    const last = streamCalls.at(-1);
    expect(last?.path).toBe("/api/resources/stream");
    expect(last?.target?.baseUrl).toBe("http://hal9000:7680");
  });

  it("reverts to the primary when the last routed job settles", async () => {
    const wrapper = mountPopover();
    addRemoteHost();
    const generation = useGenerationStore();
    generation.jobs.push({
      clientId: 1,
      status: "denoising",
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
    } as never);
    await flushPromises();
    await openPopover(wrapper);
    expect(wrapper.text()).toContain("hal9000");
    generation.jobs[0]!.status = "complete" as never;
    await flushPromises();
    // Chip and queue fall back to the primary...
    expect(wrapper.text()).toContain("local");
    expect(wrapper.text()).toContain("queue 0/200");
    // ...and the resources stream re-targets the primary (no explicit target).
    expect(streamCalls.at(-1)?.target ?? null).toBeNull();
  });

  it("keeps polling the PRIMARY status for engine recovery while displaying a remote host", async () => {
    mountPopover();
    addRemoteHost();
    const generation = useGenerationStore();
    generation.jobs.push({
      clientId: 1,
      status: "denoising",
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
    } as never);
    await flushPromises();
    // The recovery poll is apiJson (currentTarget = primary), never apiJsonTo.
    expect(apiJson).toHaveBeenCalledWith("/api/status");
  });

  it("falls back to the status gpu_info for a primary whose resources stream is silent", async () => {
    apiJson.mockResolvedValue({
      version: "0.16.0",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 0,
      queue_capacity: 200,
      gpu_info: { name: "Apple M3 Ultra", vram_total_mb: 196608, vram_used_mb: 32768 },
    });
    const wrapper = mountPopover();
    await flushPromises();
    await openPopover(wrapper);
    expect(wrapper.text()).toContain("Apple M3 Ultra");
  });

  it("falls back to telemetry VRAM when the remote resources stream is silent", async () => {
    const wrapper = mountPopover();
    addRemoteHost();
    const generation = useGenerationStore();
    generation.jobs.push({
      clientId: 1,
      status: "denoising",
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
    } as never);
    await flushPromises();
    await openPopover(wrapper);
    expect(wrapper.text()).toContain("NVIDIA GeForce RTX 4090");
  });
});

describe("StatusPopover trigger", () => {
  it("is always visible and toggles the popover open", async () => {
    const wrapper = mountPopover();
    await flushPromises();
    const trigger = wrapper.get("[data-test='status-trigger']");
    expect(trigger.attributes("aria-expanded")).toBe("false");
    await trigger.trigger("click");
    expect(wrapper.get("[data-test='status-trigger']").attributes("aria-expanded")).toBe("true");
    expect(wrapper.find("[role='dialog']").exists()).toBe(true);
  });
});
