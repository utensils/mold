import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import StatusPopover from "./StatusPopover.vue";
import { useConnectionStore } from "../../stores/connection";
import { useGenerationStore } from "../../stores/generation";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";

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
  onEvent?: ((event: string, data: string) => void) | undefined;
}
const streamCalls: StreamCall[] = [];
vi.mock("../../lib/api/sse", () => ({
  sseStream: (
    path: string,
    opts: {
      target?: { baseUrl: string } | null;
      onEvent?: (event: string, data: string) => void;
    },
  ) => {
    streamCalls.push({ path, target: opts.target ?? null, onEvent: opts.onEvent });
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

  it("shows the host selected in the Create header while idle", async () => {
    const wrapper = mountPopover();
    addRemoteHost();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    await flushPromises();
    await openPopover(wrapper);
    expect(wrapper.text()).toContain("hal9000");
    expect(wrapper.text()).toContain("queue 2/200");
    expect(wrapper.text()).toContain("flux2-klein:q4");
    expect(streamCalls.at(-1)?.target?.baseUrl).toBe("http://hal9000:7680");
  });

  it("reports an offline selected host instead of presenting missing telemetry as status", async () => {
    const wrapper = mountPopover();
    const hosts = addRemoteHost();
    const remote = hosts.extras.find((host) => host.id === "hal9000-7680")!;
    remote.label = "Parity Fixture";
    remote.status = "error";
    remote.error = "connection refused";
    delete hosts.telemetry[remote.id];
    useAppPrefsStore().settings = { generateTargetHost: remote.id } as never;

    await flushPromises();
    await openPopover(wrapper);

    const trigger = wrapper.get("[data-test='status-trigger']");
    expect(wrapper.get("[role='dialog']").text()).toContain("Parity Fixture · offline");
    expect(wrapper.get("[role='dialog']").text()).toContain("Machine is offline");
    expect(wrapper.get("[role='dialog']").text()).not.toContain("no gpu telemetry");
    expect(trigger.text()).toContain("offline");
    expect(trigger.text()).toContain("—");
    expect(trigger.get("span").classes()).toContain("bg-stop");
    expect(streamCalls.at(-1)?.target ?? null).toBeNull();
  });

  it("describes a local engine failure as an engine lifecycle state", async () => {
    const wrapper = mountPopover();
    useConnectionStore().status = "error";

    await flushPromises();
    await openPopover(wrapper);

    expect(wrapper.get("[role='dialog']").text()).toContain("engine error");
    expect(wrapper.get("[role='dialog']").text()).toContain("Engine error");
    expect(wrapper.get("[role='dialog']").text()).not.toContain("Machine is offline");
    expect(wrapper.get("[data-test='status-trigger']").text()).toContain("error");
  });

  it("keeps showing a concrete Create host selection while another host has a live job", async () => {
    const wrapper = mountPopover();
    addRemoteHost();
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    hosts.telemetry["plato-7680"] = {
      queueDepth: 1,
      queueCapacity: 200,
      version: "0.16.0",
      modelsLoaded: ["qwen-image:bf16"],
      gpuInfo: { name: "NVIDIA L40S", vram_total_mb: 46068, vram_used_mb: 12288 },
    };
    useAppPrefsStore().settings = { generateTargetHost: "plato-7680" } as never;
    useGenerationStore().jobs.push({
      clientId: 1,
      status: "denoising",
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
    } as never);
    await flushPromises();
    await openPopover(wrapper);
    expect(wrapper.text()).toContain("plato");
    expect(wrapper.text()).toContain("queue 1/200");
    expect(wrapper.text()).toContain("qwen-image:bf16");
    expect(wrapper.text()).not.toContain("hal9000");
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

  it("renders one VRAM row per GPU from a multi-GPU snapshot", async () => {
    const wrapper = mountPopover();
    await flushPromises();
    streamCalls.at(-1)?.onEvent?.(
      "snapshot",
      JSON.stringify({
        hostname: "plato",
        timestamp: 0,
        gpus: [
          {
            ordinal: 0,
            name: "NVIDIA L40S",
            backend: "cuda",
            vram_total: 46_100_000_000,
            vram_used: 400_000_000,
          },
          {
            ordinal: 1,
            name: "NVIDIA L40S",
            backend: "cuda",
            vram_total: 46_100_000_000,
            vram_used: 41_500_000_000,
          },
        ],
        system_ram: {
          total: 512_000_000_000,
          used: 455_900_000_000,
          used_by_mold: 0,
          used_by_other: 0,
        },
      }),
    );
    await openPopover(wrapper);
    const rows = wrapper.findAll("[data-test='gpu-row']");
    expect(rows).toHaveLength(2);
    expect(wrapper.text()).toContain("GPU 0");
    expect(wrapper.text()).toContain("GPU 1");
    expect(rows[0]!.text()).toContain("0.4 GB / 46.1 GB");
    expect(rows[1]!.text()).toContain("41.5 GB / 46.1 GB");
  });

  it("falls back to every status worker when a remote resource stream is silent", async () => {
    const wrapper = mountPopover();
    const hosts = addRemoteHost();
    hosts.telemetry["hal9000-7680"]!.gpuWorkers = [
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
    ];
    useGenerationStore().jobs.push({
      clientId: 1,
      status: "denoising",
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
    } as never);
    await flushPromises();
    await openPopover(wrapper);

    const rows = wrapper.findAll("[data-test='gpu-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("NVIDIA RTX 3090");
    expect(rows[1]!.text()).toContain("NVIDIA B200");
  });

  it("aggregates the trigger mini-bar VRAM across all GPUs", async () => {
    const wrapper = mountPopover();
    await flushPromises();
    streamCalls.at(-1)?.onEvent?.(
      "snapshot",
      JSON.stringify({
        hostname: "plato",
        timestamp: 0,
        gpus: [
          {
            ordinal: 0,
            name: "NVIDIA L40S",
            backend: "cuda",
            vram_total: 46_100_000_000,
            vram_used: 400_000_000,
          },
          {
            ordinal: 1,
            name: "NVIDIA L40S",
            backend: "cuda",
            vram_total: 46_100_000_000,
            vram_used: 41_500_000_000,
          },
        ],
        system_ram: {
          total: 512_000_000_000,
          used: 455_900_000_000,
          used_by_mold: 0,
          used_by_other: 0,
        },
      }),
    );
    await flushPromises();
    // (0.4 + 41.5) / (46.1 + 46.1) ≈ 45% — not GPU 0's near-zero usage.
    const bar = wrapper.get("[data-test='status-trigger'] [role='progressbar']");
    expect(bar.attributes("aria-valuenow")).toBe("45");
  });

  it("keeps a single unlabelled row for single-GPU hosts", async () => {
    const wrapper = mountPopover();
    await flushPromises();
    streamCalls.at(-1)?.onEvent?.(
      "snapshot",
      JSON.stringify({
        hostname: "halcyon",
        timestamp: 0,
        gpus: [
          {
            ordinal: 0,
            name: "Apple M3 Ultra",
            backend: "metal",
            vram_total: 196_600_000_000,
            vram_used: 32_800_000_000,
          },
        ],
        system_ram: {
          total: 196_600_000_000,
          used: 64_000_000_000,
          used_by_mold: 0,
          used_by_other: 0,
        },
      }),
    );
    await openPopover(wrapper);
    expect(wrapper.findAll("[data-test='gpu-row']")).toHaveLength(1);
    expect(wrapper.text()).toContain("Apple M3 Ultra");
    expect(wrapper.text()).not.toContain("GPU 0");
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

describe("StatusPopover host-memory pressure", () => {
  const RAM_SNAPSHOT = JSON.stringify({
    hostname: "plato",
    timestamp: 0,
    gpus: [],
    system_ram: {
      total: 64_000_000_000,
      used: 60_000_000_000,
      used_by_mold: 0,
      used_by_other: 0,
    },
  });

  async function openWithRam(hostMemory: unknown) {
    apiJson.mockResolvedValue({
      version: "0.16.0",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 0,
      queue_capacity: 200,
      ...(hostMemory === undefined ? {} : { host_memory: hostMemory }),
    });
    const wrapper = mountPopover();
    await flushPromises();
    streamCalls.at(-1)?.onEvent?.("snapshot", RAM_SNAPSHOT);
    await openPopover(wrapper);
    return wrapper;
  }

  // `/api/status` is the fresher source and the only one polled while neither
  // Machines nor a queued print is holding the cross-host queue poll open.
  it("colors the RAM row from the status snapshot with no queue polling active", async () => {
    const wrapper = await openWithRam({
      total_bytes: 64_000_000_000,
      available_bytes: 4_000_000_000,
      headroom_bytes: 0,
      safety_floor_bytes: 4_000_000_000,
    });
    const ram = wrapper.get("[data-test='status-ram']");
    expect(ram.classes()).toContain("text-stop");
    expect(ram.classes()).not.toContain("text-ink-3");
  });

  it("warns inside one safety floor of the wall", async () => {
    const wrapper = await openWithRam({
      total_bytes: 64_000_000_000,
      available_bytes: 6_000_000_000,
      headroom_bytes: 3_000_000_000,
      safety_floor_bytes: 4_000_000_000,
    });
    expect(wrapper.get("[data-test='status-ram']").classes()).toContain("text-halide");
  });

  // The status poll is primary-only by design, so its ledger describes this
  // Mac. Painting it onto a remote's RAM row would report the wrong machine.
  it("never paints the primary's ledger onto a remote display host", async () => {
    apiJson.mockResolvedValue({
      version: "0.16.0",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 0,
      queue_capacity: 200,
      host_memory: {
        total_bytes: 64_000_000_000,
        available_bytes: 1_000_000_000,
        headroom_bytes: 0,
        safety_floor_bytes: 4_000_000_000,
      },
    });
    const wrapper = mountPopover();
    addRemoteHost();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    await flushPromises();
    streamCalls.at(-1)?.onEvent?.("snapshot", RAM_SNAPSHOT);
    await openPopover(wrapper);
    const ram = wrapper.get("[data-test='status-ram']");
    expect(ram.classes()).toContain("text-ink-3");
    expect(ram.classes()).not.toContain("text-stop");
  });

  it("keeps the neutral look when neither source reports host memory", async () => {
    const wrapper = await openWithRam(undefined);
    const ram = wrapper.get("[data-test='status-ram']");
    expect(ram.classes()).toContain("text-ink-3");
    expect(ram.classes()).not.toContain("text-stop");
    expect(ram.attributes("title")).toBeUndefined();
  });
});
