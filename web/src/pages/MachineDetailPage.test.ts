import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick, ref } from "vue";
import type {
  HostCapabilities,
  HostStatus,
} from "../components/machines/hostClient";
import type {
  DownloadsListingWire,
  ModelInfoExtended,
  QueueEntry,
  ResourceSnapshot,
} from "../types";
import { addHost } from "../lib/hostRegistry";
import MachineDetailPage from "./MachineDetailPage.vue";

// Mutable fixtures the hostClient mock reads at call time.
let poll: {
  status: ReturnType<typeof ref<HostStatus | null>>;
  resources: ReturnType<typeof ref<ResourceSnapshot | null>>;
  online: ReturnType<typeof ref<boolean>>;
  lastSeen: ReturnType<typeof ref<number | null>>;
  error: ReturnType<typeof ref<string | null>>;
  loading: ReturnType<typeof ref<boolean>>;
  refresh: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
};
let caps: HostCapabilities;
let queueEntries: QueueEntry[];
let models: ModelInfoExtended[];
let downloadsListing: DownloadsListingWire;

const cancelQueueJob = vi.fn().mockResolvedValue(undefined);
const setQueueJobLane = vi.fn().mockResolvedValue(undefined);
const moveQueueJob = vi.fn().mockResolvedValue(undefined);

const routeHolder = { id: "origin" };

vi.mock("vue-router", () => ({
  useRoute: () => ({ params: { id: routeHolder.id } }),
  RouterLink: {
    name: "RouterLink",
    props: ["to"],
    template: "<a><slot /></a>",
  },
}));

vi.mock("../components/machines/hostClient", () => ({
  useHostPoll: () => poll,
  hostCapabilities: () => Promise.resolve(caps),
  hostQueue: () => Promise.resolve({ entries: queueEntries }),
  hostModels: () => Promise.resolve(models),
  hostDownloads: () => Promise.resolve(downloadsListing),
  // Wrappers defer reading the vi.fns until call time (the factory is hoisted
  // above their declarations).
  cancelQueueJob: (...a: unknown[]) => cancelQueueJob(...a),
  setQueueJobLane: (...a: unknown[]) => setQueueJobLane(...a),
  moveQueueJob: (...a: unknown[]) => moveQueueJob(...a),
}));

function makeStatus(over: Partial<HostStatus> = {}): HostStatus {
  return {
    version: "0.16.0",
    models_loaded: [],
    busy: false,
    uptime_secs: 3_600 + 720,
    queue_depth: 2,
    gpus: [
      {
        ordinal: 0,
        name: "RTX 4090",
        vram_total_bytes: 24_000_000_000,
        vram_used_bytes: 12_000_000_000,
        state: "idle",
      },
    ],
    models_disk: { total_bytes: 994_000_000_000, free_bytes: 213_000_000_000 },
    ...over,
  };
}

function makeResources(over: Partial<ResourceSnapshot> = {}): ResourceSnapshot {
  return {
    hostname: "studio",
    timestamp: 0,
    system_ram: { total: 0, used: 0, used_by_mold: 0, used_by_other: 0 },
    gpus: [
      {
        ordinal: 0,
        name: "RTX 4090",
        backend: "cuda",
        vram_total: 24_000_000_000,
        vram_used: 12_000_000_000,
        vram_used_by_mold: null,
        vram_used_by_other: null,
        gpu_utilization: 55,
      },
    ],
    ...over,
  };
}

function queued(id: string, position: number): QueueEntry {
  return {
    id,
    model: `flux-${id}`,
    state: "queued",
    started_at_unix_ms: 0,
    position,
  };
}

let wrapper: ReturnType<typeof mount> | null = null;
async function mountDetail() {
  wrapper = mount(MachineDetailPage);
  await flushPromises();
  await nextTick();
  return wrapper;
}

beforeEach(() => {
  localStorage.clear();
  const host = addHost({ url: "192.168.1.20:7680", name: "Studio" });
  routeHolder.id = host.id;
  caps = { queue: { can_reorder: false } };
  queueEntries = [];
  models = [];
  downloadsListing = { active: null, active_jobs: [], queued: [], history: [] };
  cancelQueueJob.mockClear();
  setQueueJobLane.mockClear();
  moveQueueJob.mockClear();
  poll = {
    status: ref<HostStatus | null>(makeStatus()),
    resources: ref<ResourceSnapshot | null>(makeResources()),
    online: ref(true),
    lastSeen: ref<number | null>(Date.now()),
    error: ref<string | null>(null),
    loading: ref(false),
    refresh: vi.fn(),
    stop: vi.fn(),
  };
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

describe("MachineDetailPage — telemetry", () => {
  it("maps the resource snapshot into the telemetry card", async () => {
    const w = await mountDetail();
    expect(w.get('[data-test="machine-detail-title"]').text()).toBe("Studio");
    expect(w.get('[data-test="telemetry-gpu"]').text()).toContain(
      "RTX 4090 · cuda",
    );
    expect(w.get('[data-test="telemetry-load"]').text()).toBe("55%");
    expect(w.get('[data-test="telemetry-mem"]').text()).toBe("50%");
    expect(w.get('[data-test="telemetry-queue"]').text()).toBe("2");
    expect(w.get('[data-test="telemetry-uptime"]').text()).toBe("1h 12m");
    expect(w.get('[data-test="telemetry-storage"]').text()).toContain(
      "free of",
    );
  });

  it("renders em-dash fallbacks for metrics the host does not expose", async () => {
    poll.status.value = {
      version: "0.16.0",
      models_loaded: [],
      busy: false,
      uptime_secs: 5,
    };
    poll.resources.value = null;
    const w = await mountDetail();
    expect(w.get('[data-test="telemetry-gpu"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-load"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-mem"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-temp"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-queue"]').text()).toBe("—");
  });

  it("shows a not-found card for an unknown host id", async () => {
    routeHolder.id = "ghost-host";
    const w = await mountDetail();
    expect(w.find('[data-test="detail-not-found"]').exists()).toBe(true);
  });
});

describe("MachineDetailPage — queue", () => {
  it("hides reorder controls when the host lacks can_reorder", async () => {
    caps = { queue: { can_reorder: false } };
    queueEntries = [queued("a", 0), queued("b", 1)];
    const w = await mountDetail();
    expect(w.findAll('[data-test="queue-row"]')).toHaveLength(2);
    expect(w.find('[data-test="queue-up"]').exists()).toBe(false);
    expect(w.find('[data-test="queue-down"]').exists()).toBe(false);
  });

  it("shows reorder controls when the host advertises can_reorder", async () => {
    caps = { queue: { can_reorder: true } };
    queueEntries = [queued("a", 0), queued("b", 1)];
    const w = await mountDetail();
    expect(w.find('[data-test="queue-up"]').exists()).toBe(true);
    expect(w.find('[data-test="queue-down"]').exists()).toBe(true);
  });

  it("reorders a job through the host client to an absolute position", async () => {
    caps = { queue: { can_reorder: true } };
    queueEntries = [queued("a", 0), queued("b", 1)];
    const w = await mountDetail();
    // Second row's "up" moves it to position 0.
    const downButtons = w.findAll('[data-test="queue-down"]');
    await downButtons[0]!.trigger("click");
    expect(moveQueueJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: routeHolder.id }),
      "a",
      1,
    );
  });

  it("cancels a queued job through the host client", async () => {
    queueEntries = [queued("a", 0)];
    const w = await mountDetail();
    await w.get('[data-test="queue-cancel"]').trigger("click");
    expect(cancelQueueJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: routeHolder.id }),
      "a",
    );
  });

  it("renders a per-GPU lane selector only when the host has multiple GPUs", async () => {
    queueEntries = [queued("a", 0)];
    poll.status.value = makeStatus({
      gpus: [
        {
          ordinal: 0,
          name: "RTX 4090",
          vram_total_bytes: 24_000_000_000,
          vram_used_bytes: 0,
          state: "idle",
        },
        {
          ordinal: 1,
          name: "RTX 4090",
          vram_total_bytes: 24_000_000_000,
          vram_used_bytes: 0,
          state: "idle",
        },
      ],
    });
    const w = await mountDetail();
    expect(w.find('[data-test="queue-lane"]').exists()).toBe(true);
  });
});

describe("MachineDetailPage — models", () => {
  it("lists installed models with a loaded badge", async () => {
    models = [
      {
        name: "flux-dev:q8",
        family: "flux",
        size_gb: 12,
        is_loaded: true,
        last_used: null,
        hf_repo: "bfl/flux",
        downloaded: true,
        default_steps: 4,
        default_guidance: 0,
        default_width: 1024,
        default_height: 1024,
        description: "",
      },
      {
        name: "sdxl-base",
        family: "sdxl",
        size_gb: 6,
        is_loaded: false,
        last_used: null,
        hf_repo: "sdxl",
        downloaded: false,
        default_steps: 30,
        default_guidance: 7,
        default_width: 1024,
        default_height: 1024,
        description: "",
      },
    ];
    const w = await mountDetail();
    // Only the downloaded model is installed.
    expect(w.findAll('[data-test="model-row"]')).toHaveLength(1);
    expect(w.find('[data-test="model-loaded"]').exists()).toBe(true);
  });
});
