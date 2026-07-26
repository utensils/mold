import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { __testing__, useHostRouting } from "./useHostRouting";
import {
  addHost,
  ORIGIN_HOST_ID,
  setGenerateTargetId,
} from "../lib/hostRegistry";
import { AUTO_TARGET_ID, CAPABLE_TARGET_ID } from "../lib/hostRouting";
import type { HostEntry } from "../lib/hostRegistry";
import type { ModelInfoExtended } from "../types";
import type { DeviceInfo, DeviceListResponse } from "@studio/api/devices";

/** Per-host canned `/api/status` + `/api/models` responses, keyed by host id. */
const statuses = new Map<string, unknown>();
const models = new Map<string, ModelInfoExtended[]>();
const devices = new Map<string, DeviceListResponse>();

vi.mock("../components/machines/hostClient", () => ({
  hostStatus: (host: HostEntry) => {
    const canned = statuses.get(host.id);
    return canned
      ? Promise.resolve(canned)
      : Promise.reject(new Error("unreachable"));
  },
  hostModels: (host: HostEntry) => {
    const canned = models.get(host.id);
    return canned
      ? Promise.resolve(canned)
      : Promise.reject(new Error("unreachable"));
  },
  hostDevices: (host: HostEntry) => {
    const canned = devices.get(host.id);
    return canned
      ? Promise.resolve(canned)
      : Promise.reject(new Error("legacy server"));
  },
}));

function model(
  name: string,
  overrides: Partial<ModelInfoExtended> = {},
): ModelInfoExtended {
  return {
    name,
    family: "flux",
    description: "",
    size_gb: 1,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 3.5,
    is_loaded: false,
    hf_repo: "acme/thing",
    downloaded: true,
    ...overrides,
  } as ModelInfoExtended;
}

function status(overrides: Record<string, unknown> = {}) {
  return {
    version: "0.17.0",
    models_loaded: [],
    busy: false,
    uptime_secs: 1,
    queue_depth: 0,
    ...overrides,
  };
}

function device(
  ordinal: number,
  overrides: Partial<DeviceInfo> = {},
): DeviceInfo {
  return {
    id: `cuda:${ordinal}`,
    backend: "cuda",
    ordinal,
    device_kind: "full_gpu",
    nvml_uuid: `GPU-${ordinal}`,
    physical_uuid: `GPU-${ordinal}`,
    mig_uuid: null,
    mig_parent_uuid: null,
    mig_profile: null,
    name: `GPU ${ordinal}`,
    pci_bus_id: null,
    compute_capability: "8.6",
    memory: {
      total_bytes: 24 * 1024 ** 3,
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
    admin_state: "enabled",
    health: "healthy",
    activity: "idle",
    schedulable: true,
    unschedulable_reason: null,
    loaded_models: [],
    active_work_id: null,
    planned_work_ids: [],
    ...overrides,
  };
}

describe("useHostRouting", () => {
  beforeEach(() => {
    localStorage.clear();
    statuses.clear();
    models.clear();
    devices.clear();
    __testing__.reset();
  });

  afterEach(() => {
    __testing__.reset();
  });

  it("lists this server alone when no remote is registered", async () => {
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux2-klein:q4")]);
    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.hosts.value.map((h) => h.id)).toEqual([ORIGIN_HOST_ID]);
    expect(routing.multiHost.value).toBe(false);
  });

  it("reports a remote host as ready with its live queue depth and GPU", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status({ queue_depth: 2 }));
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(
      studio.id,
      status({
        queue_depth: 5,
        gpu_info: {
          backend: "cuda",
          name: "NVIDIA RTX 4090",
          vram_total_mb: 24576,
          vram_used_mb: 0,
        },
      }),
    );
    models.set(studio.id, [model("flux-dev:q4")]);

    const routing = useHostRouting();
    await routing.refresh();

    const remote = routing.hosts.value.find((h) => h.id === studio.id);
    expect(remote?.status).toBe("ready");
    expect(remote?.queueDepth).toBe(5);
    expect(remote?.gpu).toEqual({
      backend: "cuda",
      name: "NVIDIA RTX 4090",
      vramTotalMb: 24576,
    });
    expect(routing.multiHost.value).toBe(true);
  });

  it("marks an unreachable host errored without dropping it from the list", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux2-klein:q4")]);

    const routing = useHostRouting();
    await routing.refresh();

    const remote = routing.hosts.value.find((h) => h.id === studio.id);
    expect(remote?.status).toBe("error");
    // Auto never routes to it.
    expect(routing.resolve("flux2-klein:q4")?.hostId).toBe(ORIGIN_HOST_ID);
  });

  it("carries the remote's api key into the resolved route, never a URL", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "sk-studio",
    });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(studio.id, status());
    models.set(studio.id, [model("flux-dev:q4")]);
    setGenerateTargetId(studio.id);

    const routing = useHostRouting();
    await routing.refresh();

    const route = routing.resolve("flux-dev:q4");
    expect(route?.target).toEqual({
      baseUrl: "http://studio:7680",
      apiKey: "sk-studio",
    });
    expect(route?.target.baseUrl).not.toContain("sk-studio");
  });

  it("shows only the pinned host's models when a host is pinned", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux2-klein:q4")]);
    statuses.set(studio.id, status());
    models.set(studio.id, [model("z-image:bf16")]);
    setGenerateTargetId(studio.id);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.targetModels.value.map((m) => m.name)).toEqual([
      "z-image:bf16",
    ]);
  });

  it("shows the union of ready hosts' models under Auto", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux2-klein:q4")]);
    statuses.set(studio.id, status());
    models.set(studio.id, [model("z-image:bf16")]);
    setGenerateTargetId(AUTO_TARGET_ID);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.targetModels.value.map((m) => m.name).sort()).toEqual([
      "flux2-klein:q4",
      "z-image:bf16",
    ]);
  });

  it("excludes an unreachable host's stale models from the Auto union", async () => {
    addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux2-klein:q4")]);
    setGenerateTargetId(AUTO_TARGET_ID);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.targetModels.value.map((m) => m.name)).toEqual([
      "flux2-klein:q4",
    ]);
  });

  it("gates the empty state until every listed host's models settle", async () => {
    addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, []);

    const routing = useHostRouting();
    expect(routing.modelsSettled.value).toBe(false);
    await routing.refresh();
    // Both hosts answered (one with a rejection) — the gate opens.
    expect(routing.modelsSettled.value).toBe(true);
  });

  it("routes Auto to the host that already holds the model", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    // The origin is idle but lacks the weights; the busy remote has them.
    statuses.set(ORIGIN_HOST_ID, status({ queue_depth: 0 }));
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(studio.id, status({ queue_depth: 6 }));
    models.set(studio.id, [model("z-image:bf16")]);
    setGenerateTargetId(AUTO_TARGET_ID);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.resolve("z-image:bf16")?.hostId).toBe(studio.id);
  });

  it("routes Most capable to the strongest GPU", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(
      ORIGIN_HOST_ID,
      status({
        gpu_info: {
          backend: "metal",
          name: "Apple M3 Max",
          vram_total_mb: 65536,
          vram_used_mb: 0,
        },
      }),
    );
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    statuses.set(
      studio.id,
      status({
        gpu_info: {
          backend: "cuda",
          name: "NVIDIA RTX 4090",
          vram_total_mb: 24576,
          vram_used_mb: 0,
        },
      }),
    );
    models.set(studio.id, [model("flux-dev:q4")]);
    setGenerateTargetId(CAPABLE_TARGET_ID);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.resolve("flux-dev:q4")?.hostId).toBe(studio.id);
  });

  it("ranks a multi-GPU host by its strongest usable device, not gpu_info", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(
      ORIGIN_HOST_ID,
      status({
        // Legacy gpu_info is only GPU 0. It must not hide the 80 GB device.
        gpu_info: {
          backend: "cuda",
          name: "NVIDIA RTX A2000",
          vram_total_mb: 12288,
          vram_used_mb: 0,
        },
        gpus: [
          {
            ordinal: 0,
            name: "NVIDIA RTX A2000",
            vram_total_bytes: 12 * 1024 ** 3,
            vram_used_bytes: 0,
            state: "idle",
          },
          {
            ordinal: 1,
            name: "NVIDIA B200",
            vram_total_bytes: 80 * 1024 ** 3,
            vram_used_bytes: 0,
            state: "idle",
          },
        ],
      }),
    );
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    statuses.set(
      studio.id,
      status({
        gpu_info: {
          backend: "cuda",
          name: "NVIDIA RTX 4090",
          vram_total_mb: 24576,
          vram_used_mb: 0,
        },
      }),
    );
    models.set(studio.id, [model("flux-dev:q4")]);
    setGenerateTargetId(CAPABLE_TARGET_ID);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.hosts.value[0]?.gpu).toEqual({
      backend: "cuda",
      name: "NVIDIA B200",
      vramTotalMb: 80 * 1024,
    });
    expect(routing.resolve("flux-dev:q4")?.hostId).toBe(ORIGIN_HOST_ID);
  });

  it("does not route to a modern host that reports zero routable GPU workers", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(
      ORIGIN_HOST_ID,
      status({
        gpu_info: null,
        gpus: [],
        queue_depth: 0,
      }),
    );
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    statuses.set(
      studio.id,
      status({
        gpu_info: {
          backend: "cuda",
          name: "NVIDIA RTX 4090",
          vram_total_mb: 24576,
          vram_used_mb: 0,
        },
      }),
    );
    models.set(studio.id, [model("flux-dev:q4")]);
    setGenerateTargetId(AUTO_TARGET_ID);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.hosts.value[0]?.gpu).toBeNull();
    expect(routing.resolve("flux-dev:q4")?.hostId).toBe(studio.id);
  });

  it("uses /api/devices schedulability instead of misleading legacy worker rows", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(
      ORIGIN_HOST_ID,
      status({
        gpu_info: {
          backend: "cuda",
          name: "Excluded B200",
          vram_total_mb: 196608,
          vram_used_mb: 0,
        },
        gpus: [
          {
            ordinal: 0,
            name: "Excluded B200",
            vram_total_bytes: 192 * 1024 ** 3,
            vram_used_bytes: 0,
            state: "idle",
          },
        ],
      }),
    );
    devices.set(ORIGIN_HOST_ID, {
      plan_version: 1,
      devices: [
        device(0, {
          name: "Excluded B200",
          admin_state: "startup_excluded",
          desired_enabled: false,
          schedulable: false,
          unschedulable_reason: "excluded by MOLD_GPUS",
        }),
      ],
    });
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);

    statuses.set(studio.id, status());
    devices.set(studio.id, {
      plan_version: 1,
      devices: [device(1, { name: "RTX 3090" })],
    });
    models.set(studio.id, [model("flux-dev:q4")]);
    setGenerateTargetId(AUTO_TARGET_ID);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.hosts.value[0]?.status).toBe("error");
    expect(routing.hosts.value[0]?.gpu).toBeNull();
    expect(routing.resolve("flux-dev:q4")?.hostId).toBe(studio.id);
  });

  it("reads a forgotten sticky pick as Auto", async () => {
    setGenerateTargetId("ghost-host-7680");
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux2-klein:q4")]);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.targetId.value).toBe(AUTO_TARGET_ID);
    expect(routing.resolve("flux2-klein:q4")?.hostId).toBe(ORIGIN_HOST_ID);
  });

  it("refuses to reroute a pinned host that went offline", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    setGenerateTargetId(studio.id);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.targetId.value).toBe(studio.id);
    expect(routing.resolve("flux-dev:q4")).toBeNull();
  });

  it("persists a new pick", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(studio.id, status());
    models.set(studio.id, []);

    const routing = useHostRouting();
    await routing.refresh();
    routing.setTarget(CAPABLE_TARGET_ID);

    expect(routing.targetId.value).toBe(CAPABLE_TARGET_ID);
    expect(localStorage.getItem("mold.web.generateTarget.v1")).toBe(
      CAPABLE_TARGET_ID,
    );
  });
});
