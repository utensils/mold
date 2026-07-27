import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { __testing__, useHostRouting } from "./useHostRouting";
import {
  addHost,
  ORIGIN_HOST_ID,
  setGenerateTargetId,
  updateHost,
} from "../lib/hostRegistry";
import { AUTO_TARGET_ID, CAPABLE_TARGET_ID } from "../lib/hostRouting";
import type { HostEntry } from "../lib/hostRegistry";
import type { ChainRequestWire, ModelInfoExtended } from "../types";
import type { DeviceInfo, DeviceListResponse } from "@studio/api/devices";
import type { GenerationPlacementPreview } from "@studio/api/generationPlacement";
import { ApiError } from "@studio/api/client";

/** Per-host canned `/api/status` + `/api/models` responses, keyed by host id. */
const statuses = new Map<string, unknown>();
const models = new Map<string, ModelInfoExtended[]>();
const devices = new Map<string, DeviceListResponse>();
const hostStatusCall = vi.hoisted(() => vi.fn());
const placementCall = vi.hoisted(() => vi.fn());

vi.mock("@studio/api/generationPlacement", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("@studio/api/generationPlacement")>();
  return {
    ...original,
    previewGenerationPlacement: (...args: unknown[]) => placementCall(...args),
    previewChainPlacement: (...args: unknown[]) => placementCall(...args),
  };
});

vi.mock("../components/machines/hostClient", () => ({
  hostStatus: (...args: unknown[]) => hostStatusCall(...args),
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
  hostQueue: () => Promise.resolve({ entries: [], plan: null }),
}));

function placement(
  completionMs: number,
  overrides: Partial<GenerationPlacementPreview> = {},
): GenerationPlacementPreview {
  return {
    version: 1,
    authoritative: true,
    state_version: 1,
    plan_version: 1,
    outcome: "planned",
    candidate: {
      device_id: "cuda:0",
      execution_fingerprint: "exec",
      predicted_start_after_ms: 0,
      predicted_completion_after_ms: completionMs,
      setup_ms: 0,
      setup_kind: "warm",
      estimate_confidence: "high",
    },
    ...overrides,
  };
}

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
    hostStatusCall.mockReset().mockImplementation((host: HostEntry) => {
      const canned = statuses.get(host.id);
      return canned
        ? Promise.resolve(canned)
        : Promise.reject(new Error("unreachable"));
    });
    placementCall
      .mockReset()
      .mockImplementation((target: { baseUrl: string }) =>
        Promise.resolve(
          placement(target.baseUrl.includes("studio") ? 200 : 100),
        ),
      );
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

  it("ignores an older same-host poll after a newer refresh settles", async () => {
    statuses.set(ORIGIN_HOST_ID, status({ queue_depth: 0 }));
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    devices.set(ORIGIN_HOST_ID, {
      plan_version: 1,
      devices: [device(0)],
    });
    const routing = useHostRouting();
    await routing.refresh();

    let resolveOlder!: (value: unknown) => void;
    let resolveNewer!: (value: unknown) => void;
    const older = new Promise((resolve) => {
      resolveOlder = resolve;
    });
    const newer = new Promise((resolve) => {
      resolveNewer = resolve;
    });
    hostStatusCall
      .mockImplementationOnce(() => older)
      .mockImplementationOnce(() => newer);

    const first = routing.refresh();
    const second = routing.refresh();
    resolveNewer(status({ queue_depth: 9 }));
    await second;
    resolveOlder(status({ queue_depth: 1 }));
    await first;

    expect(routing.hosts.value[0]?.queueDepth).toBe(9);
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

  it("routes only to a host whose exact request is scheduler-feasible", async () => {
    setGenerateTargetId(AUTO_TARGET_ID);
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    for (const id of [ORIGIN_HOST_ID, studio.id]) {
      statuses.set(id, status({ queue_depth: id === ORIGIN_HOST_ID ? 0 : 2 }));
      models.set(id, [model("ltx2.3-dev:bf16")]);
      devices.set(id, { plan_version: 1, devices: [device(0)] });
    }
    placementCall.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(
        target.baseUrl.includes("studio")
          ? placement(200)
          : placement(0, {
              outcome: "infeasible",
              candidate: null,
              reason: "required VAE is absent",
            }),
      ),
    );
    const routing = useHostRouting();
    await routing.refresh();

    const route = await routing.resolveFeasible({
      prompt: "shot",
      model: "ltx2.3-dev:bf16",
      width: 768,
      height: 512,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });
    expect(route?.hostId).toBe(studio.id);
  });

  it("uses scheduler completion time rather than current free memory", async () => {
    setGenerateTargetId(AUTO_TARGET_ID);
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    for (const id of [ORIGIN_HOST_ID, studio.id]) {
      statuses.set(id, status({ queue_depth: id === ORIGIN_HOST_ID ? 0 : 2 }));
      models.set(id, [model("flux-dev:q4")]);
      devices.set(id, { plan_version: 1, devices: [device(0)] });
    }
    placementCall.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(placement(target.baseUrl.includes("studio") ? 300 : 50)),
    );
    const routing = useHostRouting();
    await routing.refresh();

    const route = await routing.resolveFeasible({
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });
    expect(route?.hostId).toBe(ORIGIN_HOST_ID);
    expect(placementCall).toHaveBeenCalledTimes(2);
  });

  it("previews Batch N as N one-image siblings without mutating the reviewed request", async () => {
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    const routing = useHostRouting();
    await routing.refresh();
    const request = {
      prompt: "reviewed",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 4,
    };

    await expect(routing.resolveFeasible(request, 4)).resolves.toBeTruthy();

    expect(placementCall).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ batch_size: 1 }),
      4,
    );
    expect(request.batch_size).toBe(4);
  });

  it.each([401, 403, 426, 500])(
    "treats placement HTTP %s as definitive rather than legacy fallback",
    async (statusCode) => {
      statuses.set(ORIGIN_HOST_ID, status());
      models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
      placementCall.mockRejectedValue(
        new ApiError("placement failed", statusCode),
      );
      const routing = useHostRouting();
      await routing.refresh();

      await expect(
        routing.resolveFeasible({
          prompt: "print",
          model: "flux-dev:q4",
          width: 1024,
          height: 1024,
          steps: 20,
          guidance: 3.5,
          seed: null,
          batch_size: 1,
        }),
      ).resolves.toBeNull();
    },
  );

  it.each([404, 405])(
    "uses the exact selected legacy host when placement HTTP %s is unsupported",
    async (statusCode) => {
      const studio = addHost({
        url: "http://studio:7680",
        name: "Studio",
        instanceId: "instance-a",
      });
      setGenerateTargetId(studio.id);
      statuses.set(studio.id, status());
      models.set(studio.id, [model("ltx2.3-dev:bf16")]);
      placementCall.mockRejectedValue(
        new ApiError("not supported", statusCode),
      );
      const routing = useHostRouting();
      await routing.refresh();

      const route = await routing.resolveFeasibleChain({
        model: "ltx2.3-dev:bf16",
        width: 768,
        height: 512,
        fps: 24,
        steps: 20,
        guidance: 3.5,
        motion_tail_frames: 17,
        output_format: "mp4",
        stages: [
          { prompt: "one", frames: 97 },
          { prompt: "two", frames: 97 },
        ],
      });

      expect(route).toMatchObject({
        hostId: studio.id,
        instanceId: "instance-a",
        target: { baseUrl: studio.url },
      });
    },
  );

  it("revalidates only the frozen host when another host becomes faster", async () => {
    setGenerateTargetId(AUTO_TARGET_ID);
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    for (const id of [ORIGIN_HOST_ID, studio.id]) {
      statuses.set(id, status());
      models.set(id, [model("flux-dev:q4")]);
      devices.set(id, { plan_version: 1, devices: [device(0)] });
    }
    const request = {
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    };
    placementCall.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(placement(target.baseUrl.includes("studio") ? 200 : 50)),
    );
    const routing = useHostRouting();
    await routing.refresh();
    const frozen = await routing.resolveFeasible(request);
    expect(frozen?.hostId).toBe(ORIGIN_HOST_ID);

    placementCall.mockClear();
    placementCall.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(placement(target.baseUrl.includes("studio") ? 10 : 500)),
    );
    const revalidated = await routing.revalidateFeasible(frozen!, request);

    expect(revalidated?.hostId).toBe(ORIGIN_HOST_ID);
    expect(placementCall).toHaveBeenCalledTimes(1);
    expect(placementCall.mock.calls[0]?.[0]).toMatchObject({
      baseUrl: window.location.origin,
    });
  });

  it.each(["explicit unsupported", "HTTP 404", "HTTP 405"])(
    "keeps the exact frozen chain route during %s revalidation",
    async (kind) => {
      const studio = addHost({
        url: "http://studio:7680",
        name: "Studio",
        apiKey: "same-key",
        instanceId: "instance-a",
      });
      setGenerateTargetId(studio.id);
      statuses.set(studio.id, status());
      models.set(studio.id, [model("ltx2.3-dev:bf16")]);
      devices.set(studio.id, { plan_version: 1, devices: [device(0)] });
      const routing = useHostRouting();
      await routing.refresh();
      const request: ChainRequestWire = {
        model: "ltx2.3-dev:bf16",
        width: 768,
        height: 512,
        fps: 24,
        steps: 20,
        guidance: 3.5,
        motion_tail_frames: 17,
        output_format: "mp4",
        stages: [
          { prompt: "one", frames: 97 },
          { prompt: "two", frames: 97 },
        ],
      };
      placementCall.mockResolvedValueOnce(
        placement(0, {
          authoritative: false,
          outcome: "unsupported",
          candidate: null,
        }),
      );
      const frozen = await routing.resolveFeasibleChain(request);
      expect(frozen).toMatchObject({
        hostId: studio.id,
        instanceId: "instance-a",
      });

      if (kind === "explicit unsupported") {
        placementCall.mockResolvedValueOnce(
          placement(0, {
            authoritative: false,
            outcome: "unsupported",
            candidate: null,
          }),
        );
      } else {
        placementCall.mockRejectedValueOnce(
          new ApiError("unsupported", kind === "HTTP 404" ? 404 : 405),
        );
      }

      await expect(
        routing.revalidateFeasibleChain(frozen!, request, 3),
      ).resolves.toMatchObject({
        hostId: studio.id,
        instanceId: "instance-a",
        target: { baseUrl: studio.url, apiKey: "same-key" },
      });
    },
  );

  it("rejects frozen-host revalidation when the same URL and key report a new instance", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "same-key",
      instanceId: "instance-a",
    });
    setGenerateTargetId(studio.id);
    statuses.set(studio.id, status());
    models.set(studio.id, [model("flux-dev:q4")]);
    devices.set(studio.id, { plan_version: 1, devices: [device(0)] });
    const request = {
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    };
    const routing = useHostRouting();
    await routing.refresh();
    const frozen = await routing.resolveFeasible(request);
    expect(frozen).toMatchObject({
      hostId: studio.id,
      instanceId: "instance-a",
    });

    let release!: (value: GenerationPlacementPreview) => void;
    placementCall.mockImplementationOnce(
      () =>
        new Promise<GenerationPlacementPreview>((resolve) => {
          release = resolve;
        }),
    );
    const pending = routing.revalidateFeasible(frozen!, request, 4);
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledTimes(2));
    updateHost(studio.id, { instanceId: "instance-b" });
    release(placement(100));

    await expect(pending).resolves.toBeNull();
  });

  it("rejects frozen origin revalidation after the serving origin is replaced", async () => {
    statuses.set(ORIGIN_HOST_ID, status({ instance_id: "origin-instance-a" }));
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    devices.set(ORIGIN_HOST_ID, {
      plan_version: 1,
      devices: [device(0)],
    });
    const request = {
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    };
    const routing = useHostRouting();
    await routing.refresh();
    const frozen = await routing.resolveFeasible(request);
    expect(frozen).toMatchObject({
      hostId: ORIGIN_HOST_ID,
      instanceId: "origin-instance-a",
    });

    let release!: (value: GenerationPlacementPreview) => void;
    placementCall.mockImplementationOnce(
      () =>
        new Promise<GenerationPlacementPreview>((resolve) => {
          release = resolve;
        }),
    );
    const pending = routing.revalidateFeasible(frozen!, request, 4);
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledTimes(2));
    statuses.set(ORIGIN_HOST_ID, status({ instance_id: "origin-instance-b" }));
    await routing.refresh();
    release(placement(100));

    await expect(pending).resolves.toBeNull();
  });

  it("rejects a feasibility result when the target changes while probes are pending", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    setGenerateTargetId(studio.id);
    for (const id of [ORIGIN_HOST_ID, studio.id]) {
      statuses.set(id, status());
      models.set(id, [model("flux-dev:q4")]);
      devices.set(id, { plan_version: 1, devices: [device(0)] });
    }
    const routing = useHostRouting();
    await routing.refresh();

    let release!: (value: GenerationPlacementPreview) => void;
    placementCall.mockImplementationOnce(
      () =>
        new Promise<GenerationPlacementPreview>((resolve) => {
          release = resolve;
        }),
    );
    const pending = routing.resolveFeasible({
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledOnce());
    routing.setTarget(ORIGIN_HOST_ID);
    release(placement(100));

    await expect(pending).resolves.toBeNull();
  });

  it("rejects a feasibility result when host credentials change during probes", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      apiKey: "old-key",
    });
    setGenerateTargetId(studio.id);
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(studio.id, status());
    models.set(studio.id, [model("flux-dev:q4")]);
    devices.set(studio.id, { plan_version: 1, devices: [device(0)] });
    const routing = useHostRouting();
    await routing.refresh();

    let release!: (value: GenerationPlacementPreview) => void;
    placementCall.mockImplementationOnce(
      () =>
        new Promise<GenerationPlacementPreview>((resolve) => {
          release = resolve;
        }),
    );
    const pending = routing.resolveFeasible({
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledOnce());
    updateHost(studio.id, { apiKey: "rotated-key" });
    release(placement(100));

    await expect(pending).resolves.toBeNull();
  });

  it("rejects a feasibility result when the host URL changes during probes", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
    });
    setGenerateTargetId(studio.id);
    statuses.set(studio.id, status());
    models.set(studio.id, [model("flux-dev:q4")]);
    devices.set(studio.id, { plan_version: 1, devices: [device(0)] });
    const routing = useHostRouting();
    await routing.refresh();

    let release!: (value: GenerationPlacementPreview) => void;
    placementCall.mockImplementationOnce(
      () =>
        new Promise<GenerationPlacementPreview>((resolve) => {
          release = resolve;
        }),
    );
    const pending = routing.resolveFeasible({
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledOnce());
    updateHost(studio.id, { url: "http://replacement:7680" });
    release(placement(100));

    await expect(pending).resolves.toBeNull();
  });

  it("rejects a feasibility result when host instance identity changes during probes", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      instanceId: "instance-a",
    });
    setGenerateTargetId(studio.id);
    statuses.set(studio.id, status());
    models.set(studio.id, [model("flux-dev:q4")]);
    devices.set(studio.id, { plan_version: 1, devices: [device(0)] });
    const routing = useHostRouting();
    await routing.refresh();

    let release!: (value: GenerationPlacementPreview) => void;
    placementCall.mockImplementationOnce(
      () =>
        new Promise<GenerationPlacementPreview>((resolve) => {
          release = resolve;
        }),
    );
    const pending = routing.resolveFeasible({
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledOnce());
    updateHost(studio.id, { instanceId: "instance-b" });
    release(placement(100));

    await expect(pending).resolves.toBeNull();
  });

  it("never reroutes an explicitly selected host after authoritative infeasibility", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    setGenerateTargetId(studio.id);
    for (const id of [ORIGIN_HOST_ID, studio.id]) {
      statuses.set(id, status());
      models.set(id, [model("flux-dev:q4")]);
      devices.set(id, { plan_version: 1, devices: [device(0)] });
    }
    placementCall.mockResolvedValue(
      placement(0, {
        outcome: "infeasible",
        candidate: null,
        reason: "insufficient_vram",
      }),
    );
    const routing = useHostRouting();
    await routing.refresh();

    const route = await routing.resolveFeasible({
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });

    expect(route).toBeNull();
    expect(placementCall).toHaveBeenCalledTimes(1);
    expect(placementCall.mock.calls[0]?.[0]).toMatchObject({
      baseUrl: studio.url,
    });
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
