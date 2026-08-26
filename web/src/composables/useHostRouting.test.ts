import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __testing__,
  useHostRouting,
  type FeasibilityResult,
} from "./useHostRouting";
import {
  addHost,
  HOSTS_STORAGE_KEY,
  ORIGIN_HOST_ID,
  setGenerateTargetId,
  updateHost,
} from "../lib/hostRegistry";
import { AUTO_TARGET_ID, CAPABLE_TARGET_ID } from "../lib/hostRouting";
import type { HostEntry } from "../lib/hostRegistry";
import type {
  ChainRequestWire,
  ModelInfoExtended,
  ServerCapabilities,
} from "../types";
import type { DeviceInfo, DeviceListResponse } from "@studio/api/devices";
import type { GenerationPlacementPreview } from "@studio/api/generationPlacement";
import { ApiError } from "@studio/api/client";
import { ApiHttpError } from "../api";
import { queueStatusFor } from "@studio/lib/queuePosition";
import {
  AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256,
  authenticatedMiniMaxH3Capabilities,
} from "@studio/lib/minimaxH3Inventory.testFixtures";

/** Per-host canned `/api/status` + `/api/models` responses, keyed by host id. */
const statuses = new Map<string, unknown>();
const models = new Map<string, ModelInfoExtended[] | Error>();
const devices = new Map<string, DeviceListResponse | Error>();
const capabilities = new Map<string, ServerCapabilities | Error>();
const hostStatusCall = vi.hoisted(() => vi.fn());
const hostQueueCall = vi.hoisted(() => vi.fn());
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
    return canned instanceof Error
      ? Promise.reject(canned)
      : canned
        ? Promise.resolve(canned)
        : Promise.reject(new Error("unreachable"));
  },
  hostDevices: (host: HostEntry) => {
    const canned = devices.get(host.id);
    return canned instanceof Error
      ? Promise.reject(canned)
      : canned
        ? Promise.resolve(canned)
        : Promise.reject(new Error("legacy server"));
  },
  hostQueue: (...args: unknown[]) => hostQueueCall(...args),
  hostCapabilities: (host: HostEntry) => {
    const canned = capabilities.get(host.id);
    return canned instanceof Error
      ? Promise.reject(canned)
      : Promise.resolve(canned ?? { gallery: { can_delete: true } });
  },
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

function routeOf(result: FeasibilityResult) {
  expect(result.kind).toBe("route");
  if (result.kind !== "route")
    throw new Error(`expected route, got ${result.kind}`);
  return result.route;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

describe("useHostRouting", () => {
  beforeEach(() => {
    localStorage.clear();
    statuses.clear();
    models.clear();
    devices.clear();
    capabilities.clear();
    hostStatusCall.mockReset().mockImplementation((host: HostEntry) => {
      const canned = statuses.get(host.id);
      return canned
        ? Promise.resolve(canned)
        : Promise.reject(new Error("unreachable"));
    });
    hostQueueCall.mockReset().mockResolvedValue({ entries: [], plan: null });
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

  it("bounds the first queue read by current status capacity and merges live-only rows once", async () => {
    statuses.set(ORIGIN_HOST_ID, status({ queue_capacity: 17 }));
    models.set(ORIGIN_HOST_ID, []);
    hostQueueCall.mockResolvedValue({
      entries: [
        {
          id: "durable",
          model: "flux-dev:q4",
          state: "queued",
          started_at_unix_ms: 1,
          position: 4,
        },
      ],
      live_only_entries: [
        {
          id: "durable",
          model: "flux-dev:q4",
          state: "running",
          started_at_unix_ms: 1,
          position: 9,
        },
        {
          id: "live",
          model: "flux-dev:q4",
          state: "running",
          started_at_unix_ms: 2,
          position: 1,
        },
        {
          id: "live",
          model: "flux-dev:q4",
          state: "running",
          started_at_unix_ms: 2,
          position: 8,
        },
      ],
      plan: null,
    });

    const routing = useHostRouting();
    await routing.refresh();

    expect(hostQueueCall).toHaveBeenCalledWith(
      expect.objectContaining({ id: ORIGIN_HOST_ID }),
      undefined,
      { limit: 17 },
    );
    expect(
      hostQueueCall.mock.calls.every(
        ([, signal, page]) =>
          signal === undefined &&
          (page as { limit?: number } | undefined)?.limit === 17,
      ),
    ).toBe(true);
    expect(
      queueStatusFor(routing.queueStatus.value, ORIGIN_HOST_ID, "durable"),
    ).toMatchObject({ position: 4 });
    expect(
      queueStatusFor(routing.queueStatus.value, ORIGIN_HOST_ID, "live"),
    ).toMatchObject({ position: 1 });
  });

  it.each([undefined, null, 0, -1, 1.5, Number.NaN])(
    "keeps the legacy queue read when status capacity is %s",
    async (queueCapacity) => {
      statuses.set(ORIGIN_HOST_ID, status({ queue_capacity: queueCapacity }));
      models.set(ORIGIN_HOST_ID, []);

      await useHostRouting().refresh();

      expect(hostQueueCall).toHaveBeenCalledWith(
        expect.objectContaining({ id: ORIGIN_HOST_ID }),
      );
      expect(hostQueueCall.mock.calls.every((call) => call.length === 1)).toBe(
        true,
      );
    },
  );

  it("keeps a healthy status and the last good queue when its bounded queue page fails", async () => {
    statuses.set(ORIGIN_HOST_ID, status({ queue_depth: 1, queue_capacity: 8 }));
    models.set(ORIGIN_HOST_ID, []);
    hostQueueCall.mockResolvedValue({
      entries: [
        {
          id: "still-live",
          model: "flux-dev:q4",
          state: "queued",
          started_at_unix_ms: 1,
          position: 2,
        },
      ],
      plan: null,
    });
    const routing = useHostRouting();
    await routing.refresh();

    statuses.set(ORIGIN_HOST_ID, status({ queue_depth: 2, queue_capacity: 8 }));
    hostQueueCall.mockRejectedValue(new Error("queue unavailable"));
    await routing.refresh();

    expect(routing.hosts.value[0]).toMatchObject({
      status: "ready",
      queueDepth: 2,
    });
    expect(
      queueStatusFor(routing.queueStatus.value, ORIGIN_HOST_ID, "still-live"),
    ).toMatchObject({ position: 2 });
  });

  it("reacts to same-tab host and generation-target changes", async () => {
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux2-klein:q4")]);
    const routing = useHostRouting();
    await routing.refresh();

    const remoteId = "studio-local-7680";
    statuses.set(remoteId, status());
    models.set(remoteId, [model("flux-dev:q8")]);
    const remote = addHost({
      url: "http://studio.local:7680",
      name: "Studio",
    });
    expect(remote.id).toBe(remoteId);

    await vi.waitFor(() => {
      expect(routing.hosts.value.map((host) => host.id)).toContain(remote.id);
    });

    setGenerateTargetId(remote.id);
    expect(routing.targetId.value).toBe(remote.id);
    await vi.waitFor(() => {
      expect(routing.targetModels.value.map((entry) => entry.name)).toEqual([
        "flux-dev:q8",
      ]);
    });
  });

  it("removes advertised restricted rows and refuses routing before placement", async () => {
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [
      model("flux-dev:q8"),
      model("hf:MiniMaxAI/MiniMaxH3", { family: "minimax-h3" }),
    ]);
    capabilities.set(ORIGIN_HOST_ID, {
      gallery: { can_delete: true },
      model_access: {
        restrictions: [
          {
            code: "minimax_h3_authorization_required",
            family: "minimax-h3",
            message: "MiniMax H3 is not activated.",
            license_url: "https://example.test/license",
            authorization_url: "https://example.test/authorize",
          },
        ],
      },
    });
    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.targetModels.value.map((entry) => entry.name)).toEqual([
      "flux-dev:q8",
    ]);
    expect(routing.modelOwnerIds("hf:MiniMaxAI/MiniMaxH3")).toEqual([]);
    expect(routing.resolve("hf:MiniMaxAI/MiniMaxH3")).toBeNull();
    placementCall.mockClear();
    await expect(
      routing.resolveFeasible({
        prompt: "blocked",
        model: "hf:MiniMaxAI/MiniMaxH3",
        width: 768,
        height: 512,
        steps: 20,
        guidance: 3.5,
        seed: null,
        batch_size: 1,
      }),
    ).resolves.toEqual({
      kind: "infeasible",
      perHost: [
        {
          hostId: ORIGIN_HOST_ID,
          label: "this server",
          reason: "MiniMax H3 is not activated.",
          missingComponents: [],
          missingModel: null,
        },
      ],
    });
    expect(placementCall).not.toHaveBeenCalled();
  });

  it("routes the exact authenticated private FL2VA model row", async () => {
    const h3 = "minimax-h3-fl2va:comfy-pruned-int8";
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [
      model(h3, {
        family: "minimax-h3",
        generation_profile: {
          profile_hash: AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256,
        } as never,
      }),
    ]);
    capabilities.set(ORIGIN_HOST_ID, {
      gallery: { can_delete: true },
      ...authenticatedMiniMaxH3Capabilities(),
    } as ServerCapabilities);
    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.targetModels.value.map((entry) => entry.name)).toEqual([h3]);
    expect(routing.modelOwnerIds(h3)).toEqual([ORIGIN_HOST_ID]);
    expect(routing.resolve(h3)?.hostId).toBe(ORIGIN_HOST_ID);
  });

  it.each(
    ["missing", "unbound", "mismatched"].flatMap((rowKind) =>
      [ORIGIN_HOST_ID, AUTO_TARGET_ID].map(
        (target) => [rowKind, target] as const,
      ),
    ),
  )(
    "keeps H3 closed for an exact capability with a %s model row on target %s",
    async (rowKind, target) => {
      const h3 = "minimax-h3-fl2va:comfy-pruned-int8";
      statuses.set(ORIGIN_HOST_ID, status());
      models.set(
        ORIGIN_HOST_ID,
        rowKind === "missing"
          ? []
          : [
              model(h3, {
                family: "minimax-h3",
                ...(rowKind === "mismatched"
                  ? {
                      generation_profile: {
                        profile_hash: "b".repeat(64),
                      } as never,
                    }
                  : {}),
              }),
            ],
      );
      capabilities.set(ORIGIN_HOST_ID, {
        gallery: { can_delete: true },
        ...authenticatedMiniMaxH3Capabilities(),
      } as ServerCapabilities);
      setGenerateTargetId(target);
      const routing = useHostRouting();
      await routing.refresh();

      expect(routing.targetModels.value).toEqual([]);
      expect(routing.modelOwnerIds(h3)).toEqual([]);
      expect(routing.resolve(h3)).toBeNull();
      await expect(
        routing.resolveFeasible({
          prompt: "blocked",
          model: h3,
          width: 1344,
          height: 768,
          steps: 21,
          guidance: 0,
          seed: 42,
          batch_size: 1,
          frames: 124,
          fps: 24,
        }),
      ).resolves.toMatchObject({ kind: "infeasible" });
      expect(placementCall).not.toHaveBeenCalled();
    },
  );

  it("coalesces concurrent refresh callers into one same-host poll", async () => {
    statuses.set(ORIGIN_HOST_ID, status({ queue_depth: 0 }));
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    devices.set(ORIGIN_HOST_ID, {
      plan_version: 1,
      devices: [device(0)],
    });
    const routing = useHostRouting();
    await routing.refresh();
    hostStatusCall.mockClear();
    hostQueueCall.mockClear();

    const current = deferred<ReturnType<typeof status>>();
    hostStatusCall.mockImplementationOnce(() => current.promise);
    hostQueueCall.mockResolvedValueOnce({
      entries: [
        {
          id: "current-queue",
          model: "flux-dev:q4",
          state: "queued",
          started_at_unix_ms: 1,
          position: 1,
        },
      ],
      plan: null,
    });

    const first = routing.refresh();
    const second = routing.refresh();
    expect(hostStatusCall).toHaveBeenCalledTimes(1);
    current.resolve(status({ queue_depth: 1 }));
    await Promise.all([first, second]);

    expect(routing.hosts.value[0]?.queueDepth).toBe(1);
    expect(
      queueStatusFor(
        routing.queueStatus.value,
        ORIGIN_HOST_ID,
        "current-queue",
      ),
    ).toMatchObject({ position: 1 });
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
    expect(routeOf(route).hostId).toBe(studio.id);
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
    expect(routeOf(route).hostId).toBe(ORIGIN_HOST_ID);
    expect(placementCall).toHaveBeenCalledTimes(2);
  });

  it("routes to a clean host instead of a faster host with pending downloads", async () => {
    setGenerateTargetId(AUTO_TARGET_ID);
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    for (const id of [ORIGIN_HOST_ID, studio.id]) {
      statuses.set(id, status());
      models.set(id, [model("flux-dev:q4")]);
      devices.set(id, { plan_version: 1, devices: [device(0)] });
    }
    placementCall.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(
        target.baseUrl.includes("studio")
          ? placement(10_000)
          : placement(10, {
              pending_downloads: [
                {
                  kind: "text_encoder",
                  name: "t5-v1_1-xxl-q8.gguf",
                  repo: "acme/text-encoders",
                  bytes: 5_100_000_000,
                },
              ],
            }),
      ),
    );
    const routing = useHostRouting();
    await routing.refresh();

    const result = await routing.resolveFeasible({
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });

    expect(routeOf(result).hostId).toBe(studio.id);
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
      {},
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
      ).resolves.toEqual({
        kind: "unreachable",
        perHost: [
          {
            hostId: ORIGIN_HOST_ID,
            label: "this server",
            error: `HTTP ${statusCode}: placement failed`,
          },
        ],
      });
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
        kind: "route",
        route: {
          hostId: studio.id,
          instanceId: "instance-a",
          target: { baseUrl: studio.url },
        },
      });
    },
  );

  it.each([404, 405])(
    "routes staged H3 references to host admission when placement HTTP %s is unsupported",
    async (statusCode) => {
      statuses.set(ORIGIN_HOST_ID, status());
      models.set(ORIGIN_HOST_ID, [
        model("minimax-h3-ref2va:comfy-pruned-int8"),
      ]);
      placementCall.mockRejectedValue(
        new ApiError("not supported", statusCode),
      );
      const routing = useHostRouting();
      await routing.refresh();

      await expect(
        routing.resolveFeasible({
          prompt: "reference print",
          model: "minimax-h3-ref2va:comfy-pruned-int8",
          width: 1344,
          height: 768,
          steps: 8,
          guidance: 0,
          seed: null,
          batch_size: 1,
          references: [],
        }),
      ).resolves.toMatchObject({
        kind: "route",
        route: { hostId: ORIGIN_HOST_ID },
        preview: null,
      });
    },
  );

  it("routes staged H3 references to host admission after an explicit unsupported preview", async () => {
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("minimax-h3-ref2va:comfy-pruned-int8")]);
    placementCall.mockResolvedValue(
      placement(0, {
        authoritative: false,
        outcome: "unsupported",
        candidate: null,
      }),
    );
    const routing = useHostRouting();
    await routing.refresh();

    await expect(
      routing.resolveFeasible({
        prompt: "reference print",
        model: "minimax-h3-ref2va:comfy-pruned-int8",
        width: 1344,
        height: 768,
        steps: 8,
        guidance: 0,
        seed: null,
        batch_size: 1,
        references: [],
      }),
    ).resolves.toMatchObject({
      kind: "route",
      route: { hostId: ORIGIN_HOST_ID },
      preview: null,
    });
  });

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
    const frozen = routeOf(await routing.resolveFeasible(request));
    expect(frozen.hostId).toBe(ORIGIN_HOST_ID);

    placementCall.mockClear();
    placementCall.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(placement(target.baseUrl.includes("studio") ? 10 : 500)),
    );
    const revalidated = await routing.revalidateFeasible(frozen!, request);

    expect(routeOf(revalidated).hostId).toBe(ORIGIN_HOST_ID);
    expect(placementCall).toHaveBeenCalledTimes(1);
    expect(placementCall.mock.calls[0]?.[0]).toMatchObject({
      baseUrl: window.location.origin,
    });
  });

  it("does not retry an old frozen host when Run on changes during revalidation", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    setGenerateTargetId(studio.id);
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
    const routing = useHostRouting();
    await routing.refresh();
    const frozen = routeOf(await routing.resolveFeasible(request));
    placementCall.mockClear();

    let release!: (value: GenerationPlacementPreview) => void;
    placementCall.mockImplementationOnce(
      () =>
        new Promise<GenerationPlacementPreview>((resolve) => {
          release = resolve;
        }),
    );
    const pending = routing.revalidateFeasible(frozen, request);
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledOnce());
    routing.setTarget(ORIGIN_HOST_ID);
    release(placement(100));

    await expect(pending).resolves.toEqual({ kind: "transient", perHost: [] });
    expect(placementCall).toHaveBeenCalledOnce();
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
      const frozen = routeOf(await routing.resolveFeasibleChain(request));
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
        routing.revalidateFeasibleChain(frozen, request, 3),
      ).resolves.toMatchObject({
        kind: "route",
        route: {
          hostId: studio.id,
          instanceId: "instance-a",
          target: { baseUrl: studio.url, apiKey: "same-key" },
        },
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
    const frozen = routeOf(await routing.resolveFeasible(request));
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
    const pending = routing.revalidateFeasible(frozen, request, 4);
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledTimes(2));
    updateHost(studio.id, { instanceId: "instance-b" });
    release(placement(100));

    await expect(pending).resolves.toEqual({ kind: "transient", perHost: [] });
  });

  it("rejects an already-replaced frozen host before issuing a placement probe", async () => {
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
    const frozen = routeOf(await routing.resolveFeasible(request));
    expect(frozen).toMatchObject({
      hostId: studio.id,
      instanceId: "instance-a",
    });

    updateHost(studio.id, { instanceId: "instance-b" });
    placementCall.mockClear();

    await expect(routing.revalidateFeasible(frozen, request)).resolves.toEqual({
      kind: "transient",
      perHost: [],
    });
    expect(placementCall).not.toHaveBeenCalled();
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
    const frozen = routeOf(await routing.resolveFeasible(request));
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
    const pending = routing.revalidateFeasible(frozen, request, 4);
    await vi.waitFor(() => expect(placementCall).toHaveBeenCalledTimes(2));
    statuses.set(ORIGIN_HOST_ID, status({ instance_id: "origin-instance-b" }));
    await routing.refresh();
    release(placement(100));

    await expect(pending).resolves.toEqual({ kind: "transient", perHost: [] });
  });

  it("retries once when the target changes while probes are pending", async () => {
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

    const result = await pending;
    expect(routeOf(result).hostId).toBe(ORIGIN_HOST_ID);
    expect(placementCall).toHaveBeenCalledTimes(2);
  });

  it("retries once with current credentials when they change during probes", async () => {
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

    expect(routeOf(await pending).target).toEqual({
      baseUrl: studio.url,
      apiKey: "rotated-key",
    });
    expect(placementCall).toHaveBeenCalledTimes(2);
  });

  it("retries once with the current URL when it changes during probes", async () => {
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

    expect(routeOf(await pending).target.baseUrl).toBe(
      "http://replacement:7680",
    );
    expect(placementCall).toHaveBeenCalledTimes(2);
  });

  it("retries once with the current instance identity when it changes during probes", async () => {
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
    statuses.set(studio.id, status({ instance_id: "instance-b" }));
    updateHost(studio.id, { instanceId: "instance-b" });
    release(placement(100));

    expect(routeOf(await pending).instanceId).toBe("instance-b");
    expect(placementCall).toHaveBeenCalledTimes(2);
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

    expect(route).toEqual({
      kind: "infeasible",
      perHost: [
        {
          hostId: studio.id,
          label: "Studio",
          reason: "insufficient_vram",
          missingComponents: [],
          missingModel: null,
        },
      ],
    });
    expect(placementCall).toHaveBeenCalledTimes(1);
    expect(placementCall.mock.calls[0]?.[0]).toMatchObject({
      baseUrl: studio.url,
    });
  });

  it("retains each host's authoritative infeasibility reason and components", async () => {
    setGenerateTargetId(AUTO_TARGET_ID);
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    for (const id of [ORIGIN_HOST_ID, studio.id]) {
      statuses.set(id, status());
      models.set(id, [model("flux-dev:q4")]);
      devices.set(id, { plan_version: 1, devices: [device(0)] });
    }
    placementCall.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(
        placement(0, {
          outcome: "infeasible",
          candidate: null,
          reason: target.baseUrl.includes("studio")
            ? "stale device pin cuda:gone"
            : "required VAE is absent",
          missing_components: target.baseUrl.includes("studio")
            ? []
            : [
                {
                  kind: "vae",
                  name: "ae.safetensors",
                  present: false,
                  repair_model: "flux-dev:q4",
                },
              ],
        }),
      ),
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
    ).resolves.toEqual({
      kind: "infeasible",
      perHost: [
        {
          hostId: ORIGIN_HOST_ID,
          label: "this server",
          reason: "required VAE is absent",
          missingComponents: [
            {
              kind: "vae",
              name: "ae.safetensors",
              present: false,
              repair_model: "flux-dev:q4",
            },
          ],
          // The absent component names THIS model as its repair, so pulling
          // it here is the fix — that is a missing-model refusal.
          missingModel: {
            model: "flux-dev:q4",
            missingComponents: [
              {
                kind: "vae",
                name: "ae.safetensors",
                present: false,
                repair_model: "flux-dev:q4",
              },
            ],
          },
        },
        {
          hostId: studio.id,
          label: "Studio",
          reason: "stale device pin cuda:gone",
          missingComponents: [],
          missingModel: null,
        },
      ],
    });
  });

  it("does not discard probe errors when another selected host is infeasible", async () => {
    setGenerateTargetId(AUTO_TARGET_ID);
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    for (const id of [ORIGIN_HOST_ID, studio.id]) {
      statuses.set(id, status());
      models.set(id, [model("flux-dev:q4")]);
      devices.set(id, { plan_version: 1, devices: [device(0)] });
    }
    placementCall.mockImplementation((target: { baseUrl: string }) =>
      target.baseUrl.includes("studio")
        ? Promise.reject(new ApiError("scheduler unavailable", 503))
        : Promise.resolve(
            placement(0, {
              outcome: "infeasible",
              candidate: null,
              reason: "required VAE is absent",
            }),
          ),
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
    ).resolves.toEqual({
      kind: "infeasible",
      perHost: [
        {
          hostId: ORIGIN_HOST_ID,
          label: "this server",
          reason: "required VAE is absent",
          missingComponents: [],
          missingModel: null,
        },
      ],
      unreachable: [
        {
          hostId: studio.id,
          label: "Studio",
          error: "HTTP 503: scheduler unavailable",
        },
      ],
    });
  });

  it("routes the origin through legacy fallback while its first poll is connecting", async () => {
    placementCall.mockResolvedValue(
      placement(0, {
        authoritative: false,
        outcome: "unsupported",
        candidate: null,
      }),
    );
    const routing = useHostRouting();

    const result = await routing.resolveFeasible({
      prompt: "print",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      seed: null,
      batch_size: 1,
    });

    expect(routeOf(result).hostId).toBe(ORIGIN_HOST_ID);
  });

  it("returns scheduler temporary unavailability with the host reason", async () => {
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    placementCall.mockResolvedValue(
      placement(0, {
        authoritative: false,
        outcome: "temporarily_unavailable",
        candidate: null,
        reason: "scheduler snapshot moved",
      }),
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
    ).resolves.toEqual({
      kind: "transient",
      perHost: [
        {
          hostId: ORIGIN_HOST_ID,
          label: "this server",
          reason: "scheduler snapshot moved",
        },
      ],
    });
  });

  it("returns an authoritative paused-queue preview as transient", async () => {
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux-dev:q4")]);
    placementCall.mockResolvedValue(
      placement(0, {
        authoritative: true,
        outcome: "temporarily_unavailable",
        candidate: null,
        reason: "generation queue is paused",
      }),
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
    ).resolves.toEqual({
      kind: "transient",
      perHost: [
        {
          hostId: ORIGIN_HOST_ID,
          label: "this server",
          reason: "generation queue is paused",
        },
      ],
    });
  });

  it("keeps a never-reached host connecting without dropping it from the list", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, [model("flux2-klein:q4")]);

    const routing = useHostRouting();
    await routing.refresh();

    const remote = routing.hosts.value.find((h) => h.id === studio.id);
    expect(remote).toMatchObject({ status: "connecting", stale: false });
    // Auto never routes to it.
    expect(routing.resolve("flux2-klein:q4")?.hostId).toBe(ORIGIN_HOST_ID);
  });

  it("keeps scheduled and host-change refreshes single-flight", async () => {
    vi.useFakeTimers();
    try {
      const first = deferred<ReturnType<typeof status>>();
      const second = deferred<ReturnType<typeof status>>();
      let calls = 0;
      hostStatusCall.mockImplementation(() => {
        calls += 1;
        return calls === 1 ? first.promise : second.promise;
      });
      models.set(ORIGIN_HOST_ID, []);
      const routing = useHostRouting();

      routing.start();
      const initial = routing.refresh();
      expect(hostStatusCall).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(__testing__.POLL_INTERVAL_MS * 2);
      window.dispatchEvent(new Event("mold:hosts-changed"));
      window.dispatchEvent(new Event("mold:hosts-changed"));
      await Promise.resolve();
      expect(hostStatusCall).toHaveBeenCalledTimes(1);

      first.resolve(status({ version: "first" }));
      await initial;
      await Promise.resolve();
      expect(hostStatusCall).toHaveBeenCalledTimes(2);

      second.resolve(status({ version: "second" }));
      await routing.refresh();
      routing.stop();
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps a verified host routable with last-good state through repeated status failures", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      instanceId: "instance-a",
    });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(
      studio.id,
      status({
        queue_depth: 5,
        queue_capacity: 8,
        instance_id: "instance-a",
        gpu_info: {
          backend: "cuda",
          name: "RTX 4090",
          vram_total_mb: 24_564,
          vram_used_mb: 1_024,
        },
      }),
    );
    models.set(studio.id, [model("flux-dev:q4")]);
    capabilities.set(studio.id, {
      gallery: { can_delete: true },
      queue: { heterogeneous_batch: true },
    });
    hostQueueCall.mockImplementation((host: HostEntry) =>
      Promise.resolve({
        entries:
          host.id === studio.id
            ? [
                {
                  id: "still-queued",
                  model: "flux-dev:q4",
                  state: "queued",
                  started_at_unix_ms: 1,
                  position: 3,
                },
              ]
            : [],
        plan: null,
      }),
    );
    setGenerateTargetId(studio.id);
    const routing = useHostRouting();
    await routing.refresh();
    const capabilitySnapshot = routing.capabilitiesByHost.value[studio.id];

    hostStatusCall.mockImplementation((host: HostEntry) =>
      host.id === studio.id
        ? Promise.reject(new Error("status timeout"))
        : Promise.resolve(status()),
    );
    hostQueueCall.mockRejectedValue(new Error("status unavailable"));
    await routing.refresh();
    await routing.refresh();

    expect(
      routing.hosts.value.find((host) => host.id === studio.id),
    ).toMatchObject({
      status: "ready",
      stale: true,
      queueDepth: 5,
      gpu: { name: "RTX 4090", vramTotalMb: 24_564 },
    });
    expect(
      queueStatusFor(routing.queueStatus.value, studio.id, "still-queued"),
    ).toMatchObject({ position: 3 });
    expect(routing.capabilitiesByHost.value[studio.id]).toBe(
      capabilitySnapshot,
    );
    expect(routing.resolve("flux-dev:q4")?.hostId).toBe(studio.id);

    hostStatusCall.mockImplementation((host: HostEntry) =>
      Promise.resolve(
        host.id === studio.id
          ? status({
              queue_depth: 7,
              queue_capacity: 8,
              instance_id: "instance-a",
            })
          : status(),
      ),
    );
    hostQueueCall.mockResolvedValue({ entries: [], plan: null });
    await routing.refresh();
    expect(
      routing.hosts.value.find((host) => host.id === studio.id),
    ).toMatchObject({
      status: "ready",
      stale: false,
      queueDepth: 7,
    });
  });

  it.each(["status", "models", "devices", "queue", "capabilities"] as const)(
    "retires verified routing authority when %s rejects the host credential",
    async (endpoint) => {
      const studio = addHost({
        url: "http://studio:7680",
        name: "Studio",
        apiKey: "rejected-key",
        instanceId: "instance-a",
      });
      statuses.set(ORIGIN_HOST_ID, status());
      models.set(ORIGIN_HOST_ID, []);
      statuses.set(
        studio.id,
        status({ instance_id: "instance-a", queue_depth: 4 }),
      );
      models.set(studio.id, [model("old-model")]);
      capabilities.set(studio.id, { gallery: { can_delete: true } });
      setGenerateTargetId(studio.id);
      const routing = useHostRouting();
      await routing.refresh();

      const rejection = new ApiHttpError(
        `GET /api/${endpoint}`,
        401,
        "API key was rejected",
      );
      if (endpoint === "status") {
        hostStatusCall.mockImplementation((host: HostEntry) =>
          host.id === studio.id
            ? Promise.reject(rejection)
            : Promise.resolve(status()),
        );
      } else if (endpoint === "models") {
        models.set(studio.id, rejection);
      } else if (endpoint === "devices") {
        devices.set(studio.id, rejection);
      } else if (endpoint === "queue") {
        hostQueueCall.mockImplementation((host: HostEntry) =>
          host.id === studio.id
            ? Promise.reject(rejection)
            : Promise.resolve({ entries: [], plan: null }),
        );
      } else {
        capabilities.set(studio.id, rejection);
      }
      await routing.refresh();

      expect(
        routing.hosts.value.find((host) => host.id === studio.id),
      ).toMatchObject({
        status: "connecting",
        stale: false,
        queueDepth: null,
      });
      expect(routing.capabilitiesByHost.value[studio.id]).toBeUndefined();
      expect(routing.inventoryKnown(studio.id)).toBe(false);
      expect(routing.resolve("old-model")).toBeNull();
    },
  );

  it("fences last-good authority when a successful poll reports a replacement instance", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      instanceId: "instance-a",
    });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(
      studio.id,
      status({ instance_id: "instance-a", queue_depth: 2 }),
    );
    models.set(studio.id, [model("old-model")]);
    capabilities.set(studio.id, {
      gallery: { can_delete: true },
      model_access: { restrictions: [] },
    });
    hostQueueCall.mockResolvedValue({
      entries: [
        {
          id: "old-job",
          model: "old-model",
          state: "queued",
          started_at_unix_ms: 1,
          position: 1,
        },
      ],
      plan: null,
    });
    const routing = useHostRouting();
    await routing.refresh();

    statuses.set(
      studio.id,
      status({ instance_id: "instance-b", queue_depth: 0 }),
    );
    models.delete(studio.id);
    capabilities.set(studio.id, new Error("capabilities unavailable"));
    hostQueueCall.mockRejectedValue(new Error("queue unavailable"));
    await routing.refresh();

    expect(
      routing.hosts.value.find((host) => host.id === studio.id),
    ).toMatchObject({
      instanceId: "instance-b",
      stale: false,
      queueDepth: 0,
    });
    expect(
      queueStatusFor(routing.queueStatus.value, studio.id, "old-job"),
    ).toBeNull();
    expect(routing.capabilitiesByHost.value[studio.id]).toBeUndefined();
    expect(routing.inventoryKnown(studio.id)).toBe(false);
  });

  it("retires last-good authority across an explicit disconnect", async () => {
    const studio = addHost({
      url: "http://studio:7680",
      name: "Studio",
      instanceId: "instance-a",
    });
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(
      studio.id,
      status({ instance_id: "instance-a", queue_depth: 4 }),
    );
    models.set(studio.id, [model("old-model")]);
    capabilities.set(studio.id, { gallery: { can_delete: true } });
    const routing = useHostRouting();
    await routing.refresh();

    const stored = JSON.parse(
      localStorage.getItem(HOSTS_STORAGE_KEY) ?? "[]",
    ) as HostEntry[];
    localStorage.setItem(
      HOSTS_STORAGE_KEY,
      JSON.stringify(stored.map((host) => ({ ...host, connected: false }))),
    );
    await routing.refresh();
    localStorage.setItem(
      HOSTS_STORAGE_KEY,
      JSON.stringify(stored.map((host) => ({ ...host, connected: true }))),
    );
    statuses.delete(studio.id);
    models.delete(studio.id);
    capabilities.set(studio.id, new Error("unreachable"));
    await routing.refresh();

    expect(
      routing.hosts.value.find((host) => host.id === studio.id),
    ).toMatchObject({
      status: "connecting",
      stale: false,
      queueDepth: null,
    });
    expect(routing.capabilitiesByHost.value[studio.id]).toBeUndefined();
    expect(routing.inventoryKnown(studio.id)).toBe(false);
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

  it("freezes the exact durable-media capability into the selected host route", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    const durableMedia = {
      protocol_version: 1,
      encrypted_at_rest: true,
      generate_request_media: true,
      identity: true,
      h3_references: false,
      private_h3: false,
    };
    statuses.set(ORIGIN_HOST_ID, status());
    models.set(ORIGIN_HOST_ID, []);
    statuses.set(studio.id, status({ instance_id: "studio-instance" }));
    models.set(studio.id, [model("flux-dev:q4")]);
    capabilities.set(studio.id, {
      gallery: { can_delete: true },
      durable_media: durableMedia,
    });
    setGenerateTargetId(studio.id);

    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.resolve("flux-dev:q4")?.durableMedia).toEqual(durableMedia);
  });

  it("allows Auto across differing model profiles on the same Mold major version", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status({ version: "0.23.1", queue_depth: 1 }));
    models.set(ORIGIN_HOST_ID, [
      model("flux-dev:q4", {
        generation_profile: { profile_hash: "origin-profile" } as never,
      }),
    ]);
    statuses.set(studio.id, status({ version: "0.23.0", queue_depth: 0 }));
    models.set(studio.id, [
      model("flux-dev:q4", {
        generation_profile: { profile_hash: "studio-profile" } as never,
      }),
    ]);
    setGenerateTargetId(AUTO_TARGET_ID);
    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.resolve("flux-dev:q4")?.hostId).toBe(studio.id);
  });

  it("requires an explicit machine when model owners use different Mold major versions", async () => {
    const studio = addHost({ url: "http://studio:7680", name: "Studio" });
    statuses.set(ORIGIN_HOST_ID, status({ version: "0.23.1" }));
    models.set(ORIGIN_HOST_ID, [
      model("flux-dev:q4", {
        generation_profile: { profile_hash: "origin-profile" } as never,
      }),
    ]);
    statuses.set(studio.id, status({ version: "1.0.0" }));
    models.set(studio.id, [
      model("flux-dev:q4", {
        generation_profile: { profile_hash: "studio-profile" } as never,
      }),
    ]);
    setGenerateTargetId(AUTO_TARGET_ID);
    const routing = useHostRouting();
    await routing.refresh();

    expect(routing.modelOwnerIds("flux-dev:q4")).toEqual([
      ORIGIN_HOST_ID,
      studio.id,
    ]);
    expect(routing.resolve("flux-dev:q4")).toBeNull();
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
    ).resolves.toEqual({
      kind: "profile_mismatch",
      perHost: [
        expect.objectContaining({
          hostId: ORIGIN_HOST_ID,
          profileHash: "origin-profile",
          version: "0.23.1",
        }),
        expect.objectContaining({
          hostId: studio.id,
          profileHash: "studio-profile",
          version: "1.0.0",
        }),
      ],
    });
    expect(placementCall).not.toHaveBeenCalled();

    routing.setTarget(studio.id);
    expect(routing.resolve("flux-dev:q4")?.hostId).toBe(studio.id);
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
