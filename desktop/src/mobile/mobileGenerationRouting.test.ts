import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GenerationPlacementPreview } from "@studio/api/generationPlacement";
import { ApiError } from "../lib/api/client";
import type { HostRoute } from "../stores/hosts";
import type { MobileHost } from "./hosts";

const { previewChainPlacement, previewGenerationPlacement } = vi.hoisted(() => ({
  previewChainPlacement: vi.fn(),
  previewGenerationPlacement: vi.fn(),
}));

vi.mock("@studio/api/generationPlacement", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationPlacement")>()),
  previewChainPlacement,
  previewGenerationPlacement,
}));

import {
  previewPinnedMobileGeneration,
  routeAutomaticMobileGeneration,
} from "./mobileGenerationRouting";

function host(id: string): MobileHost {
  return {
    id,
    name: id === "studio" ? "Studio" : "Render",
    baseUrl: `http://${id}.test:7680`,
    apiKey: `${id}-secret`,
    hostname: id,
    version: "0.18.0",
    instanceId: `${id}-instance`,
    online: true,
  };
}

function routeForHost(candidate: MobileHost): HostRoute {
  return {
    hostId: candidate.id,
    label: candidate.name,
    kind: "remote",
    target: { baseUrl: candidate.baseUrl, apiKey: candidate.apiKey },
    instanceId: candidate.instanceId ?? null,
  };
}

function canonicalRouteForHost(candidate: MobileHost): HostRoute {
  return {
    ...routeForHost(candidate),
    heterogeneousBatch: true,
    heterogeneousBatchMaxOutputs: 64,
    durableBatchOutcomes: true,
    admissionProtocolVersion: 2,
    durableMedia: {
      protocol_version: 2,
      encrypted_at_rest: true,
      generate_request_media: true,
      identity: true,
      h3_references: false,
      private_h3: true,
    },
  };
}

function planned(completion: number): GenerationPlacementPreview {
  return {
    version: 1,
    authoritative: true,
    state_version: 1,
    plan_version: 1,
    outcome: "planned",
    candidate: {
      device_id: "cuda:0",
      execution_fingerprint: "test",
      predicted_start_after_ms: 0,
      predicted_completion_after_ms: completion,
      setup_ms: 0,
      setup_kind: "warm",
      estimate_confidence: "high",
    },
  };
}

function candidate(
  machine: MobileHost,
  options: {
    backend?: string;
    queueDepth?: number;
    vramTotalMb?: number;
  } = {},
) {
  return {
    host: machine,
    view: {
      id: machine.id,
      status: "ready" as const,
      queueDepth: options.queueDepth ?? 0,
      gpu: {
        backend: options.backend ?? "metal",
        name: options.backend === "cuda" ? "RTX 4090" : "Apple M3",
        vramTotalMb: options.vramTotalMb ?? 24_000,
      },
    },
  };
}

function options(candidates: ReturnType<typeof candidate>[]) {
  return {
    candidates,
    routeForHost,
    policy: "auto",
    request: { model: "test-model" },
    chain: false,
    copies: 1,
    subject: "print" as const,
    requireAuthoritative: false,
    settleMs: 0,
  };
}

describe("mobile automatic generation routing", () => {
  beforeEach(() => {
    previewChainPlacement.mockReset();
    previewGenerationPlacement.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("chooses Auto's soonest predicted completion and freezes that route", async () => {
    const studio = host("studio");
    const render = host("render");
    previewGenerationPlacement.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(planned(target.baseUrl === render.baseUrl ? 100 : 9_000)),
    );

    const result = await routeAutomaticMobileGeneration(
      options([candidate(studio), candidate(render)]),
    );

    expect(result).toMatchObject({
      kind: "route",
      host: { id: "render" },
      route: {
        hostId: "render",
        target: { baseUrl: render.baseUrl, apiKey: render.apiKey },
        instanceId: "render-instance",
      },
      legacyUnsupported: false,
    });
  });

  it("uses each candidate's captured GPU when Most capable chooses", async () => {
    const studio = host("studio");
    const render = host("render");
    previewGenerationPlacement.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve(planned(target.baseUrl === render.baseUrl ? 9_000 : 100)),
    );

    const result = await routeAutomaticMobileGeneration({
      ...options([
        candidate(studio, { backend: "metal", vramTotalMb: 128_000 }),
        candidate(render, { backend: "cuda", vramTotalMb: 24_000 }),
      ]),
      policy: "capable",
    });

    expect(result).toMatchObject({ kind: "route", host: { id: "render" } });
  });

  it("routes Auto from cached v2 telemetry without opening placement probes", async () => {
    const studio = host("studio");
    const render = host("render");
    const result = await routeAutomaticMobileGeneration({
      ...options([candidate(studio, { queueDepth: 4 }), candidate(render, { queueDepth: 1 })]),
      routeForHost: canonicalRouteForHost,
    });

    expect(result).toMatchObject({ kind: "route", host: { id: "render" } });
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
    expect(previewChainPlacement).not.toHaveBeenCalled();
  });

  it("routes Most capable from cached v2 GPU facts without opening probes", async () => {
    const studio = host("studio");
    const render = host("render");
    const result = await routeAutomaticMobileGeneration({
      ...options([
        candidate(studio, { backend: "metal", vramTotalMb: 128_000 }),
        candidate(render, { backend: "cuda", vramTotalMb: 24_000 }),
      ]),
      routeForHost: canonicalRouteForHost,
      policy: "capable",
    });

    expect(result).toMatchObject({ kind: "route", host: { id: "render" } });
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
  });

  it("falls back to a legacy server only for media-free non-authoritative work", async () => {
    const studio = host("studio");
    previewGenerationPlacement.mockRejectedValue(new ApiError("not found", 404));

    const result = await routeAutomaticMobileGeneration(options([candidate(studio)]));

    expect(result).toMatchObject({
      kind: "route",
      host: { id: "studio" },
      placement: null,
      legacyUnsupported: true,
    });
  });

  it("refuses the legacy fallback for identity requests", async () => {
    const studio = host("studio");
    previewGenerationPlacement.mockRejectedValue(new ApiError("not found", 404));

    const result = await routeAutomaticMobileGeneration({
      ...options([candidate(studio)]),
      request: { model: "test-model", id_image: "face-bytes" },
    });

    expect(result.kind).toBe("error");
  });

  it("returns abandoned instead of authorizing a stale caller", async () => {
    const studio = host("studio");
    previewGenerationPlacement.mockResolvedValue(planned(100));

    const result = await routeAutomaticMobileGeneration({
      ...options([candidate(studio)]),
      isCurrent: () => false,
    });

    expect(result).toEqual({ kind: "abandoned" });
  });

  it("stops waiting on a stalled candidate after another machine plans", async () => {
    vi.useFakeTimers();
    const studio = host("studio");
    const render = host("render");
    const stalled: { signal: AbortSignal | null } = { signal: null };
    previewGenerationPlacement.mockImplementation(
      (
        target: { baseUrl: string },
        _request: unknown,
        _copies: number,
        config: { signal: AbortSignal },
      ) => {
        if (target.baseUrl === studio.baseUrl) {
          stalled.signal = config.signal;
          return new Promise(() => {});
        }
        return Promise.resolve(planned(100));
      },
    );

    const routing = routeAutomaticMobileGeneration({
      ...options([candidate(studio), candidate(render)]),
      settleMs: 25,
    });
    await vi.advanceTimersByTimeAsync(25);

    await expect(routing).resolves.toMatchObject({ kind: "route", host: { id: "render" } });
    expect(stalled.signal?.aborted).toBe(true);
  });
});

describe("mobile pinned generation placement", () => {
  beforeEach(() => {
    previewChainPlacement.mockReset();
    previewGenerationPlacement.mockReset();
  });

  function pinnedOptions(machine = host("studio")) {
    return {
      route: routeForHost(machine),
      request: { model: "test-model" },
      chain: false,
      copies: 1,
      subject: "print" as const,
      requireAuthoritative: false,
    };
  }

  it("accepts a planned placement on the frozen route", async () => {
    previewGenerationPlacement.mockResolvedValue(planned(100));

    await expect(previewPinnedMobileGeneration(pinnedOptions())).resolves.toMatchObject({
      kind: "placement",
      legacyUnsupported: false,
      placement: { outcome: "planned" },
    });
  });

  it("permits the legacy fallback only when authoritative placement is unnecessary", async () => {
    previewGenerationPlacement.mockRejectedValue(new ApiError("missing", 404));

    await expect(previewPinnedMobileGeneration(pinnedOptions())).resolves.toEqual({
      kind: "placement",
      placement: null,
      legacyUnsupported: true,
    });
  });

  it("submits every pinned family directly behind the canonical v2 contract", async () => {
    for (const request of [
      { model: "flux-dev" },
      { model: "ltx-2", source_image: "frame" },
      {
        model: "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
        source_image: "frame",
      },
    ]) {
      await expect(
        previewPinnedMobileGeneration({
          ...pinnedOptions(),
          route: canonicalRouteForHost(host("studio")),
          request,
        }),
      ).resolves.toEqual({
        kind: "placement",
        placement: null,
        legacyUnsupported: false,
      });
    }
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
    expect(previewChainPlacement).not.toHaveBeenCalled();
  });

  it("keeps pinned sequences on the legacy placement contract", async () => {
    previewChainPlacement.mockResolvedValue(planned(100));
    await expect(
      previewPinnedMobileGeneration({
        ...pinnedOptions(),
        route: canonicalRouteForHost(host("studio")),
        request: { model: "ltx-2", stages: [] },
        chain: true,
      }),
    ).resolves.toMatchObject({ kind: "placement", placement: { outcome: "planned" } });
    expect(previewChainPlacement).toHaveBeenCalledOnce();
  });

  it("refuses a legacy identity placement", async () => {
    previewGenerationPlacement.mockRejectedValue(new ApiError("missing", 404));

    const result = await previewPinnedMobileGeneration({
      ...pinnedOptions(),
      request: { model: "test-model", id_image: "face" },
      requireAuthoritative: true,
    });

    expect(result).toMatchObject({ kind: "error" });
    if (result.kind === "error") expect(result.message).toContain("identity");
  });

  it("abandons a late answer after cancellation", async () => {
    previewGenerationPlacement.mockResolvedValue(planned(100));

    await expect(
      previewPinnedMobileGeneration({ ...pinnedOptions(), isCurrent: () => false }),
    ).resolves.toEqual({ kind: "abandoned" });
  });
});
