import { describe, expect, it } from "vitest";
import {
  AUTO_TARGET_ID,
  CAPABLE_TARGET_ID,
  backendRank,
  chooseRoutedHost,
  hostIdsForModel,
  inferBackendFromGpuName,
  isAutomaticTarget,
  normalizeTargetHost,
  normalizeTargetId,
  pickAutoHost,
  pickMostCapableHost,
  unionModelsByName,
  type CapableHostBase,
  type RoutableHostBase,
} from "./hostRouting";
import { comparePlacementPreviews } from "../api/generationPlacement";

function routable(overrides: Partial<RoutableHostBase> & { id: string }): RoutableHostBase {
  return { status: "ready", queueDepth: 0, ...overrides };
}

function capable(overrides: Partial<CapableHostBase> & { id: string }): CapableHostBase {
  return { status: "ready", queueDepth: 0, gpu: null, ...overrides };
}

function planned(completionMs: number, setupMs = 0) {
  return {
    version: 1,
    authoritative: true,
    state_version: 1,
    plan_version: 1,
    outcome: "planned",
    candidate: {
      device_id: "cuda:0",
      execution_fingerprint: "fingerprint",
      predicted_start_after_ms: 0,
      predicted_completion_after_ms: completionMs,
      setup_ms: setupMs,
      setup_kind: "warm",
      estimate_confidence: "high",
    },
  };
}

describe("pickAutoHost", () => {
  it("ignores hosts that are not ready", () => {
    expect(
      pickAutoHost([
        routable({ id: "a", status: "connecting", queueDepth: 0 }),
        routable({ id: "b", queueDepth: 4 }),
      ])?.id,
    ).toBe("b");
    expect(pickAutoHost([routable({ id: "a", status: "error" })])).toBeNull();
  });

  it("prefers an authoritative prediction over raw queue depth", () => {
    const chosen = pickAutoHost([
      routable({ id: "a", queueDepth: 0, predictedCompletionMs: 9_000 }),
      routable({ id: "b", queueDepth: 3, predictedCompletionMs: 1_000 }),
    ]);
    expect(chosen?.id).toBe("b");
  });

  it("falls back to queue depth when either host is planless", () => {
    const chosen = pickAutoHost([
      routable({ id: "a", queueDepth: 5, predictedCompletionMs: 1_000 }),
      routable({ id: "b", queueDepth: 1 }),
    ]);
    expect(chosen?.id).toBe("b");
  });

  it("counts an unknown queue depth as the busiest host", () => {
    expect(
      pickAutoHost([routable({ id: "a", queueDepth: null }), routable({ id: "b", queueDepth: 7 })])
        ?.id,
    ).toBe("b");
  });

  it("breaks a dead heat with the surface's home host", () => {
    const hosts = [routable({ id: "remote" }), routable({ id: "local" })];
    expect(pickAutoHost(hosts, { isHome: (host) => host.id === "local" })?.id).toBe("local");
    // No home rule: input order survives unless a lowest-id rule is asked for.
    expect(pickAutoHost(hosts)?.id).toBe("remote");
    expect(pickAutoHost(hosts, { lowestIdWins: true })?.id).toBe("local");
  });
});

describe("backendRank", () => {
  it("walks CUDA > Metal > unknown", () => {
    expect(backendRank("cuda")).toBe(2);
    expect(backendRank("metal")).toBe(1);
    expect(backendRank("cpu")).toBe(0);
    expect(backendRank(null)).toBe(0);
  });

  it("infers the backend from the GPU name only when the field is absent", () => {
    expect(backendRank(null, "NVIDIA GeForce RTX 4090")).toBe(2);
    expect(backendRank(null, "Apple M3 Max")).toBe(1);
    expect(backendRank("metal", "NVIDIA GeForce RTX 4090")).toBe(1);
  });

  it("names the backend families it can recognise", () => {
    expect(inferBackendFromGpuName("NVIDIA L40S")).toBe("cuda");
    expect(inferBackendFromGpuName("Apple M1 Pro")).toBe("metal");
    expect(inferBackendFromGpuName("Some Unknown Accelerator")).toBe("cpu");
  });
});

describe("pickMostCapableHost", () => {
  it("ranks backend, then VRAM, then queue depth", () => {
    const chosen = pickMostCapableHost(
      [
        capable({ id: "mac", gpu: { backend: "metal", vramTotalMb: 128_000 } }),
        capable({ id: "rig", gpu: { backend: "cuda", vramTotalMb: 24_000 } }),
      ],
      null,
    );
    expect(chosen?.id).toBe("rig");

    const bigger = pickMostCapableHost(
      [
        capable({ id: "small", gpu: { backend: "cuda", vramTotalMb: 24_000 } }),
        capable({ id: "big", gpu: { backend: "cuda", vramTotalMb: 48_000 } }),
      ],
      null,
    );
    expect(bigger?.id).toBe("big");

    const idle = pickMostCapableHost(
      [
        capable({ id: "busy", queueDepth: 3, gpu: { backend: "cuda", vramTotalMb: 24_000 } }),
        capable({ id: "idle", queueDepth: 0, gpu: { backend: "cuda", vramTotalMb: 24_000 } }),
      ],
      null,
    );
    expect(idle?.id).toBe("idle");
  });

  it("restricts to model owners when at least one ready host has it", () => {
    const hosts = [
      capable({ id: "strong", gpu: { backend: "cuda", vramTotalMb: 80_000 } }),
      capable({ id: "owner", gpu: { backend: "metal", vramTotalMb: 32_000 } }),
    ];
    expect(pickMostCapableHost(hosts, ["owner"])?.id).toBe("owner");
    // Nobody has it: the whole ready set is eligible again.
    expect(pickMostCapableHost(hosts, ["ghost"])?.id).toBe("strong");
  });

  it("uses the GPU name when the host predates gpu_info.backend", () => {
    const chosen = pickMostCapableHost(
      [
        capable({ id: "mac", gpu: { name: "Apple M2 Ultra", vramTotalMb: 192_000 } }),
        capable({ id: "rig", gpu: { name: "NVIDIA RTX 6000 Ada", vramTotalMb: 48_000 } }),
      ],
      null,
    );
    expect(chosen?.id).toBe("rig");
  });
});

describe("target normalization", () => {
  const hosts = [{ id: "studio" }, { id: "plato" }];

  it("passes both sentinels and listed hosts through", () => {
    expect(normalizeTargetId(AUTO_TARGET_ID, hosts)).toBe(AUTO_TARGET_ID);
    expect(normalizeTargetId(CAPABLE_TARGET_ID, hosts)).toBe(CAPABLE_TARGET_ID);
    expect(normalizeTargetId("plato", hosts)).toBe("plato");
  });

  it("degrades a forgotten host to Auto", () => {
    expect(normalizeTargetId("ghost", hosts)).toBe(AUTO_TARGET_ID);
    expect(normalizeTargetId(null, hosts)).toBe(AUTO_TARGET_ID);
    expect(normalizeTargetHost("ghost", hosts)).toBeNull();
    expect(normalizeTargetHost("capable", hosts)).toBe("capable");
    expect(normalizeTargetHost("studio", hosts)).toBe("studio");
  });

  it("classifies automatic policies", () => {
    expect(isAutomaticTarget(AUTO_TARGET_ID)).toBe(true);
    expect(isAutomaticTarget(CAPABLE_TARGET_ID)).toBe(true);
    expect(isAutomaticTarget(null)).toBe(true);
    expect(isAutomaticTarget("studio")).toBe(false);
  });
});

describe("fleet model views", () => {
  const modelsByHost = {
    studio: [
      { name: "flux-dev:q8", downloaded: true },
      { name: "z-image-turbo:q6", downloaded: false },
    ],
    plato: [
      { name: "z-image-turbo:q6", downloaded: true },
      { name: "flux-dev:q8", downloaded: true },
    ],
  };

  it("unions by name and lets a downloaded copy win", () => {
    const union = unionModelsByName(modelsByHost, ["studio", "plato"]);
    expect(union.map((model) => model.name).sort()).toEqual(["flux-dev:q8", "z-image-turbo:q6"]);
    expect(union.every((model) => model.downloaded)).toBe(true);
  });

  it("lists only the hosts that have a model downloaded", () => {
    expect(hostIdsForModel(modelsByHost, "z-image-turbo:q6")).toEqual(["plato"]);
    expect(hostIdsForModel(modelsByHost, "flux-dev:q8").sort()).toEqual(["plato", "studio"]);
    expect(hostIdsForModel(modelsByHost, "flux-dev:q8", ["plato"])).toEqual(["plato"]);
    expect(hostIdsForModel(modelsByHost, "missing")).toEqual([]);
  });
});

describe("chooseRoutedHost", () => {
  const studio = capable({ id: "studio", gpu: { backend: "metal", vramTotalMb: 96_000 } });
  const plato = capable({ id: "plato", gpu: { backend: "cuda", vramTotalMb: 48_000 } });

  it("takes the soonest completion including round trip under Auto", () => {
    const chosen = chooseRoutedHost(
      [
        { host: studio, roundTripMs: 10, preview: planned(1_000) },
        { host: plato, roundTripMs: 40, preview: planned(500) },
      ],
      AUTO_TARGET_ID,
      comparePlacementPreviews,
    );
    expect(chosen?.id).toBe("plato");

    const slower = chooseRoutedHost(
      [
        { host: studio, roundTripMs: 10, preview: planned(1_000) },
        { host: plato, roundTripMs: 40, preview: planned(1_200) },
      ],
      AUTO_TARGET_ID,
      comparePlacementPreviews,
    );
    expect(slower?.id).toBe("studio");
  });

  it("takes the strongest planner under Most capable even when it is slower", () => {
    const chosen = chooseRoutedHost(
      [
        { host: studio, roundTripMs: 10, preview: planned(500) },
        { host: plato, roundTripMs: 10, preview: planned(5_000) },
      ],
      CAPABLE_TARGET_ID,
      comparePlacementPreviews,
    );
    expect(chosen?.id).toBe("plato");
  });

  it("returns null when no host planned", () => {
    expect(chooseRoutedHost([], AUTO_TARGET_ID, comparePlacementPreviews)).toBeNull();
  });
});
