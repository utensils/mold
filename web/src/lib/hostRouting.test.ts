import { describe, expect, it } from "vitest";
import {
  AUTO_TARGET_ID,
  CAPABLE_TARGET_ID,
  backendRank,
  hostIdsForModel,
  normalizeTargetId,
  pickAutoHost,
  pickMostCapableHost,
  resolveRoute,
  unionModels,
  type RoutableHost,
} from "./hostRouting";
import { ORIGIN_HOST_ID } from "./hostRegistry";
import type { ModelInfoExtended } from "../types";

function host(overrides: Partial<RoutableHost> & { id: string }): RoutableHost {
  return {
    label: overrides.id,
    url: `http://${overrides.id}:7680`,
    status: "ready",
    queueDepth: 0,
    gpu: null,
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

describe("pickAutoHost", () => {
  it("returns null when nothing is ready", () => {
    expect(
      pickAutoHost([
        host({ id: "a", status: "connecting" }),
        host({ id: "b", status: "error" }),
      ]),
    ).toBeNull();
  });

  it("picks the ready host with the shallowest queue", () => {
    const picked = pickAutoHost([
      host({ id: ORIGIN_HOST_ID, queueDepth: 3 }),
      host({ id: "b", queueDepth: 1 }),
      host({ id: "c", queueDepth: 5 }),
    ]);
    expect(picked?.id).toBe("b");
  });

  it("uses predicted completion to refine equal queue depths", () => {
    const picked = pickAutoHost([
      host({ id: "slow", queueDepth: 2, predictedCompletionMs: 20_000 }),
      host({ id: "fast", queueDepth: 2, predictedCompletionMs: 10_000 }),
    ]);
    expect(picked?.id).toBe("fast");
  });

  it("does not starve an idle planless host in a mixed-version fleet", () => {
    const picked = pickAutoHost([
      host({
        id: "planned-busy",
        queueDepth: 4,
        predictedCompletionMs: 10_000,
      }),
      host({ id: "planless-idle", queueDepth: 0, predictedCompletionMs: null }),
    ]);
    expect(picked?.id).toBe("planless-idle");
  });

  it("treats an unknown queue depth as busiest", () => {
    const picked = pickAutoHost([
      host({ id: "a", queueDepth: null }),
      host({ id: "b", queueDepth: 4 }),
    ]);
    expect(picked?.id).toBe("b");
  });

  it("breaks ties toward this server (no network hop)", () => {
    const picked = pickAutoHost([
      host({ id: "remote", queueDepth: 2 }),
      host({ id: ORIGIN_HOST_ID, queueDepth: 2 }),
    ]);
    expect(picked?.id).toBe(ORIGIN_HOST_ID);
  });

  it("never returns an unready host", () => {
    const picked = pickAutoHost([
      host({ id: "down", status: "error", queueDepth: 0 }),
      host({ id: "up", queueDepth: 9 }),
    ]);
    expect(picked?.id).toBe("up");
  });
});

describe("backend ranking", () => {
  it("ranks CUDA above Metal above unknown", () => {
    expect(backendRank("cuda")).toBeGreaterThan(backendRank("metal"));
    expect(backendRank("metal")).toBeGreaterThan(backendRank(null));
  });
});

describe("pickMostCapableHost", () => {
  it("prefers CUDA over Metal even when Metal is idle", () => {
    const picked = pickMostCapableHost(
      [
        host({
          id: ORIGIN_HOST_ID,
          queueDepth: 0,
          gpu: { backend: "metal", name: "Apple M3", vramTotalMb: 65536 },
        }),
        host({
          id: "rig",
          queueDepth: 4,
          gpu: { backend: "cuda", name: "RTX 4090", vramTotalMb: 24576 },
        }),
      ],
      null,
    );
    expect(picked?.id).toBe("rig");
  });

  it("breaks a backend tie on VRAM, then queue depth", () => {
    const byVram = pickMostCapableHost(
      [
        host({
          id: "small",
          gpu: { backend: "cuda", name: "A", vramTotalMb: 12288 },
        }),
        host({
          id: "big",
          gpu: { backend: "cuda", name: "B", vramTotalMb: 49152 },
        }),
      ],
      null,
    );
    expect(byVram?.id).toBe("big");

    const byQueue = pickMostCapableHost(
      [
        host({
          id: "busy",
          queueDepth: 6,
          gpu: { backend: "cuda", name: "A", vramTotalMb: 24576 },
        }),
        host({
          id: "idle",
          queueDepth: 0,
          gpu: { backend: "cuda", name: "B", vramTotalMb: 24576 },
        }),
      ],
      null,
    );
    expect(byQueue?.id).toBe("idle");
  });

  it("restricts to hosts that already hold the model when any ready host does", () => {
    const picked = pickMostCapableHost(
      [
        host({
          id: "strong",
          gpu: { backend: "cuda", name: "A", vramTotalMb: 49152 },
        }),
        host({
          id: "weak",
          gpu: { backend: "metal", name: "Apple M1", vramTotalMb: 16384 },
        }),
      ],
      ["weak"],
    );
    expect(picked?.id).toBe("weak");
  });

  it("ignores the model restriction when no ready host has it", () => {
    const picked = pickMostCapableHost(
      [
        host({
          id: "strong",
          gpu: { backend: "cuda", name: "A", vramTotalMb: 49152 },
        }),
      ],
      ["offline-box"],
    );
    expect(picked?.id).toBe("strong");
  });

  it("returns null when nothing is ready", () => {
    expect(
      pickMostCapableHost([host({ id: "a", status: "error" })], null),
    ).toBeNull();
  });
});

describe("normalizeTargetId", () => {
  const hosts = [{ id: ORIGIN_HOST_ID }, { id: "studio" }];

  it("passes through the sentinels", () => {
    expect(normalizeTargetId(AUTO_TARGET_ID, hosts)).toBe(AUTO_TARGET_ID);
    expect(normalizeTargetId(CAPABLE_TARGET_ID, hosts)).toBe(CAPABLE_TARGET_ID);
  });

  it("passes through a listed host id", () => {
    expect(normalizeTargetId("studio", hosts)).toBe("studio");
  });

  it("reads a forgotten host as Auto", () => {
    expect(normalizeTargetId("ghost", hosts)).toBe(AUTO_TARGET_ID);
    expect(normalizeTargetId(null, hosts)).toBe(AUTO_TARGET_ID);
  });
});

describe("resolveRoute", () => {
  const origin = host({
    id: ORIGIN_HOST_ID,
    label: "this server",
    url: "http://localhost:7680",
    queueDepth: 2,
  });
  const studio = host({
    id: "studio",
    label: "Studio",
    url: "http://studio:7680",
    apiKey: "sk-studio",
    queueDepth: 0,
  });

  it("routes a sticky explicit pick to that host, with its key", () => {
    const route = resolveRoute([origin, studio], "studio");
    expect(route).toEqual({
      hostId: "studio",
      label: "Studio",
      target: { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
    });
  });

  it("refuses a sticky pick that is listed but offline", () => {
    const offline = { ...studio, status: "error" as const };
    expect(resolveRoute([origin, offline], "studio")).toBeNull();
  });

  it("falls back to Auto when the sticky pick was forgotten", () => {
    const route = resolveRoute([origin], "studio");
    expect(route?.hostId).toBe(ORIGIN_HOST_ID);
  });

  it("Auto prefers a ready host that already holds the model", () => {
    // origin has the shallower queue but not the model.
    const shallow = { ...origin, queueDepth: 0 };
    const deep = { ...studio, queueDepth: 7 };
    const route = resolveRoute([shallow, deep], AUTO_TARGET_ID, ["studio"]);
    expect(route?.hostId).toBe("studio");
  });

  it("Auto falls back to the least-busy host when nobody has the model", () => {
    const route = resolveRoute([origin, studio], AUTO_TARGET_ID, ["ghost"]);
    expect(route?.hostId).toBe("studio");
  });

  it("Most capable uses the capability ladder", () => {
    const metal = {
      ...origin,
      gpu: { backend: "metal", name: "Apple M3", vramTotalMb: 65536 },
    };
    const cuda = {
      ...studio,
      gpu: { backend: "cuda", name: "RTX 4090", vramTotalMb: 24576 },
    };
    const route = resolveRoute([metal, cuda], CAPABLE_TARGET_ID);
    expect(route?.hostId).toBe("studio");
  });

  it("returns null when no host is reachable at all", () => {
    const down = { ...origin, status: "error" as const };
    expect(resolveRoute([down], AUTO_TARGET_ID)).toBeNull();
  });

  it("omits the api key for a host that has none", () => {
    const route = resolveRoute([origin], AUTO_TARGET_ID);
    expect(route?.target.apiKey).toBeUndefined();
  });
});

describe("model unions", () => {
  it("dedupes by name across hosts, preferring a downloaded copy", () => {
    const merged = unionModels(
      {
        origin: [model("flux-dev:q4", { downloaded: false })],
        studio: [model("flux-dev:q4"), model("z-image:bf16")],
      },
      [ORIGIN_HOST_ID, "studio"],
    );
    // The `origin` key isn't in the requested host list — only `studio` counts.
    expect(merged.map((m) => m.name)).toEqual(["flux-dev:q4", "z-image:bf16"]);
    expect(merged[0]?.downloaded).toBe(true);
  });

  it("prefers a downloaded copy even when an undownloaded one comes first", () => {
    const merged = unionModels(
      {
        a: [model("flux-dev:q4", { downloaded: false })],
        b: [model("flux-dev:q4", { downloaded: true })],
      },
      ["a", "b"],
    );
    expect(merged).toHaveLength(1);
    expect(merged[0]?.downloaded).toBe(true);
  });

  it("returns an empty union for hosts that reported nothing", () => {
    expect(unionModels({}, ["a"])).toEqual([]);
  });

  it("lists the hosts that hold a downloaded model", () => {
    const byHost = {
      a: [model("flux-dev:q4", { downloaded: false })],
      b: [model("flux-dev:q4")],
      c: [model("z-image:bf16")],
    };
    expect(hostIdsForModel(byHost, "flux-dev:q4")).toEqual(["b"]);
    expect(hostIdsForModel(byHost, "nothing")).toEqual([]);
  });
});
