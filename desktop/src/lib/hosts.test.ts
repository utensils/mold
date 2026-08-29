import { describe, expect, it } from "vitest";
import {
  backendRank,
  hostIdFromUrl,
  inferBackendFromGpuName,
  mergeSavedHostsByInstanceId,
  modelAvailabilityTag,
  normalizeHostUrl,
  normalizeTargetHost,
  pickAutoHost,
  pickDisplayHost,
  pickMostCapableHost,
  resolveHostDisplayName,
  type CapableHost,
  type RoutableHost,
  type SavedHostLike,
  readyHostSignature,
} from "./hosts";

describe("hostIdFromUrl", () => {
  it("mirrors the Rust host_id slugs exactly", () => {
    // Twin of desktop/src-tauri/src/connection.rs::host_ids_are_stable_slugs —
    // per-host secret names are derived on both sides and must agree.
    expect(hostIdFromUrl("http://hal9000:7680")).toBe("hal9000-7680");
    expect(hostIdFromUrl("https://mold.example.com")).toBe("mold-example-com");
    expect(hostIdFromUrl("https://abc123-7680.proxy.runpod.net")).toBe(
      "abc123-7680-proxy-runpod-net",
    );
    expect(hostIdFromUrl("http://Studio.local:7680")).toBe("studio-local-7680");
    expect(hostIdFromUrl("studio.local:7680")).toBe("studio-local-7680");
  });
});

describe("normalizeHostUrl", () => {
  it("mirrors the Rust normalize_host_url rules", () => {
    expect(normalizeHostUrl("hal9000")).toBe("http://hal9000:7680");
    expect(normalizeHostUrl("studio.local:7680")).toBe("http://studio.local:7680");
    expect(normalizeHostUrl("http://studio.local:7680///")).toBe("http://studio.local:7680");
    expect(normalizeHostUrl("https://mold.example.com/")).toBe("https://mold.example.com");
    expect(() => normalizeHostUrl("   ")).toThrow();
  });
});

function host(overrides: Partial<RoutableHost>): RoutableHost {
  return {
    id: "h",
    kind: "remote",
    status: "ready",
    queueDepth: 0,
    ...overrides,
  };
}

describe("pickAutoHost", () => {
  it("routes to the least-busy ready host", () => {
    const local = host({ id: "local", kind: "local", queueDepth: 3 });
    const idle = host({ id: "idle", queueDepth: 0 });
    const busy = host({ id: "busy", queueDepth: 5 });
    expect(pickAutoHost([local, busy, idle])?.id).toBe("idle");
  });

  it("uses predicted completion to refine equal queue depths", () => {
    const fast = host({
      id: "fast",
      queueDepth: 2,
      predictedCompletionMs: 10_000,
    });
    const slow = host({
      id: "slow",
      queueDepth: 2,
      predictedCompletionMs: 20_000,
    });
    expect(pickAutoHost([slow, fast])?.id).toBe("fast");
  });

  it("ranks predicted completion before raw queue depth when both plans are known", () => {
    const shallowSlow = host({
      id: "shallow-slow",
      queueDepth: 1,
      predictedCompletionMs: 60_000,
    });
    const deeperFast = host({
      id: "deeper-fast",
      queueDepth: 2,
      predictedCompletionMs: 10_000,
    });
    expect(pickAutoHost([shallowSlow, deeperFast])?.id).toBe("deeper-fast");
  });

  it("prefers a known completion estimate over an unknown one on a depth tie", () => {
    const unknown = host({
      id: "local",
      kind: "local",
      queueDepth: 2,
      predictedCompletionMs: null,
    });
    const known = host({
      id: "known",
      queueDepth: 2,
      predictedCompletionMs: 20_000,
    });
    expect(pickAutoHost([unknown, known])?.id).toBe("known");
  });

  it("does not starve an idle planless host in a mixed-version fleet", () => {
    const plannedBusy = host({
      id: "planned-busy",
      queueDepth: 4,
      predictedCompletionMs: 10_000,
    });
    const planlessIdle = host({
      id: "planless-idle",
      queueDepth: 0,
      predictedCompletionMs: null,
    });
    expect(pickAutoHost([plannedBusy, planlessIdle])?.id).toBe("planless-idle");
  });

  it("prefers the local host on a queue-depth tie", () => {
    const local = host({ id: "local", kind: "local", queueDepth: 1 });
    const remote = host({ id: "remote", queueDepth: 1 });
    expect(pickAutoHost([remote, local])?.id).toBe("local");
  });

  it("skips hosts that are not ready and treats unknown depth as busiest", () => {
    const down = host({ id: "down", status: "error", queueDepth: 0 });
    const unknown = host({ id: "unknown", queueDepth: null });
    const known = host({ id: "known", queueDepth: 9 });
    expect(pickAutoHost([down, unknown, known])?.id).toBe("known");
  });

  it("returns null when nothing is ready", () => {
    expect(pickAutoHost([host({ status: "error" })])).toBeNull();
    expect(pickAutoHost([])).toBeNull();
  });
});

describe("inferBackendFromGpuName", () => {
  it("classifies known GPU names", () => {
    const table: Array<[string, string]> = [
      ["NVIDIA GeForce RTX 4090", "cuda"],
      ["RTX 3090", "cuda"],
      ["GeForce GTX 1080 Ti", "cuda"],
      ["Quadro P6000", "cuda"],
      ["Tesla T4", "cuda"],
      ["A100-SXM4-80GB", "cuda"],
      ["H100 PCIe", "cuda"],
      ["L40S", "cuda"],
      ["Apple M3 Max", "metal"],
      ["Apple Metal GPU", "metal"],
      ["M1", "metal"],
      ["M2 Ultra", "metal"],
      ["m4 pro", "metal"],
      ["AMD Radeon RX 7900", "cpu"],
      ["llvmpipe", "cpu"],
      ["", "cpu"],
    ];
    for (const [name, expected] of table) {
      expect(inferBackendFromGpuName(name), name).toBe(expected);
    }
  });
});

describe("backendRank", () => {
  it("ranks cuda > metal > cpu/unknown", () => {
    expect(backendRank("cuda")).toBe(2);
    expect(backendRank("metal")).toBe(1);
    expect(backendRank("cpu")).toBe(0);
    expect(backendRank(null)).toBe(0);
    expect(backendRank(undefined)).toBe(0);
  });

  it("falls back to GPU-name inference when the backend is missing", () => {
    expect(backendRank(null, "NVIDIA GeForce RTX 4090")).toBe(2);
    expect(backendRank(undefined, "Apple M3 Max")).toBe(1);
    expect(backendRank(null, "llvmpipe")).toBe(0);
    expect(backendRank(null, null)).toBe(0);
  });

  it("prefers the explicit backend over the name", () => {
    expect(backendRank("metal", "NVIDIA GeForce RTX 4090")).toBe(1);
  });
});

function capable(overrides: Partial<CapableHost> & { id: string }): CapableHost {
  return {
    kind: "remote",
    status: "ready",
    queueDepth: 0,
    gpu: null,
    ...overrides,
  };
}

describe("pickMostCapableHost", () => {
  it("routes to the CUDA host even with a deeper queue than a Metal host", () => {
    const mac = capable({
      id: "mac",
      kind: "local",
      queueDepth: 0,
      gpu: { backend: "metal", name: "Apple M3 Max", vramTotalMb: 65536 },
    });
    const cuda = capable({
      id: "hal9000-7680",
      queueDepth: 5,
      gpu: { backend: "cuda", name: "NVIDIA GeForce RTX 4090", vramTotalMb: 24564 },
    });
    expect(pickMostCapableHost([mac, cuda], null)?.id).toBe("hal9000-7680");
  });

  it("breaks a backend tie by total VRAM, descending", () => {
    const small = capable({
      id: "small",
      queueDepth: 0,
      gpu: { backend: "cuda", name: "RTX 4090", vramTotalMb: 24564 },
    });
    const big = capable({
      id: "big",
      queueDepth: 3,
      gpu: { backend: "cuda", name: "A100-SXM4-80GB", vramTotalMb: 81920 },
    });
    expect(pickMostCapableHost([small, big], null)?.id).toBe("big");
  });

  it("breaks a backend+VRAM tie by queue depth; unknown depth counts as busiest", () => {
    const busy = capable({
      id: "busy",
      queueDepth: 4,
      gpu: { backend: "cuda", name: "RTX 4090", vramTotalMb: 24564 },
    });
    const idle = capable({
      id: "idle",
      queueDepth: 1,
      gpu: { backend: "cuda", name: "RTX 4090", vramTotalMb: 24564 },
    });
    const unknown = capable({
      id: "unknown",
      queueDepth: null,
      gpu: { backend: "cuda", name: "RTX 4090", vramTotalMb: 24564 },
    });
    expect(pickMostCapableHost([busy, unknown, idle], null)?.id).toBe("idle");
  });

  it("restricts to ready hosts that have the model", () => {
    const cudaWithout = capable({
      id: "cuda",
      gpu: { backend: "cuda", name: "RTX 4090", vramTotalMb: 24564 },
    });
    const metalWith = capable({
      id: "mac",
      gpu: { backend: "metal", name: "Apple M3 Max", vramTotalMb: 65536 },
    });
    expect(pickMostCapableHost([cudaWithout, metalWith], ["mac"])?.id).toBe("mac");
  });

  it("falls back to every ready host when none has the model", () => {
    const cuda = capable({
      id: "cuda",
      gpu: { backend: "cuda", name: "RTX 4090", vramTotalMb: 24564 },
    });
    const metal = capable({
      id: "mac",
      gpu: { backend: "metal", name: "Apple M3 Max", vramTotalMb: 65536 },
    });
    expect(pickMostCapableHost([cuda, metal], ["gone-host"])?.id).toBe("cuda");
  });

  it("treats a missing gpu row as least capable but infers from the name when present", () => {
    const bare = capable({ id: "bare", gpu: null });
    const named = capable({
      id: "named",
      gpu: { name: "NVIDIA GeForce RTX 3060", vramTotalMb: null },
    });
    expect(pickMostCapableHost([bare, named], null)?.id).toBe("named");
  });

  it("prefers the local host only on a full tie", () => {
    const local = capable({ id: "local", kind: "local", queueDepth: 1 });
    const remote = capable({ id: "remote", queueDepth: 1 });
    expect(pickMostCapableHost([remote, local], null)?.id).toBe("local");
    const remoteIdle = capable({ id: "remote-idle", queueDepth: 0 });
    expect(pickMostCapableHost([remoteIdle, local], null)?.id).toBe("remote-idle");
  });

  it("returns null when nothing is ready", () => {
    expect(pickMostCapableHost([capable({ id: "down", status: "error" })], null)).toBeNull();
    expect(pickMostCapableHost([], null)).toBeNull();
  });
});

describe("modelAvailabilityTag", () => {
  const fleet = [
    { id: "local", label: "This Mac", primary: true },
    { id: "hal9000-7680", label: "hal9000", primary: false },
    { id: "bender-7680", label: "bender", primary: false },
  ];

  it("stays quiet when availability is unknown or the primary has the model", () => {
    expect(modelAvailabilityTag([], fleet)).toBeNull();
    expect(modelAvailabilityTag(["local"], fleet)).toBeNull();
    expect(modelAvailabilityTag(["local", "hal9000-7680", "bender-7680"], fleet)).toBeNull();
  });

  it("names the single non-primary host that has the model", () => {
    expect(modelAvailabilityTag(["hal9000-7680"], fleet)).toBe("hal9000");
  });

  it("counts hosts when several non-primary hosts have the model", () => {
    expect(modelAvailabilityTag(["hal9000-7680", "bender-7680"], fleet)).toBe("2 hosts");
  });

  it("ignores host ids that are no longer connected", () => {
    expect(modelAvailabilityTag(["gone-host"], fleet)).toBeNull();
  });
});

describe("resolveHostDisplayName", () => {
  it("prefers a user-assigned name over everything", () => {
    expect(
      resolveHostDisplayName({ name: "Render box", url: "http://192.168.1.5:7680" }, "hal9000"),
    ).toBe("Render box");
  });

  it("uses the server hostname when there is no user name", () => {
    expect(resolveHostDisplayName({ name: null, url: "http://192.168.1.5:7680" }, "hal9000")).toBe(
      "hal9000",
    );
  });

  it("falls back to the scheme-stripped URL when hostname is unknown", () => {
    expect(resolveHostDisplayName({ name: null, url: "http://192.168.1.5:7680" }, null)).toBe(
      "192.168.1.5:7680",
    );
    expect(resolveHostDisplayName(null, undefined)).toBe("");
  });
});

describe("mergeSavedHostsByInstanceId", () => {
  const host = (over: Partial<SavedHostLike>): SavedHostLike => ({
    id: "id",
    url: "http://h:7680",
    lastUsedMs: 1,
    ...over,
  });

  it("collapses the same box reached by hostname and by IP into one entry", () => {
    const byName = host({
      id: "hal9000-7680",
      url: "http://hal9000:7680",
      instanceId: "uuid-1",
      lastUsedMs: 200,
    });
    const byIp = host({
      id: "192-168-1-114-7680",
      url: "http://192.168.1.114:7680",
      instanceId: "uuid-1",
      lastUsedMs: 100,
    });
    const { hosts, dropped } = mergeSavedHostsByInstanceId([byName, byIp]);
    // Survivor is the most-recently-used slug; the IP entry drops.
    expect(hosts.map((h) => h.id)).toEqual(["hal9000-7680"]);
    expect(dropped).toEqual([{ loser: "192-168-1-114-7680", survivor: "hal9000-7680" }]);
  });

  it("keeps the entry with the most recent lastUsedMs as survivor", () => {
    const older = host({ id: "a", instanceId: "u", lastUsedMs: 10 });
    const newer = host({ id: "b", instanceId: "u", lastUsedMs: 99 });
    const { hosts, dropped } = mergeSavedHostsByInstanceId([older, newer]);
    expect(hosts.map((h) => h.id)).toEqual(["b"]);
    expect(dropped).toEqual([{ loser: "a", survivor: "b" }]);
  });

  it("carries a user name from a duplicate onto an unnamed survivor", () => {
    const survivor = host({ id: "a", instanceId: "u", lastUsedMs: 50, name: null });
    const loser = host({ id: "b", instanceId: "u", lastUsedMs: 10, name: "Render box" });
    const { hosts } = mergeSavedHostsByInstanceId([survivor, loser]);
    expect(hosts).toHaveLength(1);
    expect(hosts[0]).toMatchObject({ id: "a", name: "Render box" });
  });

  it("merges same-uuid entries even when their reported hostnames differ", () => {
    const a = host({ id: "pod-a-7680", instanceId: "u", lastUsedMs: 2 });
    const b = host({ id: "pod-b-7680", instanceId: "u", lastUsedMs: 1 });
    const { hosts, dropped } = mergeSavedHostsByInstanceId([a, b]);
    expect(hosts.map((h) => h.id)).toEqual(["pod-a-7680"]);
    expect(dropped).toEqual([{ loser: "pod-b-7680", survivor: "pod-a-7680" }]);
  });

  it("merges UUID aliases regardless of which address supplied the UUID", () => {
    const byName = host({
      id: "hal9000-7680",
      instanceId: "u",
      lastUsedMs: 2,
    });
    const byIp = host({ id: "192-168-1-114-7680", instanceId: "u", lastUsedMs: 1 });
    const { hosts, dropped } = mergeSavedHostsByInstanceId([byName, byIp]);
    expect(hosts.map((h) => h.id)).toEqual(["hal9000-7680"]);
    expect(dropped).toEqual([{ loser: "192-168-1-114-7680", survivor: "hal9000-7680" }]);
  });

  it("never merges entries that have no instance id (falls back to slug identity)", () => {
    const a = host({ id: "hal9000-7680", instanceId: null });
    const b = host({ id: "192-168-1-114-7680", instanceId: null });
    const { hosts, dropped } = mergeSavedHostsByInstanceId([a, b]);
    expect(hosts.map((h) => h.id)).toEqual(["hal9000-7680", "192-168-1-114-7680"]);
    expect(dropped).toEqual([]);
  });
});

describe("pickDisplayHost", () => {
  it("returns the primary when nothing is generating", () => {
    expect(pickDisplayHost([], "local")).toBe("local");
  });

  it("follows the most recently submitted live job's host", () => {
    expect(pickDisplayHost(["local", "hal9000-7680"], "local")).toBe("hal9000-7680");
    expect(pickDisplayHost(["hal9000-7680", "local"], "local")).toBe("local");
  });

  it("skips jobs without a routed host (single-host submissions)", () => {
    expect(pickDisplayHost(["hal9000-7680", null], "local")).toBe("hal9000-7680");
    expect(pickDisplayHost([null, null], "local")).toBe("local");
  });
});

describe("normalizeTargetHost", () => {
  const hosts = [{ id: "local" }, { id: "plato-7680" }];

  it("passes through null, capable, and live host ids", () => {
    expect(normalizeTargetHost(null, hosts)).toBeNull();
    expect(normalizeTargetHost(undefined, hosts)).toBeNull();
    expect(normalizeTargetHost("capable", hosts)).toBe("capable");
    expect(normalizeTargetHost("plato-7680", hosts)).toBe("plato-7680");
  });

  it("reads a ghost host id as Auto — matching the Host selector's display", () => {
    expect(normalizeTargetHost("gone-7680", hosts)).toBeNull();
    expect(normalizeTargetHost("gone-7680", [])).toBeNull();
  });
});

describe("readyHostSignature", () => {
  const host = (id: string, status: string) => ({ id, status });

  it("includes only ready hosts, sorted for stability", () => {
    expect(
      readyHostSignature([host("b", "ready"), host("a", "ready"), host("c", "connecting")]),
    ).toBe("a,b");
  });

  it("changes when a remote finishes its boot reconnect", () => {
    const before = readyHostSignature([host("local", "ready"), host("plato", "connecting")]);
    const after = readyHostSignature([host("local", "ready"), host("plato", "ready")]);
    expect(before).not.toBe(after);
  });

  it("is stable across reorderings of the same ready set", () => {
    expect(readyHostSignature([host("x", "ready"), host("y", "ready")])).toBe(
      readyHostSignature([host("y", "ready"), host("x", "ready")]),
    );
  });
});
