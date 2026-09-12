import { beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { createPinia, setActivePinia } from "pinia";
import { useEventsStore } from "./events";
import { useGalleryStore } from "./gallery";
import { useGenerationStore } from "./generation";

vi.mock("../lib/api/serverCapabilities", () => ({
  fetchServerCapabilities: vi.fn(),
}));

vi.mock("../lib/api/sse", () => ({
  sseStream: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../lib/ipc", () => ({
  ipc: { localGalleryList: vi.fn(), localGalleryDelete: vi.fn() },
}));

vi.mock("../lib/api/client", () => ({
  apiJson: vi.fn().mockResolvedValue([]),
  apiFetch: vi.fn(),
  apiJsonTo: vi.fn().mockResolvedValue([]),
  apiFetchTo: vi.fn(),
  ApiError: class ApiError extends Error {},
}));

import { fetchServerCapabilities } from "../lib/api/serverCapabilities";
import { sseStream } from "../lib/api/sse";
import { apiJsonTo } from "../lib/api/client";
import { ipc } from "../lib/ipc";
import { useConnectionStore } from "./connection";

const caps = (available: boolean) =>
  ({ gallery: { can_delete: true }, events: { available } }) as never;

/** Local primary ("local" host id) with a loaded gallery bucket. */
function connectWithBucket() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const gallery = useGalleryStore();
  gallery.buckets["local"] = { items: [], loading: false, error: null, loaded: true };
  return gallery;
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  vi.mocked(ipc.localGalleryList).mockResolvedValue({
    images: [],
    target: {
      baseUrl: "http://127.0.0.1:49152",
      apiKey: "desktop-test-key",
    },
  });
});

describe("events subscription", () => {
  it("opens /api/events when the server advertises it", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    connectWithBucket();
    const events = useEventsStore();

    await events.subscribe();

    expect(events.live).toBe(true);
    expect(sseStream).toHaveBeenCalledWith(
      "/api/events",
      expect.objectContaining({
        target: {
          baseUrl: "http://127.0.0.1:49152",
          apiKey: null,
        },
        terminalHttpStatuses: [401, 403, 404],
      }),
    );
  });

  it("refetches authoritative queue/device state on connect and invalidation events", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    connectWithBucket();
    const { useJobsStore } = await import("./jobs");
    const jobs = useJobsStore();
    const refreshHost = vi.spyOn(jobs, "refreshHost").mockResolvedValue(undefined);
    const events = useEventsStore();

    await events.subscribe();
    const options = vi.mocked(sseStream).mock.calls[0]![1]!;
    options.onOpen?.(new Response());
    options.onEvent?.(
      "message",
      JSON.stringify({
        type: "device_state_changed",
        state: { devices: [], plan_version: 1 },
      }),
    );
    options.onEvent?.(
      "message",
      JSON.stringify({
        type: "queue_plan_changed",
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [],
        },
      }),
    );

    await Promise.resolve();
    expect(refreshHost).toHaveBeenCalledTimes(1);
    expect(refreshHost.mock.calls[0]![0]).toMatchObject({
      id: "local",
      baseUrl: "http://127.0.0.1:49152",
      apiKey: null,
    });
  });

  it("coalesces bursts and runs exactly one follow-up refresh for in-flight invalidations", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    connectWithBucket();
    const { useJobsStore } = await import("./jobs");
    let release!: () => void;
    const first = new Promise<void>((resolve) => {
      release = resolve;
    });
    const refreshHost = vi
      .spyOn(useJobsStore(), "refreshHost")
      .mockImplementationOnce(() => first)
      .mockResolvedValue(undefined);
    const events = useEventsStore();

    await events.subscribe();
    const options = vi.mocked(sseStream).mock.calls[0]![1]!;
    options.onOpen?.(new Response());
    options.onEvent?.("message", JSON.stringify({ type: "device_state_changed" }));
    options.onEvent?.(
      "message",
      JSON.stringify({
        type: "queue_plan_changed",
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [],
        },
      }),
    );
    await Promise.resolve();
    expect(refreshHost).toHaveBeenCalledTimes(1);

    options.onEvent?.("message", JSON.stringify({ type: "device_state_changed" }));
    options.onEvent?.("message", JSON.stringify({ type: "device_state_changed" }));
    expect(refreshHost).toHaveBeenCalledTimes(1);
    release();
    await first;
    await Promise.resolve();
    await Promise.resolve();
    expect(refreshHost).toHaveBeenCalledTimes(2);
  });

  it("does not lose a new-epoch invalidation behind the prior subscription's in-flight wave", async () => {
    connectWithBucket();
    const { useJobsStore } = await import("./jobs");
    let release!: () => void;
    const oldWave = new Promise<void>((resolve) => {
      release = resolve;
    });
    const refreshHost = vi
      .spyOn(useJobsStore(), "refreshHost")
      .mockImplementationOnce(() => oldWave)
      .mockResolvedValue(undefined);
    const events = useEventsStore();

    events.refreshAuthoritativePrimary();
    await Promise.resolve();
    expect(refreshHost).toHaveBeenCalledTimes(1);

    events.unsubscribe();
    events.refreshAuthoritativePrimary();
    release();
    await oldWave;
    await vi.waitFor(() => expect(refreshHost).toHaveBeenCalledTimes(2));
  });

  it("falls back to the primary poller when the primary does not stream events", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(false));
    const events = useEventsStore();

    await events.subscribe();

    expect(events.live).toBe(false);
    expect(events.pollTimer).not.toBeNull();
    // Nothing is connected, so there is no machine to stream from either.
    expect(sseStream).not.toHaveBeenCalled();
    events.unsubscribe();
  });

  it("is idempotent", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    connectWithBucket();
    const events = useEventsStore();

    await events.subscribe();
    await events.subscribe();

    expect(sseStream).toHaveBeenCalledTimes(1);
  });
});

describe("event routing", () => {
  it("routes gallery_added and gallery_removed into the primary bucket", () => {
    const events = useEventsStore();
    const gallery = connectWithBucket();

    events.apply({
      type: "gallery_added",
      filename: "new.png",
      image: { filename: "new.png", timestamp: 5, metadata: { prompt: "p" } } as never,
    });
    expect(gallery.buckets["local"]!.items.map((i) => i.filename)).toEqual(["new.png"]);

    events.apply({ type: "gallery_removed", filename: "new.png" });
    expect(gallery.buckets["local"]!.items).toHaveLength(0);
  });

  it("routes the Library organization frames to the gallery store", () => {
    const events = useEventsStore();
    const gallery = connectWithBucket();
    const applyUpdated = vi.spyOn(gallery, "applyUpdated").mockImplementation(() => {});
    const applyTrashed = vi.spyOn(gallery, "applyTrashed").mockImplementation(() => {});
    const applyRestored = vi.spyOn(gallery, "applyRestored").mockImplementation(() => {});
    const applyCollectionsChanged = vi
      .spyOn(gallery, "applyCollectionsChanged")
      .mockImplementation(() => {});

    const updated = { type: "gallery_updated", filename: "a.png", image: null } as const;
    events.apply(updated);
    expect(applyUpdated).toHaveBeenCalledWith(updated);

    events.apply({ type: "gallery_trashed", filename: "a.png" });
    expect(applyTrashed).toHaveBeenCalledWith("a.png");

    const restored = { type: "gallery_restored", filename: "a.png", image: null } as const;
    events.apply(restored);
    expect(applyRestored).toHaveBeenCalledWith(restored);

    events.apply({ type: "gallery_collections_changed" });
    expect(applyCollectionsChanged).toHaveBeenCalledTimes(1);
  });

  it("ignores job lifecycle frames", () => {
    const events = useEventsStore();
    // Must not throw; generation tracking stays on the per-job streams.
    events.apply({ type: "job_queued", id: "j", model: "m" });
    events.apply({ type: "job_started", id: "j", model: "m" });
    events.apply({ type: "job_ended", id: "j" });
  });

  /*
   * Scene-by-scene authoring is retired on this client, but the SERVER still
   * publishes `chain_job_*` for the jobs the CLI creates (and for a long clip
   * it had to split and stitch). Those frames must fall through the way any
   * unknown type does — no throw, no dead handler — or one CLI `mold run
   * --script` on the same machine would break this app's whole event stream.
   */
  it("ignores a chain job's lifecycle frames without erroring", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    connectWithBucket();
    const events = useEventsStore();
    await events.subscribe();
    const options = vi.mocked(sseStream).mock.calls[0]![1]!;

    for (const frame of [
      { type: "chain_job_queued", id: "c1", model: "ltx2", stage_count: 3 },
      { type: "chain_job_started", id: "c1", model: "ltx2" },
      { type: "chain_job_ended", id: "c1", state: "completed" },
    ]) {
      expect(() => options.onEvent?.("message", JSON.stringify(frame))).not.toThrow();
      expect(() => events.apply(frame as never)).not.toThrow();
    }

    // The stream is still the live one; nothing tore it down.
    expect(events.live).toBe(true);
  });

  it("mirrors queue pause broadcasts onto the primary host's jobs snapshot", async () => {
    const { useConnectionStore } = await import("./connection");
    const { useJobsStore } = await import("./jobs");
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:1", apiKey: null };
    conn.status = "ready";
    const jobs = useJobsStore();
    jobs.queues["local"] = {
      hostId: "local",
      entries: [],
      paused: false,
      caps: { canPause: true, canCancelAll: true, canReorder: false },
      gpuOrdinals: [],
      error: null,
    };
    const events = useEventsStore();
    events.apply({ type: "queue_paused" });
    expect(jobs.queues["local"]?.paused).toBe(true);
    events.apply({ type: "queue_resumed" });
    expect(jobs.queues["local"]?.paused).toBe(false);
  });

  it("reactively applies versioned plans and treats device events as invalidations", async () => {
    connectWithBucket();
    const { useJobsStore } = await import("./jobs");
    const jobs = useJobsStore();
    jobs.queues["local"] = {
      hostId: "local",
      entries: [],
      paused: false,
      caps: null,
      gpuOrdinals: [],
      devices: [],
      plan: null,
      error: null,
    };
    const events = useEventsStore();
    const refreshHost = vi.spyOn(jobs, "refreshHost").mockResolvedValue(undefined);
    const plan = {
      plan_version: 7,
      state_version: 9,
      optimizer_state: "optimized",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [],
    };
    events.apply({ type: "queue_plan_changed", plan });
    expect(jobs.queues["local"]?.plan).toEqual(plan);

    events.apply({
      type: "device_state_changed",
      device_id: "cuda:stable",
      desired_enabled: false,
      admin_state: "draining",
    });
    expect(jobs.queues["local"]?.devices).toEqual([]);
    await Promise.resolve();
    expect(refreshHost).toHaveBeenCalledTimes(1);

    events.apply({
      type: "queue_plan_changed",
      plan: { ...plan, plan_version: 6 },
    });
    expect(jobs.queues["local"]?.plan?.plan_version).toBe(7);
  });
});

describe("old-server fallback poller", () => {
  it("refetches while jobs are pending and once more after the drain", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(false));
      const events = useEventsStore();
      const generation = useGenerationStore();
      connectWithBucket();

      await events.subscribe();

      // Queue busy → each tick refetches.
      generation.jobs.push({ status: "developing" } as never);
      await vi.advanceTimersByTimeAsync(5_100);
      expect(ipc.localGalleryList).toHaveBeenCalledTimes(1);

      // Queue drains → exactly one trailing refetch, then quiet.
      generation.jobs.length = 0;
      await vi.advanceTimersByTimeAsync(5_100);
      expect(ipc.localGalleryList).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(10_200);
      expect(ipc.localGalleryList).toHaveBeenCalledTimes(2);

      events.unsubscribe();
    } finally {
      vi.useRealTimers();
    }
  });

  it("stays quiet while idle", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(false));
      const events = useEventsStore();
      connectWithBucket();

      await events.subscribe();
      await vi.advanceTimersByTimeAsync(20_000);

      expect(ipc.localGalleryList).not.toHaveBeenCalled();
      expect(apiJsonTo).not.toHaveBeenCalled();
      events.unsubscribe();
    } finally {
      vi.useRealTimers();
    }
  });
});

/*
 * The Dock badge counts prints that landed on ANY connected machine while the
 * app was away, so the shared subscription is fleet-wide: one `/api/events`
 * per ready machine. Everything else stays primary-only — the gallery store
 * holds the primary's bucket, and the queue chips read the primary's queue.
 */
describe("fleet-wide event streams", () => {
  /** Local primary + two ready remotes. */
  async function connectFleet() {
    const gallery = connectWithBucket();
    const { useHostsStore } = await import("./hosts");
    const hosts = useHostsStore();
    hosts.extras = [
      {
        id: "plato",
        label: "plato",
        url: "http://plato:7680",
        apiKey: "plato-key",
        status: "ready",
        error: null,
        instanceId: "i-plato",
      },
      {
        id: "hal",
        label: "hal",
        url: "http://hal:7680",
        apiKey: null,
        status: "ready",
        error: null,
        instanceId: "i-hal",
      },
    ];
    return { gallery, hosts };
  }

  function streamTargets() {
    return vi.mocked(sseStream).mock.calls.map(([path, options]) => [path, options?.target]);
  }

  it("opens one stream per ready machine and attaches each to the durable tracker", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    await connectFleet();
    const generation = useGenerationStore();
    const attach = vi
      .spyOn(generation, "attachSharedDurableEventHost")
      .mockImplementation(() => {});
    const events = useEventsStore();

    await events.subscribe();

    expect(streamTargets()).toEqual([
      ["/api/events", { baseUrl: "http://127.0.0.1:49152", apiKey: null }],
      ["/api/events", { baseUrl: "http://plato:7680", apiKey: "plato-key" }],
      ["/api/events", { baseUrl: "http://hal:7680", apiKey: null }],
    ]);
    for (const call of vi.mocked(sseStream).mock.calls) {
      expect(call[1]?.terminalHttpStatuses).toEqual([401, 403, 404]);
    }
    expect(attach.mock.calls.map(([id]) => id)).toEqual(["local", "plato", "hal"]);
    events.unsubscribe();
  });

  it("closes and detaches a machine that stops being ready, and reopens it when it returns", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    const { hosts } = await connectFleet();
    const generation = useGenerationStore();
    vi.spyOn(generation, "attachSharedDurableEventHost").mockImplementation(() => {});
    const detach = vi
      .spyOn(generation, "detachSharedDurableEventHost")
      .mockImplementation(() => {});
    const events = useEventsStore();
    await events.subscribe();
    const platoSignal = vi.mocked(sseStream).mock.calls[1]![1]!.signal!;

    hosts.extras[0]!.status = "error";
    await nextTick();

    expect(platoSignal.aborted).toBe(true);
    expect(detach).toHaveBeenCalledWith("plato");
    expect(vi.mocked(sseStream)).toHaveBeenCalledTimes(3);

    hosts.extras[0]!.status = "ready";
    await nextTick();
    expect(vi.mocked(sseStream)).toHaveBeenCalledTimes(4);
    expect(vi.mocked(sseStream).mock.calls[3]![1]!.target).toEqual({
      baseUrl: "http://plato:7680",
      apiKey: "plato-key",
    });
    events.unsubscribe();
  });

  it("counts a landed print on every machine but files it into the gallery only from the primary", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    const { gallery } = await connectFleet();
    const generation = useGenerationStore();
    vi.spyOn(generation, "attachSharedDurableEventHost").mockImplementation(() => {});
    vi.spyOn(generation, "detachSharedDurableEventHost").mockImplementation(() => {});
    const applyAdded = vi.spyOn(gallery, "applyAdded").mockImplementation(() => {});
    const { useLandedPrintsStore } = await import("./landedPrints");
    const noteLanded = vi.spyOn(useLandedPrintsStore(), "noteLanded").mockImplementation(() => {});
    const events = useEventsStore();
    await events.subscribe();
    const [primary, plato] = vi.mocked(sseStream).mock.calls.map(([, options]) => options!);

    const frame = (filename: string) =>
      JSON.stringify({ type: "gallery_added", filename, image: { filename, timestamp: 1 } });
    primary!.onEvent?.("message", frame("mine.png"));
    expect(applyAdded).toHaveBeenCalledTimes(1);
    expect(noteLanded).toHaveBeenCalledWith("local", "mine.png");

    plato!.onEvent?.("message", frame("theirs.png"));
    expect(noteLanded).toHaveBeenCalledWith("plato", "theirs.png");
    // The gallery store holds the PRIMARY's bucket; a remote print reaches it
    // through the merged fetch, never through another machine's frame.
    expect(applyAdded).toHaveBeenCalledTimes(1);
    events.unsubscribe();
  });

  it("leaves every non-gallery frame from a secondary machine alone", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    await connectFleet();
    const generation = useGenerationStore();
    vi.spyOn(generation, "attachSharedDurableEventHost").mockImplementation(() => {});
    vi.spyOn(generation, "detachSharedDurableEventHost").mockImplementation(() => {});
    const { useJobsStore } = await import("./jobs");
    const jobs = useJobsStore();
    jobs.queues["local"] = {
      hostId: "local",
      entries: [],
      paused: false,
      caps: { canPause: true, canCancelAll: true, canReorder: false },
      gpuOrdinals: [],
      error: null,
    };
    const events = useEventsStore();
    await events.subscribe();
    const plato = vi.mocked(sseStream).mock.calls[1]![1]!;

    plato.onEvent?.("message", JSON.stringify({ type: "queue_paused" }));

    expect(jobs.queues["local"]?.paused).toBe(false);
    events.unsubscribe();
  });

  /*
   * `generation.ensureDurableHostStream` refuses to open its own stream for a
   * machine the shared subscription claims. Leaving one attached after an
   * unsubscribe strands every durable job on it with no events at all.
   */
  it("detaches every machine on unsubscribe", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    await connectFleet();
    const generation = useGenerationStore();
    vi.spyOn(generation, "attachSharedDurableEventHost").mockImplementation(() => {});
    const detach = vi
      .spyOn(generation, "detachSharedDurableEventHost")
      .mockImplementation(() => {});
    const events = useEventsStore();
    await events.subscribe();

    events.unsubscribe();

    expect(detach.mock.calls.map(([id]) => id).sort()).toEqual(["hal", "local", "plato"]);
  });

  /*
   * The capability probe asks the PRIMARY, and the old-server poller it gates
   * is primary-only too. Reading its answer as the fleet's silenced every
   * modern machine behind one machine that predates the endpoint, so the
   * streams open regardless: a machine without `/api/events` answers 404 and
   * `terminalHttpStatuses` closes it once instead of retrying.
   */
  it("still streams from a modern machine when the primary predates /api/events", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(false));
    await connectFleet();
    const generation = useGenerationStore();
    vi.spyOn(generation, "attachSharedDurableEventHost").mockImplementation(() => {});
    vi.spyOn(generation, "detachSharedDurableEventHost").mockImplementation(() => {});
    const { useLandedPrintsStore } = await import("./landedPrints");
    const noteLanded = vi.spyOn(useLandedPrintsStore(), "noteLanded").mockImplementation(() => {});
    const events = useEventsStore();

    await events.subscribe();

    expect(events.live).toBe(false);
    expect(events.pollTimer).not.toBeNull();
    expect(streamTargets()).toEqual([
      ["/api/events", { baseUrl: "http://127.0.0.1:49152", apiKey: null }],
      ["/api/events", { baseUrl: "http://plato:7680", apiKey: "plato-key" }],
      ["/api/events", { baseUrl: "http://hal:7680", apiKey: null }],
    ]);

    vi.mocked(sseStream).mock.calls[1]![1]!.onEvent?.(
      "message",
      JSON.stringify({ type: "gallery_added", filename: "theirs.png" }),
    );
    expect(noteLanded).toHaveBeenCalledWith("plato", "theirs.png");
    events.unsubscribe();
  });

  it("stops watching the fleet after unsubscribe", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    const { hosts } = await connectFleet();
    const generation = useGenerationStore();
    vi.spyOn(generation, "attachSharedDurableEventHost").mockImplementation(() => {});
    vi.spyOn(generation, "detachSharedDurableEventHost").mockImplementation(() => {});
    const events = useEventsStore();
    await events.subscribe();
    events.unsubscribe();

    hosts.extras.push({
      id: "new",
      label: "new",
      url: "http://new:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: "i-new",
    });
    await nextTick();

    expect(vi.mocked(sseStream)).toHaveBeenCalledTimes(3);
  });
});
