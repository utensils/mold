import { beforeEach, describe, expect, it, vi } from "vitest";
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
});

describe("events subscription", () => {
  it("opens /api/events when the server advertises it", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
    const events = useEventsStore();

    await events.subscribe();

    expect(events.live).toBe(true);
    expect(sseStream).toHaveBeenCalledWith("/api/events", expect.anything());
  });

  it("does not open the stream on servers without the capability", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(false));
    const events = useEventsStore();

    await events.subscribe();

    expect(events.live).toBe(false);
    expect(sseStream).not.toHaveBeenCalled();
    events.unsubscribe();
  });

  it("is idempotent", async () => {
    vi.mocked(fetchServerCapabilities).mockResolvedValue(caps(true));
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

  it("ignores job lifecycle frames", () => {
    const events = useEventsStore();
    // Must not throw; generation tracking stays on the per-job streams.
    events.apply({ type: "job_queued", id: "j", model: "m" });
    events.apply({ type: "job_started", id: "j", model: "m" });
    events.apply({ type: "job_ended", id: "j" });
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
      caps: { canPause: true, canCancelAll: true },
      error: null,
    };
    const events = useEventsStore();
    events.apply({ type: "queue_paused" });
    expect(jobs.queues["local"]?.paused).toBe(true);
    events.apply({ type: "queue_resumed" });
    expect(jobs.queues["local"]?.paused).toBe(false);
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
      expect(apiJsonTo).toHaveBeenCalledTimes(1);

      // Queue drains → exactly one trailing refetch, then quiet.
      generation.jobs.length = 0;
      await vi.advanceTimersByTimeAsync(5_100);
      expect(apiJsonTo).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(10_200);
      expect(apiJsonTo).toHaveBeenCalledTimes(2);

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

      expect(apiJsonTo).not.toHaveBeenCalled();
      events.unsubscribe();
    } finally {
      vi.useRealTimers();
    }
  });
});
