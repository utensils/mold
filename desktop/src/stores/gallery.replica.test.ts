/**
 * Buckets fed by the shared gallery replica: a host that serves the change
 * feed is applied row by row (no wholesale reassignment, no re-sort), an idle
 * page touches nothing, and tickles that land mid-fetch are never dropped.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { toRaw } from "vue";
import { conditionalApiJsonTo } from "../lib/api/client";
import { evictMedia } from "../lib/gallery/media";
import { useConnectionStore } from "./connection";
import { useGalleryStore } from "./gallery";
import { useHostsStore } from "./hosts";
import type { GalleryImage } from "../lib/api/types";

vi.mock("../lib/ipc", () => ({
  ipc: {
    localGalleryList: vi.fn().mockResolvedValue({ images: [], target: null }),
    localGalleryChanges: vi.fn(),
    forgetGalleryThumbnail: vi.fn().mockResolvedValue(undefined),
  },
}));
vi.mock("../lib/api/client", () => ({
  conditionalApiJsonTo: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../lib/gallery/media", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/gallery/media")>();
  return { ...actual, evictMedia: vi.fn(), evictHostMedia: vi.fn() };
});

const HOST = "plato-7680";
const img = (filename: string, timestamp: number, extra: Partial<GalleryImage> = {}) =>
  ({
    filename,
    timestamp,
    size_bytes: 10,
    metadata: { prompt: "p", model: "m", seed: 1 },
    ...extra,
  }) as GalleryImage;

/** Scripted `/api/gallery/changes` pages, answered in order. */
let pages: unknown[] = [];
const feedCalls: string[] = [];

function connect(feed: boolean) {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
  conn.status = "ready";
  const hosts = useHostsStore();
  hosts.extras.push({
    id: HOST,
    label: "workstation",
    url: "http://workstation:7680",
    apiKey: "pk",
    status: "ready",
    error: null,
    instanceId: null,
  });
  hosts.capabilities[HOST] = {
    gallery: { can_delete: true, ...(feed ? { changes_feed: { protocol_version: 1 } } : {}) },
  } as never;
}

function page(changes: unknown[], cursor: string, extra: Record<string, unknown> = {}) {
  return {
    protocol_version: 1,
    instance_id: "inst",
    cursor,
    more: false,
    reset: false,
    changes,
    ...extra,
  };
}
const up = (image: GalleryImage) => ({ filename: image.filename, removed: false, image });

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  pages = [];
  feedCalls.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      feedCalls.push(url);
      const next = pages.shift();
      if (!next) return new Response("{}", { status: 500 });
      return new Response(JSON.stringify(next), {
        headers: { "content-type": "application/json" },
      });
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

describe("replica-fed buckets", () => {
  it("applies a feed delta in place and leaves an idle page untouched", async () => {
    connect(true);
    const gallery = useGalleryStore();
    pages.push(page([up(img("a.png", 1)), up(img("b.png", 2))], "c1"));
    await gallery.fetchBucket(HOST);
    expect(feedCalls).toEqual(["http://workstation:7680/api/gallery/changes?limit=500"]);
    expect(conditionalApiJsonTo).not.toHaveBeenCalled();
    const bucket = gallery.buckets[HOST]!;
    const rows = toRaw(bucket.items);
    expect(rows.map((r) => r.filename)).toEqual(["b.png", "a.png"]);

    pages.push(
      page(
        [
          up(img("a.png", 1, { title: "Renamed" })),
          { filename: "b.png", removed: true },
          up(img("c.png", 3)),
        ],
        "c2",
      ),
    );
    await gallery.fetchBucket(HOST);
    expect(feedCalls[1]).toContain("since=c1");
    // The same array, spliced — never a wholesale reassignment.
    expect(toRaw(bucket.items)).toBe(rows);
    expect(bucket.items.map((r) => r.filename)).toEqual(["c.png", "a.png"]);
    expect(toRaw(bucket.items[1]!).title).toBe("Renamed");
    expect(evictMedia).toHaveBeenCalled();

    const merged = gallery.merged;
    const index = gallery.organizationIndex;
    pages.push(page([], "c2"));
    await gallery.fetchBucket(HOST);
    expect(gallery.merged).toBe(merged);
    expect(gallery.organizationIndex).toBe(index);
  });

  it("moves a trashed row out of the live grid", async () => {
    connect(true);
    const gallery = useGalleryStore();
    pages.push(page([up(img("a.png", 1)), up(img("b.png", 2))], "c1"));
    await gallery.fetchBucket(HOST);
    pages.push(page([up(img("a.png", 1, { trashed_at: 50 }))], "c2"));
    await gallery.fetchBucket(HOST);
    expect(gallery.buckets[HOST]!.items.map((r) => r.filename)).toEqual(["b.png"]);
  });

  it("keeps the conditional listing for a host without the feed", async () => {
    connect(false);
    const gallery = useGalleryStore();
    vi.mocked(conditionalApiJsonTo).mockResolvedValue([img("a.png", 1)]);
    await gallery.fetchBucket(HOST);
    expect(feedCalls).toEqual([]);
    expect(gallery.buckets[HOST]!.items.map((r) => r.filename)).toEqual(["a.png"]);
  });

  it("a tickle during a fetch queues exactly one more", async () => {
    connect(false);
    const gallery = useGalleryStore();
    vi.mocked(conditionalApiJsonTo).mockResolvedValue([img("a.png", 1)]);
    await gallery.fetchBucket(HOST);
    vi.mocked(conditionalApiJsonTo).mockClear();
    let release!: (rows: GalleryImage[]) => void;
    vi.mocked(conditionalApiJsonTo).mockReturnValueOnce(
      new Promise((resolve) => (release = resolve)) as never,
    );
    vi.mocked(conditionalApiJsonTo).mockResolvedValue([img("a.png", 1), img("b.png", 2)]);
    const first = gallery.fetchBucket(HOST);
    gallery.tickleHost(HOST);
    gallery.tickleHost(HOST);
    release([img("a.png", 1)]);
    await first;
    await vi.waitFor(() =>
      expect(gallery.buckets[HOST]!.items.map((r) => r.filename)).toEqual(["b.png", "a.png"]),
    );
    expect(conditionalApiJsonTo).toHaveBeenCalledTimes(2);
  });

  it("the Library poll stands down for a machine whose stream is live", async () => {
    connect(false);
    const gallery = useGalleryStore();
    vi.mocked(conditionalApiJsonTo).mockResolvedValue([img("a.png", 1)]);
    await gallery.fetchBucket(HOST);
    vi.mocked(conditionalApiJsonTo).mockClear();
    await gallery.pollExtras((key) => key === HOST);
    expect(conditionalApiJsonTo).not.toHaveBeenCalled();
    await gallery.pollExtras(() => false);
    expect(conditionalApiJsonTo).toHaveBeenCalledTimes(1);
  });
});
