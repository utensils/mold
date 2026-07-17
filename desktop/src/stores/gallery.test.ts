import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { ipc } from "../lib/ipc";
import { apiFetchTo, apiJsonTo } from "../lib/api/client";
import { evictHostMedia, evictMedia } from "../lib/gallery/media";
import { useConnectionStore } from "./connection";
import { useGalleryStore, type GalleryBucket } from "./gallery";
import { useHostsStore } from "./hosts";
import type { GalleryImage } from "../lib/api/types";

vi.mock("../lib/ipc", () => ({
  ipc: {
    localGalleryList: vi.fn(),
    localGalleryDelete: vi.fn(),
  },
}));

vi.mock("../lib/api/client", () => ({
  apiJsonTo: vi.fn(),
  apiFetchTo: vi.fn(),
}));

vi.mock("../lib/gallery/media", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/gallery/media")>();
  return { ...actual, evictMedia: vi.fn(), evictHostMedia: vi.fn() };
});

const img = (filename: string, timestamp: number): GalleryImage =>
  ({ filename, timestamp, metadata: { prompt: "p" } }) as never;

const loadedBucket = (items: GalleryImage[] = []): GalleryBucket => ({
  items,
  loading: false,
  error: null,
  loaded: true,
});

/** Primary = the built-in engine (host id "local", HTTP on loopback). */
function connectLocal() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
}

/** Primary = a remote host (id "hal9000-7680"). */
function connectRemote() {
  const conn = useConnectionStore();
  conn.info = { mode: "remote", baseUrl: "http://hal9000:7680", apiKey: "hk" };
  conn.status = "ready";
}

function addExtra(id = "okra-7680", url = "http://okra:7680") {
  useHostsStore().extras.push({
    id,
    label: id.replace(/-\d+$/, ""),
    url,
    apiKey: null,
    status: "ready",
    error: null,
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  vi.mocked(apiJsonTo).mockResolvedValue([]);
  vi.mocked(ipc.localGalleryList).mockResolvedValue([]);
});

describe("gallery sources", () => {
  it("has no separate This-Mac bucket when the primary is the built-in engine", async () => {
    // The built-in/external engine reads this Mac's output dir — its bucket
    // IS this Mac's gallery. A second IPC bucket would list every print twice.
    connectLocal();
    const gallery = useGalleryStore();

    expect(gallery.sources.map((s) => s.key)).toEqual(["local"]);

    await gallery.fetchAll();
    expect(ipc.localGalleryList).not.toHaveBeenCalled();
    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: "http://127.0.0.1:49152", apiKey: null },
      "/api/gallery",
    );
  });

  it("adds a This-Mac IPC bucket only when the primary is remote", async () => {
    connectRemote();
    const gallery = useGalleryStore();

    expect(gallery.sources.map((s) => s.key)).toEqual(["local", "hal9000-7680"]);

    await gallery.fetchAll();
    expect(ipc.localGalleryList).toHaveBeenCalledOnce();
    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "hk" },
      "/api/gallery",
    );
  });

  it("does not duplicate the local source when a remote primary keeps the local engine ready", () => {
    connectRemote();
    const conn = useConnectionStore();
    conn.localInfo = {
      kind: "embedded",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
      port: 7680,
    };
    conn.localStatus = "ready";

    const gallery = useGalleryStore();

    expect(gallery.sources.map((source) => source.key)).toEqual(["local", "hal9000-7680"]);
  });

  it("only ready hosts get buckets", () => {
    connectLocal();
    addExtra();
    useHostsStore().extras[0]!.status = "error";
    const gallery = useGalleryStore();
    expect(gallery.sources.map((s) => s.key)).toEqual(["local"]);
  });
});

describe("merged grid", () => {
  it("merges buckets newest-first with host labels, filters, and counts chips", async () => {
    connectRemote();
    addExtra();
    vi.mocked(apiJsonTo).mockImplementation(
      (target, _path) =>
        Promise.resolve(
          (target as { baseUrl: string }).baseUrl.includes("hal9000")
            ? [img("h1.png", 300), img("h2.png", 100)]
            : [img("o1.png", 250)],
        ) as never,
    );
    vi.mocked(ipc.localGalleryList).mockResolvedValue([img("l1.png", 200)]);
    const gallery = useGalleryStore();

    await gallery.fetchAll();

    expect(gallery.merged.map((e) => e.item.filename)).toEqual([
      "h1.png",
      "o1.png",
      "l1.png",
      "h2.png",
    ]);
    expect(gallery.merged.map((e) => e.sourceKey)).toEqual([
      "hal9000-7680",
      "okra-7680",
      "local",
      "hal9000-7680",
    ]);
    expect(gallery.merged[2]!.hostLabel).toBe("This Mac");

    gallery.filter = "hal9000-7680";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["h1.png", "h2.png"]);

    expect(gallery.chipCounts).toEqual([
      { key: "local", label: "This Mac", count: 1 },
      { key: "hal9000-7680", label: expect.any(String), count: 2 },
      { key: "okra-7680", label: "okra", count: 1 },
    ]);
  });

  it("an unknown filter falls back to the full merged set", async () => {
    connectLocal();
    vi.mocked(apiJsonTo).mockResolvedValue([img("a.png", 1)]);
    const gallery = useGalleryStore();
    await gallery.fetchAll();
    gallery.filter = "gone-host";
    expect(gallery.filtered).toHaveLength(1);
  });

  it("dedupes matching filenames in All, prefers This Mac, and lists every location", async () => {
    connectRemote();
    addExtra();
    vi.mocked(ipc.localGalleryList).mockResolvedValue([img("shared.png", 200)]);
    vi.mocked(apiJsonTo).mockImplementation(
      (target, _path) =>
        Promise.resolve(
          (target as { baseUrl: string }).baseUrl.includes("hal9000")
            ? [img("shared.png", 201), img("remote-only.png", 100)]
            : [img("shared.png", 202)],
        ) as never,
    );
    const gallery = useGalleryStore();

    await gallery.fetchAll();

    expect(gallery.merged.map((e) => e.item.filename)).toEqual(["shared.png", "remote-only.png"]);
    expect(gallery.merged[0]).toMatchObject({
      sourceKey: "local",
      hostLabel: "This Mac",
      availableOn: [
        { key: "local", label: "This Mac" },
        { key: "hal9000-7680", label: "hal9000:7680" },
        { key: "okra-7680", label: "okra" },
      ],
    });

    gallery.filter = "hal9000-7680";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["shared.png", "remote-only.png"]);
    expect(gallery.filtered[0]).toMatchObject({
      sourceKey: "hal9000-7680",
      availableOn: [{ key: "hal9000-7680", label: "hal9000:7680" }],
    });
  });
});

describe("per-bucket isolation", () => {
  it("one host failing never breaks the others", async () => {
    connectRemote();
    addExtra();
    vi.mocked(apiJsonTo).mockImplementation((target, _path) =>
      (target as { baseUrl: string }).baseUrl.includes("hal9000")
        ? Promise.reject(new Error("connection refused"))
        : (Promise.resolve([img("o1.png", 10)]) as never),
    );
    vi.mocked(ipc.localGalleryList).mockResolvedValue([img("l1.png", 20)]);
    const gallery = useGalleryStore();

    await gallery.fetchAll();

    expect(gallery.buckets["hal9000-7680"]!.error).toContain("connection refused");
    expect(gallery.buckets["hal9000-7680"]!.loaded).toBe(false);
    expect(gallery.buckets["okra-7680"]!.loaded).toBe(true);
    expect(gallery.merged.map((e) => e.item.filename)).toEqual(["l1.png", "o1.png"]);
  });
});

describe("live server events", () => {
  it("applyAdded inserts into the primary host's bucket only", () => {
    connectRemote();
    const gallery = useGalleryStore();
    gallery.buckets["hal9000-7680"] = loadedBucket([img("old.png", 100)]);
    gallery.buckets["local"] = loadedBucket([]);

    gallery.applyAdded({
      type: "gallery_added",
      filename: "new.png",
      image: img("new.png", 200),
    });

    expect(gallery.buckets["hal9000-7680"]!.items.map((i) => i.filename)).toEqual([
      "new.png",
      "old.png",
    ]);
    expect(gallery.buckets["local"]!.items).toHaveLength(0);
  });

  it("applyAdded dedupes by filename", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("a.png", 100)]);

    gallery.applyAdded({ type: "gallery_added", filename: "a.png", image: img("a.png", 100) });

    expect(gallery.buckets["local"]!.items).toHaveLength(1);
  });

  it("applyAdded is a no-op while the primary bucket is not loaded", () => {
    connectRemote();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([]);

    gallery.applyAdded({ type: "gallery_added", filename: "x.png", image: img("x.png", 1) });

    expect(gallery.buckets["hal9000-7680"]).toBeUndefined();
    expect(gallery.buckets["local"]!.items).toHaveLength(0);
  });

  it("applyAdded without a row falls back to a debounced refetch of the primary", async () => {
    vi.useFakeTimers();
    try {
      connectLocal();
      const gallery = useGalleryStore();
      gallery.buckets["local"] = loadedBucket([]);

      gallery.applyAdded({ type: "gallery_added", filename: "a.png" });
      gallery.applyAdded({ type: "gallery_added", filename: "b.png" });
      expect(apiJsonTo).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(600);
      // Two events inside the window collapse into one refetch.
      expect(apiJsonTo).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("applyRemoved drops the tile from the primary bucket", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("a.png", 100), img("b.png", 50)]);

    gallery.applyRemoved("a.png");

    expect(gallery.buckets["local"]!.items.map((i) => i.filename)).toEqual(["b.png"]);
  });
});

describe("remove", () => {
  it("deletes This-Mac prints through native IPC and evicts only that bucket", async () => {
    connectRemote();
    vi.mocked(ipc.localGalleryDelete).mockResolvedValue();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("print.png", 1)]);
    gallery.buckets["hal9000-7680"] = loadedBucket([img("print.png", 2)]);

    await gallery.remove("local", "print.png");

    expect(ipc.localGalleryDelete).toHaveBeenCalledWith("print.png");
    expect(apiFetchTo).not.toHaveBeenCalled();
    expect(gallery.buckets["local"]!.items).toHaveLength(0);
    // The same filename on the host bucket is a different print — untouched.
    expect(gallery.buckets["hal9000-7680"]!.items).toHaveLength(1);
    expect(evictMedia).toHaveBeenCalledWith("mold-local://localhost/print.png", "local");
  });

  it("deletes host prints against that host and evicts its cache bucket", async () => {
    connectRemote();
    vi.mocked(apiFetchTo).mockResolvedValue(new Response(null, { status: 200 }));
    const gallery = useGalleryStore();
    gallery.buckets["hal9000-7680"] = loadedBucket([img("print one.png", 2)]);

    await gallery.remove("hal9000-7680", "print one.png");

    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "hk" },
      "/api/gallery/image/print%20one.png",
      { method: "DELETE" },
    );
    expect(ipc.localGalleryDelete).not.toHaveBeenCalled();
    expect(gallery.buckets["hal9000-7680"]!.items).toHaveLength(0);
    expect(evictMedia).toHaveBeenCalledWith(
      "/api/gallery/thumbnail/print%20one.png",
      "hal9000-7680",
    );
    expect(evictMedia).toHaveBeenCalledWith("/api/gallery/image/print%20one.png", "hal9000-7680");
  });
});

describe("refreshHost", () => {
  it("a primary-bucket delete also evicts the default primary cache slots", async () => {
    // ImagePickerModal/StageCard render primary media without a cacheKey.
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("a.png", 1)]);
    vi.mocked(apiFetchTo).mockResolvedValue(new Response(null, { status: 200 }) as never);

    await gallery.remove("local", "a.png");

    const keys = vi.mocked(evictMedia).mock.calls.map(([, key]) => key);
    expect(keys).toContain("local");
    expect(keys).toContain(undefined);
  });

  it("evicts cached media for prints that vanished out-of-band on refetch", async () => {
    // Copilot review on #393: refetch replaced items without releasing the
    // removed prints' blob URLs — a leak when another client deletes.
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("gone.png", 1), img("kept.png", 2)]);
    vi.mocked(apiJsonTo).mockResolvedValue([img("kept.png", 2)]);

    await gallery.fetchBucket("local");

    const evicted = vi.mocked(evictMedia).mock.calls;
    expect(
      evicted.some(([path, key]) => String(path).includes("gone.png") && key === "local"),
    ).toBe(true);
    expect(evicted.some(([path]) => String(path).includes("kept.png"))).toBe(false);
  });

  it("refetches an already-loaded bucket", async () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([]);

    await gallery.refreshHost("local");

    expect(apiJsonTo).toHaveBeenCalledTimes(1);
  });

  it("never force-loads a bucket from a background event", async () => {
    connectLocal();
    const gallery = useGalleryStore();

    await gallery.refreshHost("local");

    expect(apiJsonTo).not.toHaveBeenCalled();
    expect(gallery.buckets["local"]).toBeUndefined();
  });
});

describe("bucket sync", () => {
  it("drops buckets whose source disappeared and evicts their media", async () => {
    connectRemote();
    addExtra();
    const gallery = useGalleryStore();
    await gallery.fetchAll();
    expect(gallery.buckets["okra-7680"]).toBeDefined();

    const hosts = useHostsStore();
    hosts.extras = hosts.extras.filter((h) => h.id !== "okra-7680");
    gallery.filter = "okra-7680";
    await gallery.fetchAll();

    expect(gallery.buckets["okra-7680"]).toBeUndefined();
    expect(evictHostMedia).toHaveBeenCalledWith("okra-7680");
    expect(gallery.filter).toBe("all");
  });
});
