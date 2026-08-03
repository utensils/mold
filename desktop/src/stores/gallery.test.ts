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
  authorityTarget: null,
  authorityResolved: false,
});

/** Primary = the built-in engine (host id "local", HTTP on loopback). */
function connectLocal() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" };
  conn.status = "ready";
  vi.mocked(ipc.localGalleryList).mockResolvedValue({
    images: [],
    target: { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
  });
}

function addExtra(id = "okra-7680", url = "http://okra:7680", apiKey: string | null = null) {
  useHostsStore().extras.push({
    id,
    label: id.replace(/-\d+$/, ""),
    url,
    apiKey,
    status: "ready",
    error: null,
    instanceId: null,
  });
}

/** Local primary plus hal9000 connected as an extra host (a second bucket). */
function connectLocalPlusHal() {
  connectLocal();
  addExtra("hal9000-7680", "http://hal9000:7680", "hk");
}

function localSnapshot(images: GalleryImage[], target = true) {
  vi.mocked(ipc.localGalleryList).mockResolvedValue({
    images,
    target: target ? { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" } : null,
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  vi.mocked(apiJsonTo).mockResolvedValue([]);
  vi.mocked(ipc.localGalleryList).mockResolvedValue({ images: [], target: null });
});

describe("gallery sources", () => {
  it("reads the built-in engine's gallery over HTTP, with no separate IPC bucket", async () => {
    // The built-in engine (host id "local") reads this device's output dir; its
    // /api/gallery covers IPC-saved files, so there is no second This-Mac bucket.
    connectLocal();
    const gallery = useGalleryStore();

    expect(gallery.sources.map((s) => s.key)).toEqual(["local"]);

    await gallery.fetchAll();
    expect(ipc.localGalleryList).toHaveBeenCalledTimes(1);
    expect(gallery.targetOf("local")).toEqual({
      baseUrl: "http://127.0.0.1:49152",
      apiKey: "desktop-key",
    });
  });

  it("gives every connected host its own bucket", async () => {
    connectLocalPlusHal();
    const gallery = useGalleryStore();

    expect(gallery.sources.map((s) => s.key)).toEqual(["local", "hal9000-7680"]);

    await gallery.fetchAll();
    expect(ipc.localGalleryList).toHaveBeenCalledTimes(1);
    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "hk" },
      "/api/gallery",
    );
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
    connectLocalPlusHal();
    addExtra();
    vi.mocked(ipc.localGalleryList).mockResolvedValue({
      images: [img("l1.png", 200)],
      target: { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
    });
    vi.mocked(apiJsonTo).mockImplementation((target, _path) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000"))
        return Promise.resolve([img("h1.png", 300), img("h2.png", 100)]) as never;
      if (url.includes("okra")) return Promise.resolve([img("o1.png", 250)]) as never;
      return Promise.resolve([img("l1.png", 200)]) as never; // built-in engine
    });
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
    expect(gallery.merged[2]!.hostLabel).toBe("This device");

    gallery.filter = "hal9000-7680";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["h1.png", "h2.png"]);

    expect(gallery.chipCounts).toEqual([
      { key: "local", label: "This device", count: 1 },
      { key: "hal9000-7680", label: expect.any(String), count: 2 },
      { key: "okra-7680", label: "okra", count: 1 },
    ]);
  });

  it("an unknown filter falls back to the full merged set", async () => {
    connectLocal();
    vi.mocked(ipc.localGalleryList).mockResolvedValue({
      images: [img("a.png", 1)],
      target: { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
    });
    const gallery = useGalleryStore();
    await gallery.fetchAll();
    gallery.filter = "gone-host";
    expect(gallery.filtered).toHaveLength(1);
  });

  it("dedupes matching filenames in All, prefers This device, and lists every location", async () => {
    connectLocalPlusHal();
    addExtra();
    vi.mocked(ipc.localGalleryList).mockResolvedValue({
      images: [img("shared.png", 202)],
      target: { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
    });
    vi.mocked(apiJsonTo).mockImplementation((target, _path) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000"))
        return Promise.resolve([img("shared.png", 201), img("remote-only.png", 100)]) as never;
      if (url.includes("okra")) return Promise.resolve([img("shared.png", 200)]) as never;
      return Promise.resolve([img("shared.png", 202)]) as never; // built-in engine
    });
    const gallery = useGalleryStore();

    await gallery.fetchAll();

    expect(gallery.merged.map((e) => e.item.filename)).toEqual(["shared.png", "remote-only.png"]);
    expect(gallery.merged[0]).toMatchObject({
      sourceKey: "local",
      hostLabel: "This device",
      availableOn: [
        { key: "local", label: "This device" },
        { key: "hal9000-7680", label: "hal9000" },
        { key: "okra-7680", label: "okra" },
      ],
    });

    gallery.filter = "hal9000-7680";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["shared.png", "remote-only.png"]);
    expect(gallery.filtered[0]).toMatchObject({
      sourceKey: "hal9000-7680",
      availableOn: [{ key: "hal9000-7680", label: "hal9000" }],
    });
  });

  it("dedupes matching copies across remote hosts without a local copy", async () => {
    connectLocalPlusHal();
    addExtra();
    localSnapshot([]);
    vi.mocked(apiJsonTo).mockImplementation((target) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000")) return Promise.resolve([img("shared.png", 202)]) as never;
      if (url.includes("okra")) return Promise.resolve([img("shared.png", 201)]) as never;
      return Promise.resolve([]) as never;
    });
    const gallery = useGalleryStore();

    await gallery.fetchAll();

    expect(gallery.merged).toHaveLength(1);
    expect(gallery.merged[0]).toMatchObject({
      sourceKey: "hal9000-7680",
      availableOn: [
        { key: "hal9000-7680", label: "hal9000" },
        { key: "okra-7680", label: "okra" },
      ],
    });
  });
});

describe("identity dedupe", () => {
  const identified = (
    filename: string,
    timestamp: number,
    seed: number,
    size: number,
  ): GalleryImage =>
    ({ filename, timestamp, size_bytes: size, metadata: { prompt: "p", seed } }) as never;

  it("collapses same-print copies whose filenames differ via seed + byte identity", async () => {
    connectLocalPlusHal();
    localSnapshot([
      identified("ltx-video-mirror.mp4", 301, 42, 9_000),
      identified("unrelated.png", 100, 7, 1_234),
    ]);
    vi.mocked(apiJsonTo).mockImplementation((target) => {
      const url = (target as { baseUrl: string }).baseUrl;
      // The same LTX print: saved on hal9000 under the server's name, and
      // mirrored locally by an old auto-save that invented its own name.
      if (url.includes("hal9000"))
        return Promise.resolve([identified("ltx-video-9000.mp4", 300, 42, 9_000)]) as never;
      return Promise.resolve([
        identified("ltx-video-mirror.mp4", 301, 42, 9_000),
        identified("unrelated.png", 100, 7, 1_234),
      ]) as never;
    });
    const gallery = useGalleryStore();
    await gallery.fetchAll();

    expect(gallery.merged).toHaveLength(2);
    const print = gallery.merged[0]!;
    // One logical print, the local copy preferred, both locations listed.
    expect(print.sourceKey).toBe("local");
    expect(print.availableOn.map((s) => s.key).sort()).toEqual(["hal9000-7680", "local"]);
  });

  it("recovers the seed from auto-save filenames for synthesized video rows", async () => {
    connectLocalPlusHal();
    // The origin recorded real metadata; the old local mirror's row was
    // synthesized (videos embed nothing) with seed 0 — but the desktop's
    // auto-save filename encodes the real seed.
    const origin = {
      filename: "mold-ltx-2-3-22b-dev-fp8-1721312345678.mp4",
      timestamp: 300,
      size_bytes: 9_000,
      metadata: { prompt: "a cheetah", seed: 852494036 },
    } as never as GalleryImage;
    const mirror = {
      filename: "mold-ltx-2-3-22b-dev-fp8-852494036-1721312349999.mp4",
      timestamp: 301,
      size_bytes: 9_000,
      metadata_synthetic: true,
      metadata: { prompt: "", seed: 0 },
    } as never as GalleryImage;
    localSnapshot([mirror]);
    vi.mocked(apiJsonTo).mockImplementation((target) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000")) return Promise.resolve([origin]) as never;
      return Promise.resolve([mirror]) as never;
    });
    const gallery = useGalleryStore();
    await gallery.fetchAll();

    expect(gallery.merged).toHaveLength(1);
    expect(gallery.merged[0]!.availableOn.map((s) => s.key).sort()).toEqual([
      "hal9000-7680",
      "local",
    ]);
  });

  it("opts synthesized rows without a parseable filename out of identity", async () => {
    connectLocalPlusHal();
    // Both rows are synthetic with the placeholder seed 0 and equal sizes —
    // without a real seed recovered from the filename they must never merge.
    const synth = (filename: string): GalleryImage =>
      ({
        filename,
        timestamp: 300,
        size_bytes: 9_000,
        metadata_synthetic: true,
        metadata: { prompt: "", seed: 0, model: "unknown" },
      }) as never;
    localSnapshot([synth("clip-b.mp4")]);
    vi.mocked(apiJsonTo).mockImplementation((target) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000")) return Promise.resolve([synth("clip-a.mp4")]) as never;
      return Promise.resolve([synth("clip-b.mp4")]) as never;
    });
    const gallery = useGalleryStore();
    await gallery.fetchAll();
    expect(gallery.merged).toHaveLength(2);
  });

  it("never collapses different models that share a seed and byte size", async () => {
    connectLocalPlusHal();
    const withModel = (filename: string, model: string): GalleryImage =>
      ({
        filename,
        timestamp: 300,
        size_bytes: 9_000,
        metadata: { prompt: "p", seed: 42, model },
      }) as never;
    localSnapshot([withModel("b.png", "sd15:fp16")]);
    vi.mocked(apiJsonTo).mockImplementation((target) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000"))
        return Promise.resolve([withModel("a.png", "flux-dev:q8")]) as never;
      return Promise.resolve([withModel("b.png", "sd15:fp16")]) as never;
    });
    const gallery = useGalleryStore();
    await gallery.fetchAll();
    expect(gallery.merged).toHaveLength(2);
  });

  it("never collapses identity matches written far apart (re-generations)", async () => {
    connectLocalPlusHal();
    localSnapshot([identified("b.png", 27_200, 42, 9_000)]);
    vi.mocked(apiJsonTo).mockImplementation((target) => {
      const url = (target as { baseUrl: string }).baseUrl;
      // Same seed, size, and model — but two days apart: a re-generation,
      // not a mirror.
      if (url.includes("hal9000"))
        return Promise.resolve([identified("a.png", 200_000, 42, 9_000)]) as never;
      return Promise.resolve([identified("b.png", 27_200, 42, 9_000)]) as never;
    });
    const gallery = useGalleryStore();
    await gallery.fetchAll();
    expect(gallery.merged).toHaveLength(2);
  });

  it("never collapses rows that lack a seed or byte size", async () => {
    connectLocalPlusHal();
    localSnapshot([img("b.png", 301)]);
    vi.mocked(apiJsonTo).mockImplementation((target) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000")) return Promise.resolve([img("a.png", 300)]) as never;
      return Promise.resolve([img("b.png", 301)]) as never;
    });
    const gallery = useGalleryStore();
    await gallery.fetchAll();
    expect(gallery.merged).toHaveLength(2);
  });

  it("existsLocally spots a local copy by identity even on host-chip tiles", async () => {
    connectLocalPlusHal();
    localSnapshot([identified("ltx-video-mirror.mp4", 301, 42, 9_000)]);
    vi.mocked(apiJsonTo).mockImplementation((target) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000"))
        return Promise.resolve([
          identified("ltx-video-9000.mp4", 300, 42, 9_000),
          identified("remote-only.mp4", 200, 8, 5_000),
        ]) as never;
      return Promise.resolve([identified("ltx-video-mirror.mp4", 301, 42, 9_000)]) as never;
    });
    const gallery = useGalleryStore();
    await gallery.fetchAll();

    gallery.filter = "hal9000-7680";
    const [mirrored, remoteOnly] = gallery.filtered;
    // Host-chip tiles only carry their own bucket in availableOn — the
    // local probe still recognizes the mirrored copy.
    expect(mirrored!.availableOn.map((s) => s.key)).toEqual(["hal9000-7680"]);
    expect(gallery.existsLocally(mirrored!)).toBe(true);
    expect(gallery.existsLocally(remoteOnly!)).toBe(false);
  });
});

describe("per-bucket isolation", () => {
  it("one host failing never breaks the others", async () => {
    connectLocalPlusHal();
    addExtra();
    vi.mocked(ipc.localGalleryList).mockResolvedValue({
      images: [img("l1.png", 20)],
      target: { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
    });
    vi.mocked(apiJsonTo).mockImplementation((target, _path) => {
      const url = (target as { baseUrl: string }).baseUrl;
      if (url.includes("hal9000")) return Promise.reject(new Error("connection refused"));
      if (url.includes("okra")) return Promise.resolve([img("o1.png", 10)]) as never;
      return Promise.resolve([img("l1.png", 20)]) as never; // built-in engine
    });
    const gallery = useGalleryStore();

    await gallery.fetchAll();

    expect(gallery.buckets["hal9000-7680"]!.error).toContain("connection refused");
    expect(gallery.buckets["hal9000-7680"]!.loaded).toBe(false);
    expect(gallery.buckets["okra-7680"]!.loaded).toBe(true);
    expect(gallery.merged.map((e) => e.item.filename)).toEqual(["l1.png", "o1.png"]);
  });

  it("uses a remote host's changed endpoint and key on the next refresh", async () => {
    addExtra("hal9000-7680", "http://hal9000:7680", "old-key");
    vi.mocked(apiJsonTo).mockResolvedValue([img("same.png", 10)]);
    const gallery = useGalleryStore();
    await gallery.fetchBucket("hal9000-7680");

    const host = useHostsStore().extras[0]!;
    host.url = "https://hal9000.tailnet.ts.net:8443";
    host.apiKey = "new-key";
    await gallery.fetchBucket("hal9000-7680");

    expect(apiJsonTo).toHaveBeenLastCalledWith(
      { baseUrl: "https://hal9000.tailnet.ts.net:8443", apiKey: "new-key" },
      "/api/gallery",
    );
    expect(evictMedia).toHaveBeenCalledWith("/api/gallery/image/same.png", "hal9000-7680");
  });
});

describe("live server events", () => {
  it("applyAdded inserts into the primary host's bucket only", () => {
    connectLocalPlusHal();
    const gallery = useGalleryStore();
    // The primary is always the built-in engine (host id "local").
    gallery.buckets["local"] = loadedBucket([img("old.png", 100)]);
    gallery.buckets["hal9000-7680"] = loadedBucket([]);

    gallery.applyAdded({
      type: "gallery_added",
      filename: "new.png",
      image: img("new.png", 200),
    });

    expect(gallery.buckets["local"]!.items.map((i) => i.filename)).toEqual(["new.png", "old.png"]);
    expect(gallery.buckets["hal9000-7680"]!.items).toHaveLength(0);
  });

  it("applyAdded dedupes by filename", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("a.png", 100)]);

    gallery.applyAdded({ type: "gallery_added", filename: "a.png", image: img("a.png", 100) });

    expect(gallery.buckets["local"]!.items).toHaveLength(1);
  });

  it("applyAdded is a no-op while the primary bucket is not loaded", () => {
    connectLocal();
    const gallery = useGalleryStore();
    // The primary "local" bucket was never opened — the frame must not create it.
    gallery.applyAdded({ type: "gallery_added", filename: "x.png", image: img("x.png", 1) });

    expect(gallery.buckets["local"]).toBeUndefined();
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
      expect(ipc.localGalleryList).toHaveBeenCalledTimes(1);
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

describe("local bucket while the engine is down", () => {
  it("keeps the This-device bucket and routes it over IPC when the local host isn't ready", async () => {
    // Local engine errored at start; a remote host is still connected. The
    // local server exposes a stale baseUrl, so target presence is NOT enough
    // — the "local" key must fall back to native IPC.
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "error";
    addExtra("hal9000-7680", "http://hal9000:7680", "hk");
    vi.mocked(ipc.localGalleryList).mockResolvedValue({
      images: [img("saved-remote-print.png", 50)],
      target: null,
    });
    const gallery = useGalleryStore();

    expect(gallery.sources.map((s) => s.key)).toEqual(["local", "hal9000-7680"]);

    await gallery.fetchAll();

    expect(ipc.localGalleryList).toHaveBeenCalled();
    // The dead local server must not be hit for the local bucket.
    expect(apiJsonTo).not.toHaveBeenCalledWith(
      { baseUrl: "http://127.0.0.1:49152", apiKey: null },
      "/api/gallery",
    );
    expect(gallery.merged.some((e) => e.item.filename === "saved-remote-print.png")).toBe(true);
    expect(gallery.mediaSourceOf("local")).toBe("local");
    expect(gallery.targetOf("local")).toBeNull();
  });

  it("lists This device even when there is no primary connection at all", () => {
    const gallery = useGalleryStore();
    expect(gallery.sources.map((s) => s.key)).toEqual(["local"]);
  });

  it("refreshHost('local') works over IPC while the engine is down (auto-save nudge)", async () => {
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([]);
    vi.mocked(ipc.localGalleryList).mockResolvedValue({
      images: [img("just-saved.png", 9)],
      target: null,
    });

    await gallery.refreshHost("local");

    expect(ipc.localGalleryList).toHaveBeenCalledTimes(1);
    expect(gallery.buckets["local"]!.items.map((i) => i.filename)).toEqual(["just-saved.png"]);
  });

  it("keeps the IPC-proven running-server target when host status is stale", async () => {
    const conn = useConnectionStore();
    conn.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:49152",
      apiKey: "desktop-key",
    };
    conn.status = "error";
    vi.mocked(ipc.localGalleryList).mockResolvedValue({
      images: [img("clip.mp4", 9)],
      target: { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
    });
    vi.mocked(apiFetchTo).mockResolvedValue(new Response(null, { status: 204 }));
    const gallery = useGalleryStore();

    await gallery.fetchBucket("local");

    expect(gallery.mediaSourceOf("local")).toBe("host");
    expect(gallery.targetOf("local")).toEqual({
      baseUrl: "http://127.0.0.1:49152",
      apiKey: "desktop-key",
    });
    await gallery.remove("local", "clip.mp4");
    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
      "/api/gallery/image/clip.mp4",
      { method: "DELETE" },
    );
    expect(ipc.localGalleryDelete).not.toHaveBeenCalled();
  });

  it("refreshes Running to Off authority and evicts server-addressed media", async () => {
    const gallery = useGalleryStore();
    vi.mocked(ipc.localGalleryList)
      .mockResolvedValueOnce({
        images: [img("print.png", 9)],
        target: { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
      })
      .mockResolvedValueOnce({ images: [img("print.png", 9)], target: null });

    await gallery.fetchBucket("local");
    expect(gallery.mediaSourceOf("local")).toBe("host");
    await gallery.fetchBucket("local");

    expect(gallery.mediaSourceOf("local")).toBe("local");
    expect(gallery.targetOf("local")).toBeNull();
    expect(evictMedia).toHaveBeenCalledWith("/api/gallery/image/print.png", "local");
  });

  it("refreshes Off to Running authority and evicts native-protocol media", async () => {
    const gallery = useGalleryStore();
    vi.mocked(ipc.localGalleryList)
      .mockResolvedValueOnce({ images: [img("print.png", 9)], target: null })
      .mockResolvedValueOnce({
        images: [img("print.png", 9)],
        target: { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
      });

    await gallery.fetchBucket("local");
    expect(gallery.mediaSourceOf("local")).toBe("local");
    await gallery.fetchBucket("local");

    expect(gallery.mediaSourceOf("local")).toBe("host");
    expect(gallery.targetOf("local")).toEqual({
      baseUrl: "http://127.0.0.1:49152",
      apiKey: "desktop-key",
    });
    expect(evictMedia).toHaveBeenCalledWith("mold-local://localhost/print.png", "local");
  });

  it("returns a server-backed snapshot the moment the local server is ready again", async () => {
    connectLocal();
    const gallery = useGalleryStore();

    await gallery.fetchBucket("local");

    expect(ipc.localGalleryList).toHaveBeenCalledTimes(1);
    expect(gallery.targetOf("local")).toEqual({
      baseUrl: "http://127.0.0.1:49152",
      apiKey: "desktop-key",
    });
  });
});

describe("remove", () => {
  it("deletes prints in a host-less local bucket through native IPC", async () => {
    // With no connected local host (engine down), the "local" bucket falls
    // back to native IPC for deletes and mold-local:// media.
    vi.mocked(ipc.localGalleryDelete).mockResolvedValue();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("print.png", 1)]);

    await gallery.remove("local", "print.png");

    expect(ipc.localGalleryDelete).toHaveBeenCalledWith("print.png");
    expect(apiFetchTo).not.toHaveBeenCalled();
    expect(gallery.buckets["local"]!.items).toHaveLength(0);
    expect(evictMedia).toHaveBeenCalledWith("mold-local://localhost/print.png", "local");
  });

  it("deletes host prints against that host and evicts its cache bucket", async () => {
    connectLocalPlusHal();
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
    localSnapshot([img("kept.png", 2)]);

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

    expect(ipc.localGalleryList).toHaveBeenCalledTimes(1);
  });

  it("never force-loads a bucket from a background event", async () => {
    connectLocal();
    const gallery = useGalleryStore();

    await gallery.refreshHost("local");

    expect(apiJsonTo).not.toHaveBeenCalled();
    expect(gallery.buckets["local"]).toBeUndefined();
  });
});

describe("kind + text filtering", () => {
  /** A print with controllable format/metadata for kind + search tests. */
  const media = (
    filename: string,
    timestamp: number,
    meta: Record<string, unknown> = {},
    format: string | null = null,
  ): GalleryImage =>
    ({ filename, timestamp, format, metadata: { prompt: "p", model: "m", ...meta } }) as never;

  it("filters the merged grid by media kind", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([
      media("still.png", 40),
      media("clip.mp4", 30),
      media("named-mp4.mp4", 20),
      media("frames.png", 10, { video_frames: 97 }),
    ]);

    gallery.mediaKind = "video";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual([
      "clip.mp4",
      "named-mp4.mp4",
      "frames.png",
    ]);

    gallery.mediaKind = "image";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["still.png"]);

    gallery.mediaKind = "all";
    expect(gallery.filtered).toHaveLength(4);
  });

  it("filters by a case-insensitive query over filename, model, and prompt", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([
      media("Sunset-01.png", 40, { prompt: "a beach", model: "flux-dev" }),
      media("b.png", 30, { prompt: "a CAT in a hat", model: "flux-dev" }),
      media("c.png", 20, { prompt: "a dog", model: "z-image-turbo" }),
    ]);

    gallery.query = "sunset";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["Sunset-01.png"]);

    gallery.query = "cat";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["b.png"]);

    gallery.query = "z-image";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["c.png"]);

    gallery.query = "  ";
    expect(gallery.filtered).toHaveLength(3);
  });

  it("chains host chip, kind, and query", () => {
    connectLocalPlusHal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([media("local-cat.mp4", 40, { prompt: "cat" })]);
    gallery.buckets["hal9000-7680"] = loadedBucket([
      media("hal-cat.mp4", 30, { prompt: "cat" }),
      media("hal-cat.png", 20, { prompt: "cat" }),
      media("hal-dog.mp4", 10, { prompt: "dog" }),
    ]);

    gallery.filter = "hal9000-7680";
    gallery.mediaKind = "video";
    gallery.query = "cat";
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["hal-cat.mp4"]);
  });

  it("kindCounts reports per-kind buckets and respects the host filter", () => {
    connectLocalPlusHal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([media("l1.png", 40), media("l2.mp4", 30)]);
    gallery.buckets["hal9000-7680"] = loadedBucket([
      media("h1.png", 20),
      media("h2.png", 10),
      media("h3.mp4", 5),
    ]);

    expect(gallery.kindCounts).toEqual({ all: 5, image: 3, video: 2 });

    gallery.filter = "hal9000-7680";
    expect(gallery.kindCounts).toEqual({ all: 3, image: 2, video: 1 });
  });

  it("kindCounts ignores the kind and query narrowing (chip labels stay stable)", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([media("a.png", 2), media("b.mp4", 1)]);

    gallery.mediaKind = "video";
    gallery.query = "zzz-no-match";
    expect(gallery.kindCounts).toEqual({ all: 2, image: 1, video: 1 });
  });

  it("hostFiltered exposes the chip-only set (History runs must not inherit gallery search)", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([media("a.png", 2), media("b.mp4", 1)]);

    gallery.mediaKind = "video";
    gallery.query = "zzz-no-match";
    expect(gallery.filtered).toHaveLength(0);
    expect(gallery.hostFiltered).toHaveLength(2);
  });
});

describe("removeMany", () => {
  it("routes each delete per origin exactly like single remove and reports partial failure", async () => {
    // Engine down: the "local" bucket routes over IPC; hal9000 over HTTP —
    // the same mixed-origin routing the single-delete path uses.
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "error";
    addExtra("hal9000-7680", "http://hal9000:7680", "hk");
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("l.png", 30)]);
    gallery.buckets["hal9000-7680"] = loadedBucket([img("h1.png", 20), img("h2.png", 10)]);

    vi.mocked(ipc.localGalleryDelete).mockResolvedValue();
    vi.mocked(apiFetchTo).mockImplementation((_target, path) => {
      if (String(path).includes("h2.png")) return Promise.reject(new Error("500"));
      return Promise.resolve(new Response(null, { status: 200 })) as never;
    });
    // The post-failure refetch reflects the server's real state: h2 survived.
    vi.mocked(apiJsonTo).mockResolvedValue([img("h2.png", 10)]);

    const result = await gallery.removeMany([
      { sourceKey: "local", filename: "l.png" },
      { sourceKey: "hal9000-7680", filename: "h1.png" },
      { sourceKey: "hal9000-7680", filename: "h2.png" },
    ]);

    expect(result).toEqual({ deleted: 2, failed: 1 });
    expect(ipc.localGalleryDelete).toHaveBeenCalledWith("l.png");
    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "hk" },
      "/api/gallery/image/h1.png",
      { method: "DELETE" },
    );
    expect(gallery.buckets["local"]!.items).toHaveLength(0);
    // The failed origin's bucket was refetched to reconverge with the server.
    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "hk" },
      "/api/gallery",
    );
    expect(gallery.buckets["hal9000-7680"]!.items.map((i) => i.filename)).toEqual(["h2.png"]);
  });

  it("does not refetch any bucket when every delete succeeds", async () => {
    connectLocalPlusHal();
    const gallery = useGalleryStore();
    gallery.buckets["hal9000-7680"] = loadedBucket([img("h1.png", 20), img("h2.png", 10)]);
    vi.mocked(apiFetchTo).mockResolvedValue(new Response(null, { status: 200 }) as never);

    const result = await gallery.removeMany([
      { sourceKey: "hal9000-7680", filename: "h1.png" },
      { sourceKey: "hal9000-7680", filename: "h2.png" },
    ]);

    expect(result).toEqual({ deleted: 2, failed: 0 });
    expect(apiJsonTo).not.toHaveBeenCalled();
    expect(gallery.buckets["hal9000-7680"]!.items).toHaveLength(0);
  });
});

describe("delete everywhere", () => {
  it("deletes every matching device copy when one merged print is selected", async () => {
    connectLocalPlusHal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("shared.png", 20)]);
    gallery.buckets["hal9000-7680"] = loadedBucket([img("shared.png", 19)]);
    vi.mocked(apiFetchTo).mockResolvedValue(new Response(null, { status: 204 }) as never);

    const result = await gallery.removeEntriesEverywhere([gallery.merged[0]!]);

    expect(result).toEqual({ deletedPrints: 1, failedPrints: 0, deletedCopies: 2 });
    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" },
      "/api/gallery/image/shared.png",
      { method: "DELETE" },
    );
    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "hk" },
      "/api/gallery/image/shared.png",
      { method: "DELETE" },
    );
    expect(gallery.merged).toHaveLength(0);
  });

  it("holds every matching copy during the single-print undo window", () => {
    connectLocalPlusHal();
    const gallery = useGalleryStore();
    gallery.buckets["local"] = loadedBucket([img("shared.png", 20)]);
    gallery.buckets["hal9000-7680"] = loadedBucket([img("shared.png", 19)]);

    const locations = gallery.beginDeleteEverywhere(gallery.merged[0]!);
    expect(locations).toEqual([
      { sourceKey: "local", filename: "shared.png" },
      { sourceKey: "hal9000-7680", filename: "shared.png" },
    ]);
    expect(gallery.merged).toHaveLength(0);

    gallery.cancelDeleteEverywhere(locations);
    expect(gallery.merged).toHaveLength(1);
  });
});

describe("bucket sync", () => {
  it("drops buckets whose source disappeared and evicts their media", async () => {
    connectLocalPlusHal();
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

describe("new-since-last-visit count (G11 nav badge)", () => {
  it("is zero until the Library has been visited once", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets.local = loadedBucket([img("a.png", 2), img("b.png", 1)]);
    // Never visited → the first visit only establishes the baseline.
    expect(gallery.newCount).toBe(0);
  });

  it("counts prints developed since the last visit and resets on the next", () => {
    connectLocal();
    const gallery = useGalleryStore();
    gallery.buckets.local = loadedBucket([img("a.png", 1)]);
    gallery.markLibrarySeen();
    expect(gallery.newCount).toBe(0);

    gallery.buckets.local.items = [img("b.png", 3), img("c.png", 2), img("a.png", 1)];
    expect(gallery.newCount).toBe(2);

    gallery.markLibrarySeen();
    expect(gallery.newCount).toBe(0);
  });
});

describe("undoable single-print delete (G12)", () => {
  it("hides a print optimistically, restores on undo, and only DELETEs on commit", async () => {
    // No connected host → the "local" source falls back to native IPC delete.
    const gallery = useGalleryStore();
    gallery.buckets.local = loadedBucket([img("a.png", 2), img("b.png", 1)]);

    gallery.beginDelete("local", "a.png");
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["b.png"]);
    expect(vi.mocked(ipc.localGalleryDelete)).not.toHaveBeenCalled();

    // Undo: no server call, the print returns to the grid.
    gallery.cancelDelete("local", "a.png");
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["a.png", "b.png"]);

    // Commit: the real DELETE fires and the row leaves the bucket.
    gallery.beginDelete("local", "a.png");
    await gallery.commitDelete("local", "a.png");
    expect(vi.mocked(ipc.localGalleryDelete)).toHaveBeenCalledWith("a.png");
    expect(gallery.buckets.local!.items.map((i) => i.filename)).toEqual(["b.png"]);
    expect(gallery.pendingDeletions.size).toBe(0);
  });

  it("restores the print when a committed delete fails", async () => {
    vi.mocked(ipc.localGalleryDelete).mockRejectedValueOnce(new Error("nope"));
    const gallery = useGalleryStore();
    gallery.buckets.local = loadedBucket([img("a.png", 1)]);

    gallery.beginDelete("local", "a.png");
    await expect(gallery.commitDelete("local", "a.png")).rejects.toThrow("nope");
    // The pending mark cleared, so the surviving bucket row is visible again.
    expect(gallery.pendingDeletions.size).toBe(0);
    expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["a.png"]);
  });

  it("ignores a commit for a print that was already undone", async () => {
    const gallery = useGalleryStore();
    gallery.buckets.local = loadedBucket([img("a.png", 1)]);
    await gallery.commitDelete("local", "a.png"); // never began
    expect(vi.mocked(ipc.localGalleryDelete)).not.toHaveBeenCalled();
  });
});
