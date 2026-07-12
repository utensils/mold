import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { ipc } from "../lib/ipc";
import { useGalleryStore } from "./gallery";

vi.mock("../lib/ipc", () => ({
  ipc: {
    localGalleryList: vi.fn(),
    localGalleryDelete: vi.fn(),
  },
}));

vi.mock("../lib/api/client", () => ({
  apiJson: vi.fn(),
  apiFetch: vi.fn(),
}));

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
});

describe("gallery source", () => {
  it("loads This Mac without changing the engine target", async () => {
    vi.mocked(ipc.localGalleryList).mockResolvedValue([]);
    const gallery = useGalleryStore();

    await gallery.fetch("local");

    expect(gallery.source).toBe("local");
    expect(ipc.localGalleryList).toHaveBeenCalledOnce();
    expect(gallery.loaded).toBe(true);
  });

  it("deletes local prints through native IPC", async () => {
    vi.mocked(ipc.localGalleryDelete).mockResolvedValue();
    const gallery = useGalleryStore();
    gallery.source = "local";
    gallery.items = [{ filename: "print.png" }] as never;

    await gallery.remove("print.png");

    expect(ipc.localGalleryDelete).toHaveBeenCalledWith("print.png");
    expect(gallery.items).toHaveLength(0);
  });
});

describe("live server events", () => {
  const img = (filename: string, timestamp: number) =>
    ({ filename, timestamp, metadata: { prompt: "p" } }) as never;

  it("applyAdded inserts a carried row newest-first without a refetch", () => {
    const gallery = useGalleryStore();
    gallery.source = "engine";
    gallery.loaded = true;
    gallery.items = [img("old.png", 100), img("older.png", 50)] as never;

    gallery.applyAdded({ type: "gallery_added", filename: "new.png", image: img("new.png", 200) });

    expect(gallery.items.map((i) => i.filename)).toEqual(["new.png", "old.png", "older.png"]);
  });

  it("applyAdded dedupes by filename", () => {
    const gallery = useGalleryStore();
    gallery.source = "engine";
    gallery.loaded = true;
    gallery.items = [img("a.png", 100)] as never;

    gallery.applyAdded({ type: "gallery_added", filename: "a.png", image: img("a.png", 100) });

    expect(gallery.items).toHaveLength(1);
  });

  it("applyAdded is a no-op while showing the local source", () => {
    const gallery = useGalleryStore();
    gallery.source = "local";
    gallery.loaded = true;
    gallery.items = [];

    gallery.applyAdded({ type: "gallery_added", filename: "x.png", image: img("x.png", 1) });

    expect(gallery.items).toHaveLength(0);
  });

  it("applyAdded without a row falls back to a debounced refetch", async () => {
    vi.useFakeTimers();
    try {
      const { apiJson } = await import("../lib/api/client");
      vi.mocked(apiJson).mockResolvedValue([]);
      const gallery = useGalleryStore();
      gallery.source = "engine";
      gallery.loaded = true;

      gallery.applyAdded({ type: "gallery_added", filename: "a.png" });
      gallery.applyAdded({ type: "gallery_added", filename: "b.png" });
      expect(apiJson).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(600);
      // Two events inside the window collapse into one refetch.
      expect(apiJson).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("applyRemoved drops the tile", () => {
    const gallery = useGalleryStore();
    gallery.source = "engine";
    gallery.loaded = true;
    gallery.items = [img("a.png", 100), img("b.png", 50)] as never;

    gallery.applyRemoved("a.png");

    expect(gallery.items.map((i) => i.filename)).toEqual(["b.png"]);
  });
});
