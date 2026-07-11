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
