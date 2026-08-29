/**
 * Refetch discipline: a poll that comes back 304 must not invalidate the
 * merged grid, and gallery rows are raw (non-reactive) snapshots that are
 * replaced, never mutated, by optimistic organization edits.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { isReactive, toRaw } from "vue";
import { apiFetchTo, conditionalApiJsonTo } from "../lib/api/client";
import { ipc } from "../lib/ipc";
import { evictMedia } from "../lib/gallery/media";
import { useConnectionStore } from "./connection";
import { useGalleryStore } from "./gallery";
import { useHostsStore } from "./hosts";
import type { GalleryImage } from "../lib/api/types";
import * as organization from "@studio/api/galleryOrganization";

vi.mock("../lib/ipc", () => ({
  ipc: {
    localGalleryList: vi.fn().mockResolvedValue({ images: [], target: null }),
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
vi.mock("@studio/api/galleryOrganization", () => ({
  organizeGallery: vi.fn().mockResolvedValue(undefined),
  patchGalleryImage: vi.fn().mockResolvedValue(null),
  mutateGalleryBulk: vi.fn(),
  listCollections: vi.fn(),
  listTags: vi.fn(),
  listTrash: vi.fn(),
}));

const img = (filename: string, timestamp: number): GalleryImage =>
  ({
    filename,
    timestamp,
    size_bytes: 10,
    metadata: { prompt: "p", model: "m", seed: 1 },
  }) as never;

function connectPlato() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().extras.push({
    id: "plato-7680",
    label: "plato",
    url: "http://plato:7680",
    apiKey: "pk",
    status: "ready",
    error: null,
    instanceId: null,
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
});

describe("gallery refetch discipline", () => {
  it("leaves the bucket and every derived index untouched when the listing answers 304", async () => {
    connectPlato();
    const gallery = useGalleryStore();
    const listing = [img("a.png", 2), img("b.png", 1)];
    vi.mocked(conditionalApiJsonTo).mockResolvedValue(listing);
    await gallery.fetchBucket("plato-7680");
    const merged = gallery.merged;
    const index = gallery.organizationIndex;
    const rows = gallery.buckets["plato-7680"]!.items;
    vi.mocked(evictMedia).mockClear();

    // `conditionalApiJsonTo` hands back the cached array itself on 304.
    await gallery.fetchBucket("plato-7680");

    expect(gallery.buckets["plato-7680"]!.items).toBe(rows);
    expect(gallery.merged).toBe(merged);
    expect(gallery.organizationIndex).toBe(index);
    expect(evictMedia).not.toHaveBeenCalled();
  });

  it("forgets a permanently deleted print's tiles under the same version the tile was filed as", async () => {
    connectPlato();
    const gallery = useGalleryStore();
    // An older host reports no media_version: the tile key used the
    // `timestamp:size` fallback, so the eviction must too.
    vi.mocked(conditionalApiJsonTo).mockResolvedValue([img("legacy.png", 5)]);
    vi.mocked(apiFetchTo).mockResolvedValue(new Response(null, { status: 204 }));
    await gallery.fetchBucket("plato-7680");
    await gallery.remove("plato-7680", "legacy.png", { permanent: true });
    expect(ipc.forgetGalleryThumbnail).toHaveBeenCalledWith(
      "plato-7680",
      "http://plato:7680",
      "legacy.png",
      "5:10",
    );
    // A trash (non-permanent) delete keeps the tile for the Trash grid.
    vi.mocked(ipc.forgetGalleryThumbnail).mockClear();
    vi.mocked(conditionalApiJsonTo).mockResolvedValue([img("kept.png", 6)]);
    await gallery.fetchBucket("plato-7680");
    await gallery.remove("plato-7680", "kept.png");
    expect(ipc.forgetGalleryThumbnail).not.toHaveBeenCalled();
  });

  it("keeps rows raw and replaces them on an optimistic edit", async () => {
    connectPlato();
    useHostsStore().capabilities["plato-7680"] = {
      gallery: { can_delete: true, organize: true },
    } as never;
    const gallery = useGalleryStore();
    vi.mocked(conditionalApiJsonTo).mockResolvedValue([img("a.png", 2)]);
    await gallery.fetchBucket("plato-7680");
    const before = gallery.buckets["plato-7680"]!.items[0]!;
    expect(isReactive(before)).toBe(false);
    expect(gallery.organizationOf(gallery.merged[0]!).favorite).toBe(false);

    await gallery.runOrganizationOp({
      kind: "setFavorite",
      hostId: "plato-7680",
      filenames: ["a.png"],
      favorite: true,
    });
    expect(organization.organizeGallery).toHaveBeenCalledTimes(1);
    const after = gallery.buckets["plato-7680"]!.items[0]!;
    expect(after).not.toBe(before);
    expect(toRaw(after).favorite).toBe(true);
    expect(isReactive(after)).toBe(false);
    // The index saw the replacement.
    expect(gallery.organizationOf(gallery.merged[0]!).favorite).toBe(true);
  });
});
