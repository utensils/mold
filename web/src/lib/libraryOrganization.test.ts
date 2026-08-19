import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HostEntry } from "./hostRegistry";
import type { HostGalleryImage } from "./multiHostGallery";
import type { Collection, GalleryImage } from "../types";

const api = vi.hoisted(() => ({
  patchGalleryImage: vi.fn(async () => null),
  organizeGallery: vi.fn(async () => undefined),
  createCollection: vi.fn(async (_t: unknown, body: { name: string }) => ({
    id: `new-${body.name}`,
    name: body.name,
    slug: body.name.toLowerCase(),
    description: null,
    cover_filename: null,
    count: 0,
    created_at: 0,
    updated_at: 0,
  })),
  updateCollection: vi.fn(async () => ({})),
  deleteCollection: vi.fn(async () => undefined),
  setCollectionItems: vi.fn(async () => null),
  trashMany: vi.fn(async () => undefined),
  restoreTrashed: vi.fn(async () => undefined),
  deleteGalleryImageForever: vi.fn(async () => undefined),
  emptyTrash: vi.fn(async () => ({ purged: 2 })),
  listCollections: vi.fn(async () => []),
  listTags: vi.fn(async () => []),
}));

vi.mock("@studio/api/galleryOrganization", () => api);

import {
  applyOrganizationMutation,
  collectionCards,
  deleteCollectionEverywhere,
  downloadFilename,
  emptyTrashEverywhere,
  entryMatchesSearch,
  fetchOrganization,
  filterByOrganization,
  mergedCollections,
  mergedTags,
  renameCollectionEverywhere,
  retentionHosts,
  setCollectionCover,
  type HostOrganizationSnapshot,
} from "./libraryOrganization";
import { mergeLogicalEntries } from "./multiHostGallery";

const ORIGIN: HostEntry = {
  id: "origin",
  name: "this server",
  url: "http://localhost:7680",
};
const PLATO: HostEntry = {
  id: "plato",
  name: "plato",
  url: "http://plato:7680",
  apiKey: "plato-key",
};
const hostById = (id: string) =>
  id === "origin" ? ORIGIN : id === "plato" ? PLATO : null;

function collection(
  id: string,
  name: string,
  extra: Partial<Collection> = {},
): Collection {
  return {
    id,
    name,
    slug: name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    description: null,
    cover_filename: null,
    count: 1,
    created_at: 1,
    updated_at: 1,
    ...extra,
  };
}

function entry(
  hostId: string,
  filename: string,
  extra: Partial<GalleryImage> = {},
): HostGalleryImage {
  return {
    hostId,
    hostLabel: hostId,
    filename,
    timestamp: 100,
    format: "png",
    metadata: { prompt: "a print", model: "flux", seed: 1 } as never,
    ...extra,
  };
}

function snapshot(
  hostId: string,
  extra: Partial<HostOrganizationSnapshot> = {},
): HostOrganizationSnapshot {
  return {
    hostId,
    hostLabel: hostId,
    organize: true,
    trash: { enabled: true, retentionDays: 30 },
    collections: [],
    tags: [],
    trashed: [],
    ...extra,
  };
}

beforeEach(() => {
  for (const fn of Object.values(api)) fn.mockClear();
});

describe("fetchOrganization", () => {
  it("reads capabilities first and only asks capable hosts for collections, tags, and trash", async () => {
    const fetchers = {
      capabilities: vi.fn(async (host: HostEntry) =>
        host.id === "origin"
          ? {
              gallery: {
                can_delete: true,
                organize: true,
                trash: { enabled: true, retention_days: 7 },
              },
            }
          : { gallery: { can_delete: true } },
      ),
      collections: vi.fn(async () => [collection("c1", "Smurfs")]),
      tags: vi.fn(async () => [{ name: "blue", count: 2 }]),
      trash: vi.fn(async () => [
        { ...entry("origin", "old.png"), trashed_at: 10, purge_at: 20 },
      ]),
    };
    const snapshots = await fetchOrganization([ORIGIN, PLATO], fetchers);
    expect(snapshots[0]).toMatchObject({
      hostId: "origin",
      organize: true,
      trash: { enabled: true, retentionDays: 7 },
    });
    expect(snapshots[0]!.collections).toHaveLength(1);
    expect(snapshots[0]!.trashed[0]).toMatchObject({
      hostId: "origin",
      hostLabel: "this server",
      trashed_at: 10,
    });
    // An older host contributes nothing and is never probed for the rest.
    expect(snapshots[1]).toMatchObject({
      hostId: "plato",
      organize: false,
      trash: null,
      collections: [],
      tags: [],
      trashed: [],
    });
    expect(fetchers.collections).toHaveBeenCalledTimes(1);
    expect(fetchers.collections).toHaveBeenCalledWith(
      { baseUrl: "http://localhost:7680", apiKey: null },
      undefined,
    );
    expect(fetchers.trash).toHaveBeenCalledTimes(1);
  });

  it("degrades a failing probe to no organization instead of throwing", async () => {
    const snapshots = await fetchOrganization([ORIGIN], {
      capabilities: vi.fn(async () => {
        throw new Error("down");
      }),
      collections: vi.fn(async () => []),
      tags: vi.fn(async () => []),
      trash: vi.fn(async () => []),
    });
    expect(snapshots[0]).toMatchObject({ organize: false, trash: null });
  });
});

describe("merging", () => {
  it("merges collections by slug and tags by case-insensitive name", () => {
    const snapshots = [
      snapshot("origin", {
        collections: [collection("a", "Smurfs", { count: 3 })],
        tags: [
          { name: "Blue", count: 2 },
          { name: "keep", count: 1 },
        ],
      }),
      snapshot("plato", {
        collections: [
          collection("b", "smurfs", { count: 2 }),
          collection("c", "Rivers"),
        ],
        tags: [{ name: "blue", count: 5 }],
      }),
    ];
    const merged = mergedCollections(snapshots);
    expect(merged.map((c) => c.slug)).toEqual(["rivers", "smurfs"]);
    expect(merged[1]!.hosts.map((h) => h.hostId)).toEqual(["origin", "plato"]);
    expect(mergedTags(snapshots)).toEqual([
      { name: "Blue", count: 7 },
      { name: "keep", count: 1 },
    ]);
  });

  it("lists retention hosts origin-first and skips hosts without a trash", () => {
    const hosts = retentionHosts([
      snapshot("plato", { trash: { enabled: true, retentionDays: 7 } }),
      snapshot("legacy", { trash: null }),
      snapshot("origin", { trash: { enabled: true, retentionDays: 30 } }),
    ]);
    expect(hosts).toEqual([
      { label: "origin", retentionDays: 30 },
      { label: "plato", retentionDays: 7 },
    ]);
  });

  it("builds collection cards with logical counts, hosts, freshness, and covers", () => {
    const snapshots = [
      snapshot("origin", {
        collections: [
          collection("a", "Smurfs", {
            updated_at: 10,
            cover_filename: "s2.png",
          }),
        ],
      }),
      snapshot("plato", {
        collections: [collection("b", "Smurfs", { updated_at: 20 })],
      }),
    ];
    const raw = [
      entry("origin", "s1.png", { collections: ["a"], timestamp: 300 }),
      entry("origin", "s2.png", { collections: ["a"], timestamp: 200 }),
      entry("plato", "s3.png", { collections: ["b"], timestamp: 100 }),
      entry("origin", "loose.png", { timestamp: 400 }),
    ];
    const merged = mergedCollections(snapshots);
    const logical = mergeLogicalEntries(raw, {
      resolveCollectionSlug: (hostId, id) =>
        hostId === "origin" && id === "a"
          ? "smurfs"
          : hostId === "plato" && id === "b"
            ? "smurfs"
            : null,
    });
    const [card] = collectionCards(merged, logical, snapshots, raw);
    expect(card).toMatchObject({
      slug: "smurfs",
      name: "Smurfs",
      count: 3,
      hostLabels: ["origin", "plato"],
      updatedAt: 20,
    });
    expect(card!.covers.map((c) => c.filename)).toEqual([
      "s2.png",
      "s1.png",
      "s3.png",
    ]);
  });
});

describe("filtering", () => {
  const smurf = {
    ...entry("origin", "smurf.png", {
      favorite: true,
      tags: ["Smurf", "blue"],
      title: "Smurf 04",
    }),
    organization: {
      title: "Smurf 04",
      favorite: true,
      tags: ["blue", "Smurf"],
      collections: ["smurfs"],
      trashedAt: null,
      purgeAt: null,
      unresolvedCollectionIds: [],
    },
  };
  const frog = entry("origin", "frog.png", { tags: ["green"] });

  it("applies favorites, AND-ed tags, and collection membership", () => {
    expect(
      filterByOrganization([smurf, frog], { favoritesOnly: true }),
    ).toEqual([smurf]);
    expect(
      filterByOrganization([smurf, frog], { tags: ["BLUE", "smurf"] }),
    ).toEqual([smurf]);
    expect(
      filterByOrganization([smurf, frog], { tags: ["blue", "green"] }),
    ).toEqual([]);
    expect(
      filterByOrganization([smurf, frog], { collectionSlug: "smurfs" }),
    ).toEqual([smurf]);
  });

  it("searches title and tags as well as prompt, model, and filename", () => {
    expect(entryMatchesSearch(smurf, "smurf 04")).toBe(true);
    expect(entryMatchesSearch(frog, "GREEN")).toBe(true);
    expect(entryMatchesSearch(frog, "flux")).toBe(true);
    expect(entryMatchesSearch(frog, "nope")).toBe(false);
  });

  it("names the download after the title's slug when one exists", () => {
    expect(downloadFilename("Smurf 04!", "mold-flux-1.png")).toBe(
      "smurf-04.png",
    );
    expect(downloadFilename(null, "mold-flux-1.png")).toBe("mold-flux-1.png");
  });
});

describe("mutations fan out to every copy's host", () => {
  const copies = [
    entry("origin", "twin.png"),
    entry("plato", "twin.png"),
    entry("plato", "other.png"),
  ];
  const originTarget = { baseUrl: "http://localhost:7680", apiKey: null };
  const platoTarget = { baseUrl: "http://plato:7680", apiKey: "plato-key" };

  it("sets titles per copy and favorites in one bulk call per host", async () => {
    const context = { hostById, snapshots: [] };
    await applyOrganizationMutation(
      copies,
      { kind: "setTitle", title: "Twin" },
      context,
    );
    expect(api.patchGalleryImage).toHaveBeenCalledWith(
      originTarget,
      "twin.png",
      { title: "Twin" },
    );
    expect(api.patchGalleryImage).toHaveBeenCalledWith(
      platoTarget,
      "other.png",
      { title: "Twin" },
    );

    await applyOrganizationMutation(
      copies,
      { kind: "setFavorite", favorite: true },
      context,
    );
    expect(api.organizeGallery).toHaveBeenCalledWith(platoTarget, {
      filenames: ["twin.png", "other.png"],
      favorite: true,
    });
  });

  it("creates a missing collection by name before adding, reusing an existing one by slug", async () => {
    const snapshots = [
      snapshot("origin", { collections: [collection("c-origin", "Smurfs")] }),
      snapshot("plato"),
    ];
    const result = await applyOrganizationMutation(
      copies,
      { kind: "addToCollection", name: "Smurfs" },
      { hostById, snapshots },
    );
    expect(result.failed).toEqual([]);
    expect(api.createCollection).toHaveBeenCalledTimes(1);
    expect(api.createCollection).toHaveBeenCalledWith(platoTarget, {
      name: "Smurfs",
    });
    expect(api.setCollectionItems).toHaveBeenCalledWith(
      originTarget,
      "c-origin",
      { add: ["twin.png"], remove: [] },
    );
    expect(api.setCollectionItems).toHaveBeenCalledWith(
      platoTarget,
      "new-Smurfs",
      { add: ["twin.png", "other.png"], remove: [] },
    );
  });

  it("reports a missing host as a failure and keeps going elsewhere", async () => {
    const result = await applyOrganizationMutation(
      [...copies, entry("ghost", "gone.png")],
      { kind: "trash" },
      { hostById, snapshots: [] },
    );
    expect(result.ok.sort()).toEqual(["origin", "plato"]);
    expect(result.failed).toEqual([
      { hostId: "ghost", error: "That host isn't connected anymore." },
    ]);
    expect(api.trashMany).toHaveBeenCalledWith(platoTarget, [
      "twin.png",
      "other.png",
    ]);
  });

  it("restores, deletes forever, and empties the trash per host", async () => {
    await applyOrganizationMutation(
      copies,
      { kind: "restore" },
      { hostById, snapshots: [] },
    );
    expect(api.restoreTrashed).toHaveBeenCalledWith(originTarget, ["twin.png"]);
    await applyOrganizationMutation(
      copies,
      { kind: "deleteForever" },
      { hostById, snapshots: [] },
    );
    expect(api.deleteGalleryImageForever).toHaveBeenCalledTimes(3);
    await emptyTrashEverywhere(
      [
        snapshot("origin", { trashed: [entry("origin", "t.png")] }),
        snapshot("plato", { trashed: [] }),
      ],
      hostById,
    );
    expect(api.emptyTrash).toHaveBeenCalledTimes(1);
    expect(api.emptyTrash).toHaveBeenCalledWith(originTarget);
  });

  it("renames, deletes, and re-covers a merged collection on its hosts", async () => {
    const merged = mergedCollections([
      snapshot("origin", { collections: [collection("a", "Smurfs")] }),
      snapshot("plato", { collections: [collection("b", "Smurfs")] }),
    ])[0]!;
    await renameCollectionEverywhere(merged, "Blue folk", hostById);
    expect(api.updateCollection).toHaveBeenCalledWith(originTarget, "a", {
      name: "Blue folk",
    });
    expect(api.updateCollection).toHaveBeenCalledWith(platoTarget, "b", {
      name: "Blue folk",
    });
    await setCollectionCover(
      merged,
      { hostId: "plato", filename: "s3.png" },
      hostById,
    );
    expect(api.updateCollection).toHaveBeenLastCalledWith(platoTarget, "b", {
      cover_filename: "s3.png",
    });
    await deleteCollectionEverywhere(merged, hostById);
    expect(api.deleteCollection).toHaveBeenCalledWith(originTarget, "a");
    expect(api.deleteCollection).toHaveBeenCalledWith(platoTarget, "b");
  });
});
