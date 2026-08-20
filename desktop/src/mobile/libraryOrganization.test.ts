import { describe, expect, it, vi } from "vitest";
import type { Collection } from "@studio/lib/api/galleryOrganization";
import { collectionSlugResolver, planOrganizationFanout } from "@studio/lib/libraryOrganization";
import type { ServerCapabilities } from "../lib/api/types";
import {
  buildOrganizationIndex,
  collectionCards,
  collectionOnHost,
  deleteActionCopy,
  fanoutFailureMessage,
  filterLibraryPrints,
  libraryOrganizationSupport,
  logicalCopiesOf,
  mergedCollectionsFor,
  mergeHostTags,
  mergeTrashSnapshot,
  purgeChipLabel,
  requestTitle,
  reusedPrintTitle,
  runOrganizationFanout,
  selectionDeleteKind,
  tagChipPlan,
  trashRetentionHosts,
  validateCollectionName,
  type FanoutApi,
  type MobileGalleryImage,
} from "./libraryOrganization";

function capabilities(gallery: unknown): ServerCapabilities {
  return { gallery } as unknown as ServerCapabilities;
}

function copy(
  hostId: string,
  filename: string,
  extra: Partial<MobileGalleryImage> = {},
): MobileGalleryImage & { hostId: string } {
  return {
    hostId,
    filename,
    timestamp: 1_700_000_000,
    format: "png",
    size_bytes: 1234,
    metadata: {
      prompt: "a lighthouse",
      model: "flux-dev:q8",
      seed: 7,
      steps: 4,
      guidance: 3.5,
      width: 512,
      height: 512,
    },
    ...extra,
  };
}

function collection(id: string, name: string, count = 1, cover: string | null = null): Collection {
  return {
    id,
    name,
    slug: name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    description: null,
    cover_filename: cover,
    count,
    created_at: 1,
    updated_at: 1,
  };
}

const hosts = [
  { id: "studio", name: "Studio" },
  { id: "plato", name: "plato" },
];

describe("libraryOrganizationSupport", () => {
  it("hides every organization affordance until a host advertises it", () => {
    const none = libraryOrganizationSupport(hosts, { studio: null, plato: undefined });
    expect(none.organize).toBe(false);
    expect(none.trash).toBe(false);

    const legacy = libraryOrganizationSupport(hosts, {
      studio: capabilities({ can_delete: true }),
    });
    expect(legacy.organize).toBe(false);
    expect(legacy.trash).toBe(false);
  });

  it("reads organize and per-host trash retention from the capabilities", () => {
    const support = libraryOrganizationSupport(hosts, {
      studio: capabilities({
        can_delete: true,
        organize: true,
        trash: { enabled: true, retention_days: 30 },
      }),
      plato: capabilities({ can_delete: true, organize: true, trash: { enabled: false } }),
    });
    expect(support.organize).toBe(true);
    expect(support.trash).toBe(true);
    expect([...support.organizeHostIds]).toEqual(["studio", "plato"]);
    expect([...support.trashHostIds]).toEqual(["studio"]);
    expect(support.retentionDays).toEqual({ studio: 30 });
  });

  it("trashes only when every selected host can, else keeps the hard delete", () => {
    const support = { trashHostIds: new Set(["studio"]) };
    expect(selectionDeleteKind(["studio"], support)).toBe("trash");
    expect(selectionDeleteKind(["studio", "plato"], support)).toBe("delete");
    expect(selectionDeleteKind([], support)).toBe("delete");
  });

  it("words the two-tap action per kind", () => {
    expect(deleteActionCopy("trash", 3, false)).toEqual({ status: "3 selected", button: "Trash" });
    expect(deleteActionCopy("trash", 3, true)).toEqual({
      status: "Move 3 to trash?",
      button: "Confirm",
    });
    expect(deleteActionCopy("delete", 1, true)).toEqual({
      status: "Delete 1 everywhere?",
      button: "Confirm",
    });
    expect(deleteActionCopy("delete-forever", 2, true)).toEqual({
      status: "Delete 2 forever?",
      button: "Confirm",
    });
    expect(deleteActionCopy("delete-forever", 2, false).button).toBe("Delete forever");
    expect(deleteActionCopy("trash", 2, true, true).button).toBe("Moving…");
  });
});

describe("organization index and filters", () => {
  const perHost = {
    studio: [collection("c1", "Smurfs", 2)],
    plato: [collection("c9", "Smurfs", 1), collection("c2", "River studies", 3)],
  };
  const resolver = collectionSlugResolver([
    { hostId: "studio", collections: perHost.studio },
    { hostId: "plato", collections: perHost.plato },
  ]);
  const copies = [
    copy("studio", "a.png", { favorite: true, tags: ["Blue"], collections: ["c1"] }),
    copy("plato", "a.png", { tags: ["blue", "outdoor"], collections: ["c9", "c2"] }),
    copy("plato", "b.png", { timestamp: 1_600_000_000, title: "Solo", size_bytes: 99 }),
  ];

  it("unions every copy of a logical print under each physical key", () => {
    const index = buildOrganizationIndex(copies, resolver);
    const studio = index.get("studio|a.png");
    expect(studio?.favorite).toBe(true);
    expect(studio?.tags).toEqual(["Blue", "outdoor"]);
    expect(studio?.collections).toEqual(["river-studies", "smurfs"]);
    expect(index.get("plato|a.png")).toBe(studio);
    expect(index.get("plato|b.png")?.title).toBe("Solo");
    expect(logicalCopiesOf(copies, { hostId: "plato", filename: "a.png" })).toHaveLength(2);
    expect(logicalCopiesOf(copies, { hostId: "plato", filename: "b.png" })).toHaveLength(1);
  });

  it("filters representatives by favorite, tag, host, and collection", () => {
    const index = buildOrganizationIndex(copies, resolver);
    const organizationOf = (print: { hostId: string; filename: string }) =>
      index.get(`${print.hostId}|${print.filename}`);
    const representatives = [copies[0]!, copies[2]!];
    const copiesOf = (print: { hostId: string; filename: string }) =>
      logicalCopiesOf(copies, print);
    const base = { favoritesOnly: false, tag: null, hostId: null, collectionSlug: null };

    expect(filterLibraryPrints(representatives, base, organizationOf)).toHaveLength(2);
    expect(
      filterLibraryPrints(representatives, { ...base, favoritesOnly: true }, organizationOf),
    ).toEqual([copies[0]]);
    expect(
      filterLibraryPrints(representatives, { ...base, tag: "outdoor" }, organizationOf),
    ).toEqual([copies[0]]);
    expect(
      filterLibraryPrints(representatives, { ...base, collectionSlug: "smurfs" }, organizationOf),
    ).toEqual([copies[0]]);
    // The host chip matches any copy of the logical print, not only the
    // representative's own host.
    expect(
      filterLibraryPrints(representatives, { ...base, hostId: "plato" }, organizationOf, copiesOf),
    ).toEqual(representatives);
    expect(
      filterLibraryPrints(representatives, { ...base, hostId: "studio" }, organizationOf, copiesOf),
    ).toEqual([copies[0]]);
  });

  it("merges tags case-insensitively and keeps the active chip visible", () => {
    const merged = mergeHostTags({
      studio: [
        { name: "Blue", count: 2 },
        { name: "smurf", count: 9 },
      ],
      plato: [
        { name: "blue", count: 5 },
        { name: "outdoor", count: 1 },
      ],
      offline: undefined,
    });
    expect(merged).toEqual([
      { name: "smurf", count: 9 },
      { name: "Blue", count: 7 },
      { name: "outdoor", count: 1 },
    ]);
    const many = Array.from({ length: 12 }, (_, index) => ({
      name: `t${index}`,
      count: 12 - index,
    }));
    const plan = tagChipPlan(many, "t10", 8);
    expect(plan.visible).toHaveLength(8);
    expect(plan.visible.at(-1)?.name).toBe("t10");
    expect(plan.overflow.map((tag) => tag.name)).toEqual(["t7", "t8", "t9", "t11"]);
    expect(tagChipPlan(many, null, 8).overflow).toHaveLength(4);
  });

  it("scopes the tag merge to the given hosts so a forgotten machine leaves no ghost chips", () => {
    const buckets = {
      studio: [{ name: "Blue", count: 2 }],
      ghost: [{ name: "Haunt", count: 4 }],
    };
    expect(mergeHostTags(buckets, ["studio"])).toEqual([{ name: "Blue", count: 2 }]);
    // Without a scope every retained bucket still merges (legacy behaviour).
    expect(mergeHostTags(buckets)).toEqual([
      { name: "Haunt", count: 4 },
      { name: "Blue", count: 2 },
    ]);
  });

  it("merges collections across hosts into cards with host labels", () => {
    const merged = mergedCollectionsFor(perHost, hosts);
    const cards = collectionCards(merged, { studio: "Studio", plato: "plato" });
    expect(cards.map((card) => card.name)).toEqual(["River studies", "Smurfs"]);
    expect(cards[1]).toMatchObject({ slug: "smurfs", count: 3, hostsLabel: "Studio · plato" });
    expect(collectionOnHost(perHost, "plato", "smurfs")?.id).toBe("c9");
    expect(collectionOnHost(perHost, "studio", "river-studies")).toBeUndefined();
  });

  it("validates collection names", () => {
    expect(validateCollectionName("  River   studies ")).toEqual({
      ok: true,
      value: "River studies",
    });
    expect(validateCollectionName("   ").ok).toBe(false);
    expect(validateCollectionName("???").ok).toBe(false);
  });
});

describe("trash helpers", () => {
  it("lists retention for trash-capable hosts and labels purge countdowns", () => {
    const support = libraryOrganizationSupport(hosts, {
      studio: capabilities({ can_delete: true, trash: { enabled: true, retention_days: 30 } }),
      plato: capabilities({ can_delete: true, trash: { enabled: true, retention_days: 0 } }),
    });
    expect(trashRetentionHosts(hosts, support)).toEqual([
      { label: "Studio", retentionDays: 30 },
      { label: "plato", retentionDays: 0 },
    ]);
    const now = Date.UTC(2026, 7, 19);
    expect(purgeChipLabel(now / 1000 + 3 * 86_400, now)).toBe("Purges in 3 d");
    expect(purgeChipLabel(now / 1000 - 1, now)).toBe("Purges today");
    expect(purgeChipLabel(null, now)).toBeNull();
  });

  it("retains the last snapshot for trash-capable hosts a pass could not read", () => {
    const previous = [
      copy("studio", "stale.png", { timestamp: 5 }),
      copy("plato", "kept.png", { timestamp: 9 }),
      copy("forgotten", "orphan.png", { timestamp: 3 }),
    ];
    const outcome = mergeTrashSnapshot({
      previous,
      refreshed: [copy("studio", "fresh.png", { timestamp: 7 })],
      refreshedHostIds: new Set(["studio"]),
      trashCapableHostIds: new Set(["studio", "plato"]),
      rejectedHosts: 0,
      skippedHosts: 1,
    });
    // plato (skipped offline) keeps its prior prints; studio is replaced by
    // the fresh read; the no-longer-capable host's copies are dropped.
    expect(outcome.copies.map((entry) => entry.filename)).toEqual(["kept.png", "fresh.png"]);
    expect(outcome.failedHosts).toBe(1);
    // An incomplete snapshot is never authoritative: the scope stays
    // retry-eligible so re-entering Trash refetches.
    expect(outcome.complete).toBe(false);
  });

  it("marks the trash snapshot complete only when every capable host was read", () => {
    const outcome = mergeTrashSnapshot({
      previous: [copy("studio", "old.png", { timestamp: 1 })],
      refreshed: [copy("studio", "new.png", { timestamp: 2 })],
      refreshedHostIds: new Set(["studio"]),
      trashCapableHostIds: new Set(["studio"]),
      rejectedHosts: 0,
      skippedHosts: 0,
    });
    expect(outcome.copies.map((entry) => entry.filename)).toEqual(["new.png"]);
    expect(outcome.complete).toBe(true);
    expect(outcome.failedHosts).toBe(0);

    const rejected = mergeTrashSnapshot({
      previous: [],
      refreshed: [],
      refreshedHostIds: new Set<string>(),
      trashCapableHostIds: new Set(["studio"]),
      rejectedHosts: 1,
      skippedHosts: 0,
    });
    expect(rejected.complete).toBe(false);
    expect(rejected.failedHosts).toBe(1);
  });
});

describe("titles", () => {
  it("validates the Create title and restores one from metadata", () => {
    expect(requestTitle("  Smurf 04 ")).toEqual({ ok: true, title: "Smurf 04" });
    expect(requestTitle("   ")).toEqual({ ok: true, title: null });
    expect(requestTitle("a b").ok).toBe(false);
    expect(reusedPrintTitle({ title: " Grain test " } as never)).toBe("Grain test");
    expect(reusedPrintTitle({} as never)).toBe("");
    expect(reusedPrintTitle(null)).toBe("");
  });
});

describe("runOrganizationFanout", () => {
  function fakeApi(): FanoutApi {
    return {
      patchGalleryImage: vi.fn().mockResolvedValue(null),
      organizeGallery: vi.fn().mockResolvedValue(undefined),
      createCollection: vi
        .fn()
        .mockImplementation((_target, body: { name: string }) =>
          Promise.resolve(collection("new", body.name)),
        ),
      setCollectionItems: vi.fn().mockResolvedValue(null),
      trashMany: vi.fn().mockResolvedValue(undefined),
      restoreTrashed: vi.fn().mockResolvedValue(undefined),
      deleteGalleryImageForever: vi.fn().mockResolvedValue(undefined),
      deleteGalleryImage: vi.fn().mockResolvedValue(undefined),
    };
  }
  const fanoutHosts = {
    studio: {
      id: "studio",
      name: "Studio",
      target: { baseUrl: "http://studio", apiKey: "s" },
      collections: [collection("c1", "Smurfs")],
    },
    plato: {
      id: "plato",
      name: "plato",
      target: { baseUrl: "http://plato", apiKey: "p" },
      collections: [],
    },
  };
  const targets = [
    { hostId: "studio", filename: "a.png" },
    { hostId: "plato", filename: "a.png" },
    { hostId: "plato", filename: "b.png" },
  ];

  it("creates the collection on hosts lacking it, then adds every copy", async () => {
    const api = fakeApi();
    const ops = planOrganizationFanout(targets, { kind: "addToCollection", name: "Smurfs" });
    const result = await runOrganizationFanout(ops, fanoutHosts, api);
    expect(result.failures).toEqual([]);
    expect(api.createCollection).toHaveBeenCalledTimes(1);
    expect(api.createCollection).toHaveBeenCalledWith(fanoutHosts.plato.target, {
      name: "Smurfs",
    });
    expect(api.setCollectionItems).toHaveBeenCalledWith(fanoutHosts.studio.target, "c1", {
      add: ["a.png"],
      remove: [],
    });
    expect(api.setCollectionItems).toHaveBeenCalledWith(fanoutHosts.plato.target, "new", {
      add: ["a.png", "b.png"],
      remove: [],
    });
    expect(result.createdCollections.map((entry) => entry.hostId)).toEqual(["plato"]);
  });

  it("routes titles through PATCH and favorites/tags through organize", async () => {
    const api = fakeApi();
    await runOrganizationFanout(
      planOrganizationFanout(targets, { kind: "setTitle", title: "Grain" }),
      fanoutHosts,
      api,
    );
    expect(api.patchGalleryImage).toHaveBeenCalledTimes(3);
    expect(api.patchGalleryImage).toHaveBeenCalledWith(fanoutHosts.plato.target, "b.png", {
      title: "Grain",
    });
    await runOrganizationFanout(
      planOrganizationFanout(targets, { kind: "setTitle", title: null }),
      fanoutHosts,
      api,
    );
    expect(api.patchGalleryImage).toHaveBeenLastCalledWith(expect.anything(), expect.any(String), {
      title: "",
    });
    await runOrganizationFanout(
      planOrganizationFanout(targets, { kind: "setFavorite", favorite: true }),
      fanoutHosts,
      api,
    );
    expect(api.organizeGallery).toHaveBeenCalledWith(fanoutHosts.plato.target, {
      filenames: ["a.png", "b.png"],
      favorite: true,
    });
    await runOrganizationFanout(
      planOrganizationFanout(targets, { kind: "addTags", tags: ["blue"] }),
      fanoutHosts,
      api,
    );
    expect(api.organizeGallery).toHaveBeenLastCalledWith(expect.anything(), {
      filenames: expect.any(Array),
      add_tags: ["blue"],
    });
  });

  it("trashes on trash-capable hosts and hard-deletes on the rest", async () => {
    const api = fakeApi();
    const result = await runOrganizationFanout(
      planOrganizationFanout(targets, { kind: "trash" }),
      fanoutHosts,
      api,
      { trashHostIds: new Set(["studio"]) },
    );
    expect(result.failures).toEqual([]);
    expect(api.trashMany).toHaveBeenCalledWith(fanoutHosts.studio.target, ["a.png"]);
    expect(api.deleteGalleryImage).toHaveBeenCalledTimes(2);
    expect(api.trashMany).toHaveBeenCalledTimes(1);
  });

  it("reports a failed host without blocking the others", async () => {
    const api = fakeApi();
    api.restoreTrashed = vi
      .fn()
      .mockImplementation((target: { baseUrl: string }) =>
        target.baseUrl === "http://plato"
          ? Promise.reject(new Error("409 Conflict"))
          : Promise.resolve(),
      );
    const result = await runOrganizationFanout(
      planOrganizationFanout(targets, { kind: "restore" }),
      fanoutHosts,
      api,
    );
    expect(result.succeededHostIds).toEqual(["studio"]);
    expect(result.failures).toHaveLength(1);
    expect(result.failures[0]?.hostName).toBe("plato");
    expect(
      fanoutFailureMessage("restore 2 prints", result.failures, (error) =>
        error instanceof Error ? error.message : String(error),
      ),
    ).toBe("Couldn’t restore 2 prints on plato. 409 Conflict");
    const missing = await runOrganizationFanout(
      planOrganizationFanout([{ hostId: "gone", filename: "x.png" }], { kind: "deleteForever" }),
      fanoutHosts,
      api,
    );
    expect(missing.failures[0]?.hostId).toBe("gone");
  });
});
