import { describe, expect, it } from "vitest";
import type { Collection } from "./api/galleryOrganization";
import {
  RETENTION_OPTIONS,
  collectionSlug,
  collectionSlugResolver,
  displayTitle,
  mergeCollectionsAcrossHosts,
  normalizeTagName,
  planOrganizationFanout,
  purgeCountdown,
  purgeCountdownFromPurgeAt,
  retentionLabel,
  sortCollections,
  sortTags,
  tagKey,
  titleSlug,
  trashRetentionSummary,
  unionOrganization,
  validatePrintTitle,
} from "./libraryOrganization";

/**
 * Parity fixtures for `mold_core::title_slug` — keep this table identical to
 * the Rust unit test so both sides can be cross-checked by eye.
 *
 * Algorithm (both sides): ASCII-lowercase; every char outside `[a-z0-9]`
 * becomes `-`; collapse runs of `-`; trim leading/trailing `-`; cut to the
 * max length; trim any trailing `-` the cut exposed; empty ⇒ None/null.
 */
const TITLE_SLUG_FIXTURES: ReadonlyArray<[string, string | null]> = [
  ["Sunset over the bay", "sunset-over-the-bay"],
  ["  Hello,   World!  ", "hello-world"],
  ["Café au lait", "caf-au-lait"],
  ["UPPER_case-mixed.v2", "upper-case-mixed-v2"],
  ["a", "a"],
  ["42", "42"],
  ["---", null],
  ["", null],
  ["   ", null],
  ["!!!", null],
  ["日本語", null],
  ["İstanbul", "stanbul"],
  ["a".repeat(45), "a".repeat(40)],
  [
    "abcdefghij abcdefghij abcdefghij abcdefghij",
    "abcdefghij-abcdefghij-abcdefghij-abcdefg",
  ],
  // The 40-char cut lands on the separator; the dangling `-` is trimmed.
  [
    "abcdefghijklmnopqrstuvwxyz0123456789abc def",
    "abcdefghijklmnopqrstuvwxyz0123456789abc",
  ],
];

describe("titleSlug", () => {
  it.each(TITLE_SLUG_FIXTURES)("%j → %j", (input, expected) => {
    expect(titleSlug(input)).toBe(expected);
  });

  it("never exceeds 40 chars and never starts or ends with a dash", () => {
    for (const input of [
      "x".repeat(200),
      "-".repeat(5) + "y".repeat(50) + "-",
    ]) {
      const slug = titleSlug(input);
      expect(slug).not.toBeNull();
      expect(slug!.length).toBeLessThanOrEqual(40);
      expect(slug).toMatch(/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/);
    }
  });
});

describe("collectionSlug", () => {
  it("shares the title algorithm but caps at 80 and returns an empty string", () => {
    expect(collectionSlug("Sunset over the bay")).toBe("sunset-over-the-bay");
    expect(collectionSlug("  Hello,   World!  ")).toBe("hello-world");
    expect(collectionSlug("")).toBe("");
    expect(collectionSlug("!!!")).toBe("");
    expect(collectionSlug("a".repeat(100))).toBe("a".repeat(80));
    expect(collectionSlug("abcdefghij ".repeat(8))).toBe(
      "abcdefghij-".repeat(7) + "abc",
    );
  });

  it("merges names that differ only in case, punctuation, or spacing", () => {
    expect(collectionSlug("Studio Shots")).toBe(
      collectionSlug("studio   shots!"),
    );
    expect(collectionSlug("Studio Shots")).toBe(collectionSlug("STUDIO-SHOTS"));
  });
});

describe("normalizeTagName / tagKey", () => {
  it("trims, collapses whitespace, and drops a leading hash", () => {
    expect(normalizeTagName("  #Portrait   Study  ")).toBe("Portrait Study");
    expect(normalizeTagName("\t#\n")).toBe("");
    expect(normalizeTagName("#")).toBe("");
    expect(normalizeTagName("plain")).toBe("plain");
  });

  it("strips control characters", () => {
    expect(normalizeTagName("a\u0000b\u001fc")).toBe("abc");
  });

  it("tagKey is the case-insensitive merge key", () => {
    expect(tagKey("Portrait")).toBe("portrait");
    expect(tagKey("  #PORTRAIT ")).toBe("portrait");
  });
});

describe("validatePrintTitle", () => {
  it("trims and treats empty as 'no title'", () => {
    expect(validatePrintTitle("  Golden hour  ")).toEqual({
      ok: true,
      value: "Golden hour",
    });
    expect(validatePrintTitle("")).toEqual({ ok: true, value: null });
    expect(validatePrintTitle("   \t ")).toEqual({ ok: true, value: null });
  });

  it("accepts exactly 120 characters and rejects 121", () => {
    expect(validatePrintTitle("x".repeat(120))).toEqual({
      ok: true,
      value: "x".repeat(120),
    });
    const long = validatePrintTitle("x".repeat(121));
    expect(long.ok).toBe(false);
    if (!long.ok) expect(long.reason).toMatch(/120/);
  });

  it("counts the trimmed length, not the raw length", () => {
    expect(validatePrintTitle(" ".repeat(10) + "x".repeat(120)).ok).toBe(true);
  });

  it("rejects control characters but keeps unicode", () => {
    const bad = validatePrintTitle("tab\there");
    expect(bad.ok).toBe(false);
    if (!bad.ok) expect(bad.reason).toMatch(/control/i);
    expect(validatePrintTitle("nul\u0000").ok).toBe(false);
    expect(validatePrintTitle("Café — 日本語 ✨")).toEqual({
      ok: true,
      value: "Café — 日本語 ✨",
    });
  });
});

describe("displayTitle", () => {
  const filename = "mold-flux-dev-1712345678~sunset.png";

  it("prefers the title, then a prompt excerpt, then the filename stem", () => {
    expect(
      displayTitle({
        title: "Sunset",
        metadata: { prompt: "a long prompt" },
        filename,
      }),
    ).toBe("Sunset");
    expect(
      displayTitle({ title: "  ", metadata: { prompt: "a cat" }, filename }),
    ).toBe("a cat");
    expect(displayTitle({ title: null, metadata: {}, filename })).toBe(
      "mold-flux-dev-1712345678~sunset",
    );
    expect(displayTitle({ filename })).toBe("mold-flux-dev-1712345678~sunset");
  });

  it("collapses prompt whitespace and truncates on a word boundary with an ellipsis", () => {
    const prompt =
      "A majestic   mountain\nrange at dawn, painted in oils with dramatic lighting and mist";
    const shown = displayTitle({ metadata: { prompt }, filename });
    expect(shown.length).toBeLessThanOrEqual(48);
    expect(shown.endsWith("…")).toBe(true);
    expect(shown).not.toContain("\n");
    expect(shown).not.toMatch(/\s…$/);
    expect(shown).toBe("A majestic mountain range at dawn, painted in…");
  });

  it("truncates titles too and honours a custom limit", () => {
    expect(displayTitle({ title: "abcdefghijkl", filename }, 8)).toBe(
      "abcdefg…",
    );
    expect(displayTitle({ title: "short", filename }, 8)).toBe("short");
  });

  it("keeps a stem that has no extension", () => {
    expect(displayTitle({ filename: "no-extension" })).toBe("no-extension");
  });
});

function collection(
  overrides: Partial<Collection> & { id: string },
): Collection {
  return {
    name: overrides.id,
    slug: collectionSlug(overrides.name ?? overrides.id),
    description: null,
    cover_filename: null,
    count: 0,
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

describe("mergeCollectionsAcrossHosts", () => {
  it("merges by slug, keeps the first-seen name, sums counts, and takes the first cover", () => {
    const merged = mergeCollectionsAcrossHosts([
      {
        hostId: "local",
        hostLabel: "This Mac",
        collections: [
          collection({ id: "c1", name: "Studio Shots", count: 3 }),
          collection({
            id: "c2",
            name: "Zebras",
            count: 1,
            cover_filename: "z.png",
          }),
        ],
      },
      {
        hostId: "plato",
        hostLabel: "plato",
        collections: [
          collection({
            id: "p9",
            name: "studio shots",
            count: 2,
            cover_filename: "s.png",
          }),
          collection({ id: "p1", name: "Alpha", count: 5 }),
        ],
      },
    ]);

    expect(merged.map((entry) => entry.slug)).toEqual([
      "alpha",
      "studio-shots",
      "zebras",
    ]);
    const studio = merged.find((entry) => entry.slug === "studio-shots")!;
    expect(studio.name).toBe("Studio Shots");
    expect(studio.count).toBe(5);
    expect(studio.hosts).toEqual([
      { hostId: "local", id: "c1", count: 3 },
      { hostId: "plato", id: "p9", count: 2 },
    ]);
    expect(studio.cover).toEqual({ hostId: "plato", filename: "s.png" });
    expect(merged.find((entry) => entry.slug === "zebras")!.cover).toEqual({
      hostId: "local",
      filename: "z.png",
    });
    expect(merged.find((entry) => entry.slug === "alpha")!.cover).toBeNull();
  });

  it("derives a slug from the name when a host omits it and skips unsluggable rows", () => {
    const merged = mergeCollectionsAcrossHosts([
      {
        hostId: "h",
        hostLabel: "h",
        collections: [
          { ...collection({ id: "x", name: "Night Walks" }), slug: "" },
          { ...collection({ id: "y", name: "!!!" }), slug: "" },
        ],
      },
    ]);
    expect(merged).toHaveLength(1);
    expect(merged[0]!.slug).toBe("night-walks");
  });

  it("returns an empty list for no hosts", () => {
    expect(mergeCollectionsAcrossHosts([])).toEqual([]);
  });
});

describe("unionOrganization", () => {
  const resolve = collectionSlugResolver([
    {
      hostId: "local",
      collections: [collection({ id: "c1", name: "Studio Shots" })],
    },
    {
      hostId: "plato",
      collections: [
        collection({ id: "p9", name: "studio shots" }),
        collection({ id: "p2", name: "Drafts" }),
      ],
    },
  ]);

  it("prefers the local title, ORs favorite, unions tags, and maps collections to slugs", () => {
    const union = unionOrganization(
      [
        {
          hostId: "plato",
          item: {
            title: "Remote title",
            favorite: false,
            tags: ["Portrait", "b&w"],
            collections: ["p9", "p2"],
          },
        },
        {
          hostId: "local",
          item: {
            title: "Local title",
            favorite: true,
            tags: ["portrait", "Alpha"],
            collections: ["c1"],
          },
        },
      ],
      { localHostId: "local", resolveCollectionSlug: resolve },
    );
    expect(union.title).toBe("Local title");
    expect(union.favorite).toBe(true);
    expect(union.tags).toEqual(["Alpha", "b&w", "Portrait"]);
    expect(union.collections).toEqual(["drafts", "studio-shots"]);
    expect(union.trashedAt).toBeNull();
    expect(union.purgeAt).toBeNull();
    expect(union.unresolvedCollectionIds).toEqual([]);
  });

  it("falls back to the first non-empty title when the local copy is untitled", () => {
    const union = unionOrganization(
      [
        { hostId: "local", item: { title: "  " } },
        { hostId: "a", item: { title: null } },
        { hostId: "b", item: { title: "From b" } },
        { hostId: "c", item: { title: "From c" } },
      ],
      { localHostId: "local", resolveCollectionSlug: resolve },
    );
    expect(union.title).toBe("From b");
  });

  it("keeps the first-seen tag casing for case-insensitive duplicates", () => {
    const union = unionOrganization(
      [
        { hostId: "a", item: { tags: ["Portrait"] } },
        { hostId: "b", item: { tags: ["PORTRAIT", "portrait"] } },
      ],
      { resolveCollectionSlug: resolve },
    );
    expect(union.tags).toEqual(["Portrait"]);
  });

  it("takes the earliest trash and purge times and reports unresolved collection ids", () => {
    const union = unionOrganization(
      [
        {
          hostId: "a",
          item: { trashed_at: 200, purge_at: 900, collections: ["ghost"] },
        },
        { hostId: "b", item: { trashed_at: 100, purge_at: null } },
        { hostId: "c", item: {} },
      ],
      { resolveCollectionSlug: resolve },
    );
    expect(union.trashedAt).toBe(100);
    expect(union.purgeAt).toBe(900);
    expect(union.unresolvedCollectionIds).toEqual([
      { hostId: "a", id: "ghost" },
    ]);
    expect(union.collections).toEqual([]);
  });

  it("is empty for no copies", () => {
    expect(unionOrganization([], { resolveCollectionSlug: resolve })).toEqual({
      title: null,
      favorite: false,
      tags: [],
      collections: [],
      trashedAt: null,
      purgeAt: null,
      unresolvedCollectionIds: [],
    });
  });
});

describe("planOrganizationFanout", () => {
  const copies = [
    { hostId: "local", filename: "a.png" },
    { hostId: "plato", filename: "a.png" },
    { hostId: "local", filename: "b.png" },
  ];

  it("groups copies per host in first-seen order", () => {
    expect(
      planOrganizationFanout(copies, { kind: "setFavorite", favorite: true }),
    ).toEqual([
      {
        hostId: "local",
        filenames: ["a.png", "b.png"],
        kind: "setFavorite",
        favorite: true,
      },
      {
        hostId: "plato",
        filenames: ["a.png"],
        kind: "setFavorite",
        favorite: true,
      },
    ]);
  });

  it("dedupes a filename repeated on one host", () => {
    const ops = planOrganizationFanout(
      [
        { hostId: "h", filename: "a.png" },
        { hostId: "h", filename: "a.png" },
      ],
      { kind: "trash" },
    );
    expect(ops).toEqual([{ hostId: "h", filenames: ["a.png"], kind: "trash" }]);
  });

  it("carries the title, tag lists, and collection identity through", () => {
    expect(
      planOrganizationFanout(copies.slice(0, 1), {
        kind: "setTitle",
        title: "New",
      }),
    ).toEqual([
      { hostId: "local", filenames: ["a.png"], kind: "setTitle", title: "New" },
    ]);
    expect(
      planOrganizationFanout(copies.slice(0, 1), {
        kind: "setTitle",
        title: null,
      }),
    ).toEqual([
      { hostId: "local", filenames: ["a.png"], kind: "setTitle", title: null },
    ]);
    expect(
      planOrganizationFanout(copies.slice(0, 1), {
        kind: "addTags",
        tags: ["x", "y"],
      }),
    ).toEqual([
      {
        hostId: "local",
        filenames: ["a.png"],
        kind: "addTags",
        tags: ["x", "y"],
      },
    ]);
    expect(
      planOrganizationFanout(copies.slice(0, 1), {
        kind: "removeTags",
        tags: ["x"],
      }),
    ).toEqual([
      {
        hostId: "local",
        filenames: ["a.png"],
        kind: "removeTags",
        tags: ["x"],
      },
    ]);
    expect(
      planOrganizationFanout(copies.slice(0, 1), {
        kind: "removeFromCollection",
        slug: "s",
      }),
    ).toEqual([
      {
        hostId: "local",
        filenames: ["a.png"],
        kind: "removeFromCollection",
        slug: "s",
      },
    ]);
    for (const kind of ["restore", "deleteForever"] as const) {
      expect(planOrganizationFanout(copies.slice(0, 1), { kind })).toEqual([
        { hostId: "local", filenames: ["a.png"], kind },
      ]);
    }
  });

  it("add-to-collection yields an ensureCollection per host so callers create by name", () => {
    const ops = planOrganizationFanout(copies, {
      kind: "addToCollection",
      slug: "studio-shots",
      name: "Studio Shots",
    });
    expect(ops).toEqual([
      {
        hostId: "local",
        filenames: ["a.png", "b.png"],
        kind: "addToCollection",
        ensureCollection: { name: "Studio Shots", slug: "studio-shots" },
      },
      {
        hostId: "plato",
        filenames: ["a.png"],
        kind: "addToCollection",
        ensureCollection: { name: "Studio Shots", slug: "studio-shots" },
      },
    ]);
  });

  it("derives the slug from the name when add-to-collection omits it", () => {
    const [op] = planOrganizationFanout(copies.slice(0, 1), {
      kind: "addToCollection",
      name: "Night Walks",
    });
    expect(op).toMatchObject({
      ensureCollection: { name: "Night Walks", slug: "night-walks" },
    });
  });

  it("is empty for no copies", () => {
    expect(planOrganizationFanout([], { kind: "trash" })).toEqual([]);
  });
});

describe("retention", () => {
  it("labels the retention options", () => {
    expect(RETENTION_OPTIONS).toEqual([1, 7, 30, 90, 365, 0]);
    expect(RETENTION_OPTIONS.map(retentionLabel)).toEqual([
      "1 day",
      "7 days",
      "30 days",
      "90 days",
      "1 year",
      "Forever",
    ]);
    expect(retentionLabel(2)).toBe("2 days");
    expect(retentionLabel(730)).toBe("2 years");
    expect(retentionLabel(400)).toBe("400 days");
  });

  it("treats negative or non-finite retention as forever", () => {
    expect(retentionLabel(-1)).toBe("Forever");
    expect(retentionLabel(Number.NaN)).toBe("Forever");
  });
});

describe("purgeCountdown", () => {
  const trashedAt = 1_700_000_000; // unix secs
  const day = 86_400;

  it("counts whole days remaining, rounding up", () => {
    const now = (trashedAt + 27 * day + 1) * 1000; // 2.99… days left
    expect(purgeCountdown(trashedAt, 30, now)).toEqual({
      kind: "purges",
      days: 3,
      label: "Purges in 3 d",
    });
    expect(
      purgeCountdown(trashedAt, 30, (trashedAt + 29 * day) * 1000),
    ).toEqual({
      kind: "purges",
      days: 1,
      label: "Purges in 1 d",
    });
  });

  it("says today once the purge moment is within the day or already past", () => {
    expect(
      purgeCountdown(trashedAt, 30, (trashedAt + 30 * day) * 1000),
    ).toEqual({
      kind: "today",
      label: "Purges today",
    });
    expect(
      purgeCountdown(trashedAt, 30, (trashedAt + 45 * day) * 1000),
    ).toEqual({
      kind: "today",
      label: "Purges today",
    });
  });

  it("keeps forever for retention 0 or a missing trash stamp", () => {
    expect(purgeCountdown(trashedAt, 0, Date.now())).toEqual({
      kind: "kept",
      label: "Kept until you empty the trash",
    });
    expect(purgeCountdown(null, 30, Date.now())).toEqual({
      kind: "kept",
      label: "Kept until you empty the trash",
    });
  });

  it("accepts the server's purge_at directly", () => {
    const purgeAt = trashedAt + 30 * day;
    expect(
      purgeCountdownFromPurgeAt(purgeAt, (trashedAt + 20 * day) * 1000),
    ).toEqual({
      kind: "purges",
      days: 10,
      label: "Purges in 10 d",
    });
    expect(purgeCountdownFromPurgeAt(null, Date.now())).toEqual({
      kind: "kept",
      label: "Kept until you empty the trash",
    });
    expect(purgeCountdownFromPurgeAt(undefined, Date.now()).kind).toBe("kept");
  });
});

describe("trashRetentionSummary", () => {
  it("states one host's retention", () => {
    const summary = trashRetentionSummary([
      { label: "This Mac", retentionDays: 30 },
    ]);
    expect(summary.text).toBe("Prints stay in the trash 30 d before purge");
    expect(summary.segments).toEqual([
      { text: "Prints stay in the trash ", mono: false },
      { text: "30 d", mono: true },
      { text: " before purge", mono: false },
    ]);
  });

  it("collapses hosts that agree and names the ones that differ", () => {
    const summary = trashRetentionSummary([
      { label: "This Mac", retentionDays: 30 },
      { label: "bender", retentionDays: 30 },
      { label: "plato", retentionDays: 7 },
      { label: "hal9000", retentionDays: 0 },
    ]);
    expect(summary.text).toBe(
      "Prints stay in the trash 30 d before purge · plato keeps 7 d · hal9000 keeps trash forever",
    );
    expect(
      summary.segments.filter((segment) => segment.mono).map((s) => s.text),
    ).toEqual(["30 d", "7 d"]);
  });

  it("phrases forever on the primary host", () => {
    expect(
      trashRetentionSummary([
        { label: "This Mac", retentionDays: 0 },
        { label: "plato", retentionDays: 7 },
      ]).text,
    ).toBe("Prints stay in the trash until you empty it · plato keeps 7 d");
  });

  it("is empty for no hosts", () => {
    expect(trashRetentionSummary([])).toEqual({ text: "", segments: [] });
  });
});

describe("sorting", () => {
  it("sortCollections orders by name, case-insensitively, without mutating", () => {
    const input = [
      collection({ id: "b", name: "beta" }),
      collection({ id: "a", name: "Alpha" }),
      collection({ id: "c", name: "alpha" }),
    ];
    const sorted = sortCollections(input);
    expect(sorted.map((entry) => entry.id)).toEqual(["a", "c", "b"]);
    expect(input.map((entry) => entry.id)).toEqual(["b", "a", "c"]);
  });

  it("sortTags orders by count desc then name", () => {
    const sorted = sortTags([
      { name: "zeta", count: 2 },
      { name: "alpha", count: 2 },
      { name: "Beta", count: 9 },
      { name: "gamma", count: 0 },
    ]);
    expect(sorted.map((tag) => tag.name)).toEqual([
      "Beta",
      "alpha",
      "zeta",
      "gamma",
    ]);
  });
});
