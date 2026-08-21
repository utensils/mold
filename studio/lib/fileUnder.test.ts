import { describe, expect, it } from "vitest";
import type { Collection } from "./api/galleryOrganization";
import {
  AUTO_TAG_SETTING_WEB,
  MAX_REQUEST_TAGS,
  REQUEST_TAG_MAX_LEN,
  addTag,
  buildFileUnderRequestFields,
  clearCollection,
  deriveGhostTag,
  downloadFileName,
  effectiveCollection,
  effectiveTags,
  emptyFileUnderState,
  fileUnderAvailable,
  matchCollection,
  pickCollection,
  removeTag,
  restoreGhostTag,
  suggestTags,
  validateNewTag,
  validateRequestTag,
  type FileUnderCollectionLike,
  type FileUnderState,
} from "./fileUnder";

/** A C0 control character; never legal in a tag (mirrors the Rust guard). */
const BELL = String.fromCharCode(7);

// ── Fixtures ────────────────────────────────────────────────────────────────

function collection(
  name: string,
  overrides: Partial<Collection> = {},
): Collection {
  return {
    id: `id-${name}`,
    name,
    slug: name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    description: null,
    cover_filename: null,
    count: 0,
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

const COLLECTIONS: Collection[] = [
  collection("Sunsets"),
  collection("Portraits"),
];

function state(overrides: Partial<FileUnderState> = {}): FileUnderState {
  return { ...emptyFileUnderState(), ...overrides };
}

// ── Ghost tag ───────────────────────────────────────────────────────────────

describe("deriveGhostTag", () => {
  it("is the title's slug", () => {
    expect(deriveGhostTag("Sunset over the bay", true)).toBe(
      "sunset-over-the-bay",
    );
  });

  it("is null when auto-tagging is off", () => {
    expect(deriveGhostTag("Sunset over the bay", false)).toBeNull();
  });

  it.each([[null], [undefined], [""], ["   "], ["日本語"], ["!!!"]])(
    "is null for an untitled or unsluggable title (%j)",
    (title) => {
      expect(deriveGhostTag(title, true)).toBeNull();
    },
  );
});

// ── Collection matching ─────────────────────────────────────────────────────

describe("matchCollection", () => {
  it("matches the collection whose slug equals the title slug", () => {
    expect(matchCollection("Sunsets", COLLECTIONS)?.name).toBe("Sunsets");
  });

  it("matches through the slug, so case and punctuation do not matter", () => {
    expect(matchCollection("SUNSETS!", COLLECTIONS)?.name).toBe("Sunsets");
    expect(matchCollection("  sun sets  ", COLLECTIONS)).toBeNull();
  });

  it("falls back to the name's slug when the row carries none", () => {
    const rows: FileUnderCollectionLike[] = [{ id: "a", name: "Night Walks" }];
    expect(matchCollection("Night walks", rows)?.id).toBe("a");
  });

  it("returns null when nothing matches", () => {
    expect(matchCollection("Barn owls", COLLECTIONS)).toBeNull();
  });

  it.each([[""], ["   "], ["日本語"]])(
    "returns null for an unsluggable title (%j)",
    (title) => {
      expect(matchCollection(title, COLLECTIONS)).toBeNull();
    },
  );
});

// ── Tags ────────────────────────────────────────────────────────────────────

describe("effectiveTags", () => {
  it("puts the ghost tag first, then the manual tags in order", () => {
    const s = state({ manualTags: ["golden hour", "kodak"] });
    expect(effectiveTags(s, "Sunset over the bay", true)).toEqual([
      "sunset-over-the-bay",
      "golden hour",
      "kodak",
    ]);
  });

  it("drops the ghost tag once it is removed", () => {
    const s = state({ ghostRemoved: true, manualTags: ["kodak"] });
    expect(effectiveTags(s, "Sunset over the bay", true)).toEqual(["kodak"]);
  });

  it("omits the ghost tag when auto-tagging is off", () => {
    const s = state({ manualTags: ["kodak"] });
    expect(effectiveTags(s, "Sunset over the bay", false)).toEqual(["kodak"]);
  });

  it("dedupes case-insensitively, keeping the first casing", () => {
    const s = state({ manualTags: ["Kodak", "kodak", "KODAK"] });
    expect(effectiveTags(s, "", true)).toEqual(["Kodak"]);
  });

  it("dedupes a manual tag that repeats the ghost tag", () => {
    const s = state({ manualTags: ["Sunsets", "kodak"] });
    expect(effectiveTags(s, "Sunsets", true)).toEqual(["sunsets", "kodak"]);
  });

  it("normalizes every tag through normalizeTagName", () => {
    const s = state({ manualTags: ["  #golden   hour ", "kodak"] });
    expect(effectiveTags(s, "", true)).toEqual(["golden hour", "kodak"]);
  });

  it("drops tags that normalize to nothing", () => {
    const s = state({ manualTags: ["   ", "#", "kodak"] });
    expect(effectiveTags(s, "", true)).toEqual(["kodak"]);
  });
});

describe("addTag", () => {
  it("normalizes and appends", () => {
    expect(addTag(emptyFileUnderState(), "  #Golden  Hour ").manualTags).toEqual(
      ["Golden Hour"],
    );
  });

  it("ignores a blank tag", () => {
    expect(addTag(emptyFileUnderState(), "   ").manualTags).toEqual([]);
  });

  it("ignores a case-insensitive duplicate", () => {
    const s = addTag(state({ manualTags: ["Kodak"] }), "kodak");
    expect(s.manualTags).toEqual(["Kodak"]);
  });

  it("does not mutate the input state", () => {
    const before = emptyFileUnderState();
    addTag(before, "kodak");
    expect(before.manualTags).toEqual([]);
  });
});

describe("removeTag", () => {
  it("removes a manual tag case-insensitively", () => {
    const s = removeTag(
      state({ manualTags: ["Kodak", "grain"] }),
      "KODAK",
      "",
      true,
    );
    expect(s.manualTags).toEqual(["grain"]);
  });

  it("marks the ghost removed and keeps the manual tags", () => {
    const s = removeTag(
      state({ manualTags: ["kodak"] }),
      "sunsets",
      "Sunsets",
      true,
    );
    expect(s.ghostRemoved).toBe(true);
    expect(s.manualTags).toEqual(["kodak"]);
  });

  it("removing the ghost also drops an identical manual tag", () => {
    const s = removeTag(
      state({ manualTags: ["Sunsets", "kodak"] }),
      "sunsets",
      "Sunsets",
      true,
    );
    expect(effectiveTags(s, "Sunsets", true)).toEqual(["kodak"]);
  });

  it("does not touch the ghost flag when auto-tagging is off", () => {
    const s = removeTag(
      state({ manualTags: ["sunsets"] }),
      "sunsets",
      "Sunsets",
      false,
    );
    expect(s.ghostRemoved).toBe(false);
    expect(s.manualTags).toEqual([]);
  });
});

describe("restoreGhostTag", () => {
  it("re-offers a removed ghost tag", () => {
    const s = restoreGhostTag(state({ ghostRemoved: true }));
    expect(effectiveTags(s, "Sunsets", true)).toEqual(["sunsets"]);
  });
});

// ── Collection row ──────────────────────────────────────────────────────────

describe("effectiveCollection", () => {
  it("pre-selects the collection matching the title slug", () => {
    const chosen = effectiveCollection(
      emptyFileUnderState(),
      "Sunsets",
      COLLECTIONS,
    );
    expect(chosen).toEqual({
      id: "id-Sunsets",
      name: "Sunsets",
      slug: "sunsets",
      source: "title",
    });
  });

  it("returns null when no collection matches the title", () => {
    expect(
      effectiveCollection(emptyFileUnderState(), "Barn owls", COLLECTIONS),
    ).toBeNull();
  });

  it("lets an explicit pick win over the title match", () => {
    const s = pickCollection(emptyFileUnderState(), {
      id: "id-Portraits",
      name: "Portraits",
    });
    expect(effectiveCollection(s, "Sunsets", COLLECTIONS)).toEqual({
      id: "id-Portraits",
      name: "Portraits",
      slug: "portraits",
      source: "picked",
    });
  });

  it("keeps an explicit pick for a collection that does not exist yet", () => {
    const s = pickCollection(emptyFileUnderState(), { name: "Barn Owls" });
    expect(effectiveCollection(s, "Sunsets", COLLECTIONS)).toEqual({
      name: "Barn Owls",
      slug: "barn-owls",
      source: "picked",
    });
  });

  it("keeps a cleared match cleared while the title slugs the same", () => {
    const cleared = clearCollection(
      emptyFileUnderState(),
      "Sunsets",
      COLLECTIONS,
    );
    expect(effectiveCollection(cleared, "Sunsets", COLLECTIONS)).toBeNull();
    // Same slug, different typing — the clear must still stick.
    expect(effectiveCollection(cleared, "  sunsets  ", COLLECTIONS)).toBeNull();
    expect(effectiveCollection(cleared, "SUNSETS!", COLLECTIONS)).toBeNull();
  });

  it("re-offers a NEW matching title after a clear", () => {
    const cleared = clearCollection(
      emptyFileUnderState(),
      "Sunsets",
      COLLECTIONS,
    );
    expect(effectiveCollection(cleared, "Portraits", COLLECTIONS)?.name).toBe(
      "Portraits",
    );
  });

  it("clears an explicit pick without re-offering the title match", () => {
    const picked = pickCollection(emptyFileUnderState(), { name: "Barn Owls" });
    const cleared = clearCollection(picked, "Sunsets", COLLECTIONS);
    expect(effectiveCollection(cleared, "Sunsets", COLLECTIONS)).toBeNull();
  });

  it("re-picking after a clear wins again", () => {
    const cleared = clearCollection(
      emptyFileUnderState(),
      "Sunsets",
      COLLECTIONS,
    );
    const again = pickCollection(cleared, {
      id: "id-Sunsets",
      name: "Sunsets",
    });
    expect(effectiveCollection(again, "Sunsets", COLLECTIONS)?.source).toBe(
      "picked",
    );
  });

  it("never auto-creates: an unmatched title leaves the row empty", () => {
    const s = emptyFileUnderState();
    expect(effectiveCollection(s, "A brand new name", COLLECTIONS)).toBeNull();
    expect(
      buildFileUnderRequestFields(s, "A brand new name", false, COLLECTIONS),
    ).toEqual({});
  });
});

// ── Wire fields ─────────────────────────────────────────────────────────────

describe("buildFileUnderRequestFields", () => {
  it("omits both fields entirely when nothing is filed", () => {
    const fields = buildFileUnderRequestFields(
      emptyFileUnderState(),
      "",
      true,
      COLLECTIONS,
    );
    expect(fields).toEqual({});
    expect("tags" in fields).toBe(false);
    expect("collection" in fields).toBe(false);
  });

  it("sends the ghost tag and the manual tags", () => {
    const s = state({ manualTags: ["kodak"] });
    expect(
      buildFileUnderRequestFields(s, "Sunset over the bay", true, []).tags,
    ).toEqual(["sunset-over-the-bay", "kodak"]);
  });

  it("omits tags when auto-tagging is off and nothing was typed", () => {
    const fields = buildFileUnderRequestFields(
      emptyFileUnderState(),
      "Sunset over the bay",
      false,
      [],
    );
    expect(fields.tags).toBeUndefined();
  });

  it("sends the collection by name only — never the host-local id", () => {
    const fields = buildFileUnderRequestFields(
      emptyFileUnderState(),
      "Sunsets",
      false,
      COLLECTIONS,
    );
    expect(fields.collection).toEqual({ name: "Sunsets" });
  });

  it("clamps to MAX_REQUEST_TAGS", () => {
    const manualTags = Array.from({ length: 30 }, (_, i) => `tag-${i}`);
    const fields = buildFileUnderRequestFields(
      state({ manualTags }),
      "",
      true,
      [],
    );
    expect(fields.tags).toHaveLength(MAX_REQUEST_TAGS);
    expect(fields.tags?.[0]).toBe("tag-0");
  });

  it("drops a tag the server would reject", () => {
    const s = state({
      manualTags: ["kodak", "x".repeat(REQUEST_TAG_MAX_LEN + 1)],
    });
    expect(buildFileUnderRequestFields(s, "", true, []).tags).toEqual([
      "kodak",
    ]);
  });
});

// ── Capability gate ─────────────────────────────────────────────────────────

describe("fileUnderAvailable", () => {
  it.each([
    [{ gallery: { can_delete: true, organize: true } }, true],
    [{ gallery: { can_delete: true, organize: false } }, false],
    [{ gallery: { can_delete: true } }, false],
    [{ gallery: null }, false],
    [{}, false],
    [null, false],
    [undefined, false],
  ])("%j → %s", (capabilities, expected) => {
    expect(fileUnderAvailable(capabilities)).toBe(expected);
  });

  it("is false for a truthy non-boolean organize field", () => {
    expect(fileUnderAvailable({ gallery: { organize: "yes" } })).toBe(false);
  });
});

// ── Validation (Rust is the authority) ──────────────────────────────────────

describe("validateRequestTag", () => {
  it("accepts an ordinary tag", () => {
    expect(validateRequestTag("golden hour")).toBeNull();
  });

  it("accepts exactly the maximum length", () => {
    expect(validateRequestTag("x".repeat(REQUEST_TAG_MAX_LEN))).toBeNull();
  });

  it("rejects one character past the maximum", () => {
    expect(validateRequestTag("x".repeat(REQUEST_TAG_MAX_LEN + 1))).toMatch(
      /64 characters/,
    );
  });

  it("counts Unicode scalars, not UTF-16 code units", () => {
    // 64 astral emoji are 128 UTF-16 units but 64 characters on both sides.
    expect(validateRequestTag("😀".repeat(REQUEST_TAG_MAX_LEN))).toBeNull();
    expect(
      validateRequestTag("😀".repeat(REQUEST_TAG_MAX_LEN + 1)),
    ).not.toBeNull();
  });

  it.each([[""], ["   "], ["#"], ["  #  "]])(
    "rejects the empty tag %j",
    (raw) => {
      expect(validateRequestTag(raw)).not.toBeNull();
    },
  );

  it("rejects control characters instead of silently stripping them", () => {
    expect(validateRequestTag(`kodak${BELL}gold`)).toMatch(/control character/);
  });
});

describe("validateNewTag", () => {
  it("accepts a fresh tag", () => {
    expect(validateNewTag("kodak", ["grain"])).toBeNull();
  });

  it("rejects a case-insensitive duplicate", () => {
    expect(validateNewTag("KODAK", ["kodak"])).toMatch(/already/i);
  });

  it("rejects once the request cap is reached", () => {
    const active = Array.from({ length: MAX_REQUEST_TAGS }, (_, i) => `t${i}`);
    expect(validateNewTag("one-more", active)).toMatch(/20/);
  });

  it("passes the per-tag validation through", () => {
    expect(validateNewTag(`kodak${BELL}`, [])).toMatch(/control character/);
  });
});

// ── Download name ───────────────────────────────────────────────────────────

/**
 * Parity table — mirrors `mold-core`'s `print_title` `download_file_name`.
 * Keep it identical to the Rust fixture table so both sides cross-check by
 * eye: `{title-slug}__{model}__s{seed}.{ext}`, falling back to
 * `{model}__s{seed}.{ext}` when the title is absent or unsluggable. The model
 * runs through the same slug algorithm (80-char cap) so ids carrying `:` —
 * `flux-dev:q4`, `cv:12345` — stay filesystem-safe.
 */
const DOWNLOAD_FILE_NAME_FIXTURES: ReadonlyArray<
  [
    {
      title?: string | null;
      model: string;
      seed?: number | string | null;
      ext?: string | null;
    },
    string,
  ]
> = [
  [
    {
      title: "Sunset over the bay",
      model: "flux-dev",
      seed: 12345,
      ext: "png",
    },
    "sunset-over-the-bay__flux-dev__s12345.png",
  ],
  [{ title: null, model: "flux-dev", seed: 7, ext: "png" }, "flux-dev__s7.png"],
  [{ model: "flux-dev", seed: 7, ext: "png" }, "flux-dev__s7.png"],
  [
    { title: "   ", model: "flux-dev", seed: 7, ext: "png" },
    "flux-dev__s7.png",
  ],
  [
    { title: "日本語", model: "flux-dev", seed: 7, ext: "png" },
    "flux-dev__s7.png",
  ],
  [
    { title: "Café au lait", model: "flux-dev:q4", seed: 0, ext: "jpg" },
    "caf-au-lait__flux-dev-q4__s0.jpg",
  ],
  [
    { title: "A", model: "cv:12345", seed: 42, ext: "mp4" },
    "a__cv-12345__s42.mp4",
  ],
  // A leading dot and stray case on the extension are normalized.
  [
    { title: "Sunsets", model: "flux-dev", seed: 1, ext: ".PNG" },
    "sunsets__flux-dev__s1.png",
  ],
  // u64 seeds arrive as strings when they exceed Number.MAX_SAFE_INTEGER.
  [
    {
      title: "Sunsets",
      model: "flux-dev",
      seed: "18446744073709551615",
      ext: "png",
    },
    "sunsets__flux-dev__s18446744073709551615.png",
  ],
  // An unresolved seed drops its segment rather than writing `sundefined`.
  [
    { title: "Sunsets", model: "flux-dev", seed: null, ext: "png" },
    "sunsets__flux-dev.png",
  ],
  // The title slug keeps the 40-char cap it has everywhere else.
  [
    { title: "a".repeat(45), model: "flux-dev", seed: 1, ext: "png" },
    `${"a".repeat(40)}__flux-dev__s1.png`,
  ],
  // Long compact ids survive: the model cap is 80, not 40.
  [
    {
      title: null,
      model: "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
      seed: 5,
      ext: "mp4",
    },
    "minimax-h3-fl2va-comfy-pruned-int8-turbo-4step-768p__s5.mp4",
  ],
  // Nothing sluggable at all still yields a usable name.
  [{ title: null, model: "日本語", seed: null, ext: "png" }, "print.png"],
  [
    { title: "Sunsets", model: "flux-dev", seed: 1, ext: null },
    "sunsets__flux-dev__s1",
  ],
];

describe("downloadFileName", () => {
  it.each(DOWNLOAD_FILE_NAME_FIXTURES)("%j → %s", (input, expected) => {
    expect(downloadFileName(input)).toBe(expected);
  });
});

// ── Suggestions ─────────────────────────────────────────────────────────────

const EXISTING_TAGS = [
  { name: "golden hour", count: 12 },
  { name: "grain", count: 30 },
  { name: "kodak gold", count: 4 },
  { name: "gold leaf", count: 9 },
  { name: "portrait", count: 2 },
];

describe("suggestTags", () => {
  it("orders prefix matches before substring matches", () => {
    expect(suggestTags(EXISTING_TAGS, "gold", []).map((t) => t.name)).toEqual([
      "golden hour",
      "gold leaf",
      "kodak gold",
    ]);
  });

  it("is case-insensitive", () => {
    expect(suggestTags(EXISTING_TAGS, "GOLD", []).map((t) => t.name)).toEqual([
      "golden hour",
      "gold leaf",
      "kodak gold",
    ]);
  });

  it("excludes already-active tags case-insensitively", () => {
    expect(
      suggestTags(EXISTING_TAGS, "gold", ["Golden Hour"]).map((t) => t.name),
    ).toEqual(["gold leaf", "kodak gold"]);
  });

  it("returns everything by count desc for an empty query", () => {
    expect(suggestTags(EXISTING_TAGS, "   ", []).map((t) => t.name)).toEqual([
      "grain",
      "golden hour",
      "gold leaf",
      "kodak gold",
      "portrait",
    ]);
  });

  it("returns nothing when the query matches nothing", () => {
    expect(suggestTags(EXISTING_TAGS, "zzz", [])).toEqual([]);
  });

  it("matches the typed tag through normalization (a leading #)", () => {
    expect(suggestTags(EXISTING_TAGS, "#grain", []).map((t) => t.name)).toEqual(
      ["grain"],
    );
  });
});

// ── Storage keys ────────────────────────────────────────────────────────────

describe("storage keys", () => {
  it("pins the web auto-tag setting key", () => {
    expect(AUTO_TAG_SETTING_WEB).toBe("mold.create.autoTagTitle.v1");
  });
});
