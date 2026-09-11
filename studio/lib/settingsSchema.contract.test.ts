import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  PER_STYLE_FIELDS,
  SECTIONS,
  schemaFor,
  sectionForConfigKey,
} from "./settingsSchema";

/*
 * The engine's key registry is the authority, and this test is what keeps the
 * client's schema level with it.
 *
 * Without it, a key added to `config_keys.rs` reaches Settings as a raw row
 * labelled by its own name and helped by "Server-provided configuration key." —
 * which is how web's Advanced section came to hold 25 uncurated keys. That
 * sentence now means exactly one thing: a key NEWER than this client.
 *
 * It reads the Rust source as text rather than `/api/config` rows, because a
 * registered key need not be returned: `routes_config.rs` pushes a row only
 * where `get_static_value` answers, and `umt5_variant` has no arm (#778). A
 * rows-based test would never cover it.
 */

const REGISTRY_RELATIVE = "crates/mold-core/src/config_keys.rs";

/* `import.meta.url` is not a file URL in every environment these tests run in
 * (studio's own runner and desktop's both collect this file), so the registry
 * is found by walking up from the working directory. */
function registryPath(): string {
  let directory = process.cwd();
  for (;;) {
    const candidate = resolve(directory, REGISTRY_RELATIVE);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory) {
      throw new Error(
        `could not find ${REGISTRY_RELATIVE} above ${process.cwd()}`,
      );
    }
    directory = parent;
  }
}

const source = readFileSync(registryPath(), "utf8");

/** The body of a `pub const NAME: … = &[ … ];` slice literal. */
function sliceBody(name: string): string {
  const start = source.indexOf(`pub const ${name}`);
  expect(start, `${name} in config_keys.rs`).toBeGreaterThanOrEqual(0);
  const open = source.indexOf("&[", start);
  const close = source.indexOf("\n];", open);
  expect(close, `${name}'s closing bracket`).toBeGreaterThan(open);
  return source.slice(open, close);
}

/** `pub const NAME: &str = "value";` — ALL_KEYS names three keys by identifier. */
function strConst(name: string): string {
  const match = source.match(new RegExp(`pub const ${name}: &str = "([^"]+)"`));
  expect(match, `${name} in config_keys.rs`).toBeTruthy();
  return match![1]!;
}

const allKeysBody = sliceBody("ALL_KEYS");
const ALL_KEYS = [
  ...allKeysBody.matchAll(/key: (?:"([^"]+)"|([A-Z][A-Z0-9_]*))/g),
].map((match) => match[1] ?? strConst(match[2]!));

const MODEL_FIELDS = [
  ...sliceBody("MODEL_FIELDS").matchAll(/\("([^"]+)",/g),
].map((m) => m[1]!);

describe("the engine key registry, parsed from config_keys.rs", () => {
  it("is read correctly (positive control)", () => {
    // The three identifier-named keys resolve, and the literal ones survive.
    expect(ALL_KEYS).toContain("gallery.trash_retention_days");
    expect(ALL_KEYS).toContain("queue.held_retention_days");
    expect(ALL_KEYS).toContain("generate.auto_tag_title");
    expect(ALL_KEYS).toContain("default_model");
    expect(ALL_KEYS).toContain("umt5_variant");
    expect(new Set(ALL_KEYS).size).toBe(ALL_KEYS.length);
  });

  /* Asserted as a floor, not an equality: adding a key to the registry should
   * fail on the schema assertion below, which names the key and says what to
   * do, rather than on a count nobody can act on. */
  it("has not shrunk below the 47 keys this schema was written against", () => {
    expect(ALL_KEYS.length).toBeGreaterThanOrEqual(47);
  });
});

describe("every engine key has a curated schema", () => {
  it("covers all of ALL_KEYS", () => {
    const missing = ALL_KEYS.filter((key) => schemaFor(key) === null);
    expect(
      missing,
      `these engine keys have no schema and would render as raw "Server-provided configuration key." rows — add each to ENGINE_KEY_SCHEMAS in studio/lib/settingsSchema.ts: ${missing.join(", ")}`,
    ).toEqual([]);
  });

  it("routes each one to exactly one section that the schema declares", () => {
    for (const key of ALL_KEYS) {
      const id = sectionForConfigKey(key);
      expect(id, `${key} routes nowhere`).not.toBeNull();
      expect(
        SECTIONS.filter((section) => section.id === id),
        `${key} → ${id}`,
      ).toHaveLength(1);
      expect(schemaFor(key)?.section, `${key}'s schema section`).toBe(id);
    }
  });

  it("still sends a key this client has never heard of to Advanced", () => {
    // The forward-compatibility path, and the positive control for the two
    // assertions above.
    expect(schemaFor("definitely.not.a.key")).toBeNull();
    expect(sectionForConfigKey("definitely.not.a.key")).toBe("advanced");
  });
});

describe("PER_STYLE_FIELDS", () => {
  it("is MODEL_FIELDS, in the engine's own order", () => {
    expect(MODEL_FIELDS.length).toBeGreaterThanOrEqual(8);
    expect([...PER_STYLE_FIELDS]).toEqual(MODEL_FIELDS);
  });
});
