import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  ENGINE_KEY_SCHEMAS,
  ENV_KNOB_SCHEMAS,
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
const TAURI_COMMANDS_RELATIVE = "desktop/src-tauri/src/commands.rs";

/* `import.meta.url` is not a file URL in every environment these tests run in
 * (studio's own runner and desktop's both collect this file), so a Rust source
 * is found by walking up from the working directory. */
function repoFile(relative: string): string {
  let directory = process.cwd();
  for (;;) {
    const candidate = resolve(directory, relative);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory) {
      throw new Error(`could not find ${relative} above ${process.cwd()}`);
    }
    directory = parent;
  }
}

const source = readFileSync(repoFile(REGISTRY_RELATIVE), "utf8");

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

/**
 * A select's options must be values the setter accepts. The setter arm for a
 * key that validates an enum reads `"<key>" => { … validate_enum(v, &[…], key)`;
 * the option list is parsed out of that arm so an option the engine would 422
 * cannot ship. Keys whose setter validates no enum (a model name, a retention
 * ladder) are the schema's own to choose.
 */
function engineEnumFor(key: string): string[] | null {
  const arm = source.indexOf(`"${key}" => {`);
  if (arm < 0) return null;
  const next = source.indexOf('\n        "', arm + 1);
  const body = source.slice(arm, next < 0 ? undefined : next);
  const match = body.match(/validate_enum\(\s*\w+,\s*&\[([^\]]*)\]/);
  if (!match) return null;
  return [...match[1]!.matchAll(/"([^"]+)"/g)].map((m) => m[1]!);
}

describe("every select option is a value the engine accepts", () => {
  it("reads t5_variant's ladder (positive control)", () => {
    expect(engineEnumFor("t5_variant")).toEqual([
      "auto",
      "fp16",
      "q8",
      "q6",
      "q5",
      "q4",
      "q3",
    ]);
  });

  it("offers nothing the setter would refuse", () => {
    const checked: string[] = [];
    for (const schema of ENGINE_KEY_SCHEMAS) {
      if (schema.editor !== "select" || !schema.options) continue;
      const accepted = engineEnumFor(schema.key);
      if (!accepted) continue;
      checked.push(schema.key);
      for (const option of schema.options) {
        expect(
          accepted,
          `${schema.key} offers "${option.value}", which validate_enum in config_keys.rs refuses`,
        ).toContain(option.value);
      }
    }
    expect(checked).toEqual(
      expect.arrayContaining(["t5_variant", "qwen3_variant", "logging.level"]),
    );
  });
});

/*
 * A knob the Speed & memory section offers but `apply_engine_environment`
 * never copies into the engine's process is a control that silently does
 * nothing — and a variable the bridge copies with no row is a knob the app
 * pretends it cannot set. The two lists must be EQUAL, not merely nested:
 * asserting one direction only is how `MOLD_RESERVE_VRAM_MB` stayed
 * engine-shaping, documented and unofferable for a whole campaign. The Rust is
 * read rather than restated here: a list maintained by hand in two languages
 * is exactly the drift this test exists to catch.
 */
describe("the env knobs and the Tauri side's ENGINE_ENV_KEYS", () => {
  it("are the same set", () => {
    const commands = readFileSync(repoFile(TAURI_COMMANDS_RELATIVE), "utf8");
    const block = commands.match(
      /pub const ENGINE_ENV_KEYS: &\[&str\] = &\[([\s\S]*?)\];/,
    );
    expect(block, "ENGINE_ENV_KEYS not found in commands.rs").not.toBeNull();
    const allowlisted = [...block![1]!.matchAll(/"([A-Z0-9_]+)"/g)].map(
      (m) => m[1]!,
    );
    expect(allowlisted.length).toBeGreaterThan(0);
    const offered = ENV_KNOB_SCHEMAS.map((knob) =>
      knob.key.replace(/^env\./, ""),
    );
    expect([...offered].sort()).toEqual([...allowlisted].sort());
  });
});
