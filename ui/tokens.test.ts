/*
 * Structural invariants for tokens.css.
 *
 * The file deliberately duplicates palette values across scopes so that both
 * the browser cascade (system-follow via media query, forced modes via
 * data-theme) and design-tool token compilers (which only register custom
 * properties under plain attribute scopes, not :not() selectors) read the
 * same palettes. Duplication is safe only while these mirrors hold.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(join(__dirname, "tokens.css"), "utf8");

/** Strip comments, then collect `selector -> declarations` per media context. */
function parseBlocks(source: string) {
  const text = source.replace(/\/\*[\s\S]*?\*\//g, "");
  const blocks: { media: string | null; selector: string; body: string }[] = [];
  let i = 0;
  const readBody = (from: number): [string, number] => {
    let depth = 1;
    let j = from;
    while (j < text.length && depth > 0) {
      if (text[j] === "{") depth++;
      else if (text[j] === "}") depth--;
      j++;
    }
    return [text.slice(from, j - 1), j];
  };
  while (i < text.length) {
    const open = text.indexOf("{", i);
    if (open === -1) break;
    const selector = text.slice(i, open).trim().split("}").pop()!.trim();
    if (selector.startsWith("@media")) {
      const [inner, end] = readBody(open + 1);
      for (const b of parseBlocks(inner)) {
        blocks.push({ ...b, media: selector });
      }
      i = end;
    } else {
      const [body, end] = readBody(open + 1);
      blocks.push({ media: null, selector, body });
      i = end;
    }
  }
  return blocks;
}

function props(body: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const m of body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    // CSS treats runs of whitespace (incl. prettier's value wrapping) and
    // paren-adjacent whitespace as insignificant — normalize before comparing.
    out.set(
      m[1],
      m[2]
        .replace(/\s+/g, " ")
        .replace(/\(\s+/g, "(")
        .replace(/\s+\)/g, ")")
        .trim(),
    );
  }
  return out;
}

const blocks = parseBlocks(css);
const find = (selector: string, media: "light" | null) => {
  const hit = blocks.find(
    (b) =>
      b.selector === selector &&
      (media === "light"
        ? b.media?.includes("prefers-color-scheme: light")
        : b.media === null),
  );
  expect(
    hit,
    `block \`${selector}\`${media ? " in light media query" : ""}`,
  ).toBeDefined();
  return props(hit!.body);
};

const expectMirror = (
  a: Map<string, string>,
  b: Map<string, string>,
  label: string,
) => {
  expect([...a.keys()].sort(), `${label}: property sets differ`).toEqual(
    [...b.keys()].sort(),
  );
  for (const [k, v] of a) {
    expect(b.get(k), `${label}: ${k}`).toBe(v);
  }
};

describe("tokens.css scope structure", () => {
  it("uses only compiler-recognized scopes (no :not() selectors)", () => {
    for (const b of blocks) {
      expect(
        b.selector,
        "selectors must stay plain attribute scopes",
      ).not.toContain(":not(");
    }
  });

  it("system-follow light mirrors forced light (Safelight)", () => {
    expectMirror(
      find(":root", "light"),
      find(':root[data-theme="light"]', null),
      "safelight light",
    );
  });

  it("system-follow light mirrors forced light (Mold)", () => {
    expectMirror(
      find(':root[data-theme-family="mold"]', "light"),
      find(':root[data-theme-family="mold"][data-theme="light"]', null),
      "mold light",
    );
  });

  it("forced dark mirrors the base palette for every light-overridden prop (Safelight)", () => {
    const base = find(":root", null);
    const forcedDark = find(':root[data-theme="dark"]', null);
    // The forced-dark block exists to undo the light media block, so it must
    // cover exactly the same property names, with the base dark values.
    expect([...forcedDark.keys()].sort()).toEqual(
      [...find(":root", "light").keys()].sort(),
    );
    for (const [k, v] of forcedDark) {
      expect(base.get(k), `forced dark ${k} must equal the base value`).toBe(v);
    }
  });

  it("forced dark mirrors the family palette for every light-overridden prop (Mold)", () => {
    const moldBase = find(':root[data-theme-family="mold"]', null);
    const forcedDark = find(
      ':root[data-theme-family="mold"][data-theme="dark"]',
      null,
    );
    expect([...forcedDark.keys()].sort()).toEqual(
      [...find(':root[data-theme-family="mold"]', "light").keys()].sort(),
    );
    for (const [k, v] of forcedDark) {
      expect(
        moldBase.get(k),
        `mold forced dark ${k} must equal the family base value`,
      ).toBe(v);
    }
  });

  it("keeps the cascade tie-breaks: safelight before mold, family base before its light media block", () => {
    const order = (selector: string, media: boolean) => {
      const rx = new RegExp(
        selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\s*\\{",
      );
      const idx = [...css.matchAll(new RegExp(rx, "g"))].map((m) => m.index!);
      // Media-wrapped and bare occurrences are disambiguated by surrounding text.
      for (const at of idx) {
        const before = css.slice(0, at);
        const inMedia =
          (before.match(/@media[^{]*\{/g) ?? []).length >
          (before.match(/\}\s*\}/g) ?? []).length;
        if (inMedia === media) return at;
      }
      return -1;
    };
    const safelightForcedLight = order(':root[data-theme="light"]', false);
    const moldBase = order(':root[data-theme-family="mold"]', false);
    const moldLightMedia = order(':root[data-theme-family="mold"]', true);
    expect(safelightForcedLight).toBeGreaterThan(-1);
    expect(moldBase).toBeGreaterThan(safelightForcedLight);
    expect(moldLightMedia).toBeGreaterThan(moldBase);
  });
});

describe("tokens.css design-tool annotations", () => {
  // Tokens whose kind a compiler cannot infer from the value carry an
  // explicit `/* @kind ... */` trailing comment at their base definition.
  const kinds: Record<string, string> = {
    "--grad": "color",
    "--f-display": "font",
    "--f-body": "font",
    "--f-mono": "font",
    "--dur-quick": "other",
    "--dur-base": "other",
    "--dur-slow": "other",
    "--ease": "other",
  };
  for (const [token, kind] of Object.entries(kinds)) {
    it(`${token} is annotated @kind ${kind}`, () => {
      const rx = new RegExp(
        token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") +
          String.raw`\s*:[^;]+;\s*/\*\s*@kind ` +
          kind +
          String.raw`\s*\*/`,
      );
      expect(
        rx.test(css),
        `${token} needs a trailing /* @kind ${kind} */`,
      ).toBe(true);
    });
  }
});
