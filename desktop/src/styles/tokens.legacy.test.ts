import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

/*
 * The pre-redesign darkroom vocabulary (desk/bath/bench/rebate, safelight and
 * halide, the utility/body/display type roles, the chrome/media radii) lives
 * on only in ui/tokens.css's LEGACY BRIDGE, and only for the web SPA and the
 * phone surface. Desktop, ui/, and studio/ speak --mold-* and the utility
 * names desktop/src/styles/tokens.css maps from it. This is the migration's
 * definition of done.
 */

const ROOTS = ["src", "../ui", "../studio"];
// The desktop @theme map and the shared token sheet DEFINE the vocabulary;
// the phone and the font sheets sit on the bridge by design.
const SKIP = new Set(["src/mobile", "src/styles/tokens.css", "../ui/tokens.css", "../ui/fonts"]);
const EXTENSIONS = new Set([".vue", ".ts", ".css"]);

/** Whole-token boundaries: `--edge` must not match `--edge-code`'s successor
 * nor a data-test id like `generation-edge-code`. */
const BOUND = "(?<![\\w-])";
const END = "(?![\\w-])";
const LEGACY_PATTERNS = [
  new RegExp(
    `${BOUND}--(desk|bath|bench|rebate|halide|safelight|stop|ink|ink-2|ink-3|danger|edge|ce|sel-[a-z]+|card-hi|grad|print|on-media|on-status|f-(display|body|mono)|radius-(control|control-sm|control-lg|card|card-lg|pill)|control-edge|empty-surface|dur-(quick|base|slow)|ease)${END}`,
  ),
  new RegExp(
    `${BOUND}(bg|text|border|fill|stroke|ring|from|to|via|shadow|accent|caret|divide|placeholder)-(desk|bath|bench|rebate|halide|safelight|stop|ink|ink-2|ink-3|edge|ce|control-edge|print-surface|empty-surface|card-hi)${END}`,
  ),
  new RegExp(
    `${BOUND}(font-(display|body|utility)|text-(display|display-sm|display-lg|body|body-lg|caption|data|edge-code)|rounded-(chrome|media|pill|card|card-lg)|shadow-raised|edge-code|data-mono|kbd-hint|grain-shimmer)${END}`,
  ),
];

function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (SKIP.has(path) || entry === "node_modules" || entry.startsWith("dist")) continue;
    if (statSync(path).isDirectory()) yield* walk(path);
    else if (EXTENSIONS.has(path.slice(path.lastIndexOf(".")))) yield path;
  }
}

function offenders(text: string): string[] {
  const hits: string[] = [];
  text.split("\n").forEach((line, index) => {
    // `font-display:` is a CSS property, not the retired type role.
    const probe = line.replace(/font-display:/g, "");
    for (const pattern of LEGACY_PATTERNS) {
      const match = probe.match(pattern);
      if (match) hits.push(`${index + 1}: ${match[0]}`);
    }
  });
  return hits;
}

/** Two font-size utilities on one element: the later one wins silently. */
const SIZE_UTILITY = /(?<![\w:-])text-(micro|xs|sm|base|md|lg|xl)(?![\w-])/g;
function doubleSized(text: string): string[] {
  // Static class attributes only: a `:class` ternary names alternatives.
  return [...text.matchAll(/(?<![:\w-])class="([^"]*)"/g)]
    .map((m) => m[1]!)
    .filter((attr) => [...attr.matchAll(SIZE_UTILITY)].length > 1);
}

describe("font-size utilities", () => {
  it("are recognised by the guard (positive control)", () => {
    expect(doubleSized('class="font-mono text-xs text-sm"')).toHaveLength(1);
    expect(doubleSized('class="text-xs sm:text-sm"')).toHaveLength(0);
    expect(doubleSized('class="text-micro text-fg-dim"')).toHaveLength(0);
  });

  it("never stack on one element in desktop sources", () => {
    const found: string[] = [];
    for (const file of walk("src")) {
      if (!file.endsWith(".vue")) continue;
      for (const hit of doubleSized(readFileSync(file, "utf8"))) found.push(`${file}: ${hit}`);
    }
    expect(found).toEqual([]);
  });
});

describe("legacy token vocabulary", () => {
  it("is recognised by the guard (positive control)", () => {
    // One hit per pattern family: the colour utilities and the type roles.
    expect(offenders('class="bg-bench text-ink-3 font-utility rounded-chrome"')).toHaveLength(2);
    expect(offenders("color: var(--safelight);")).toHaveLength(1);
    expect(offenders("border-radius: var(--radius-control);")).toHaveLength(1);
    expect(offenders("font-display: block;")).toHaveLength(0);
    expect(offenders('data-test="generation-edge-code"')).toHaveLength(0);
    expect(offenders("--mold-border-control")).toHaveLength(0);
  });

  it("appears nowhere in desktop, ui, or studio sources", () => {
    const found: string[] = [];
    for (const root of ROOTS) {
      for (const file of walk(root)) {
        if (file === "src/styles/tokens.legacy.test.ts") continue;
        for (const hit of offenders(readFileSync(file, "utf8"))) found.push(`${file}:${hit}`);
      }
    }
    expect(found).toEqual([]);
  });
});
