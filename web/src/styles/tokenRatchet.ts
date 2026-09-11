import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

/*
 * The measurers behind web/src/styles/tokens.legacy.test.ts, in a module of
 * their own so scripts/web-token-ratchet.ts can regenerate the frozen tables
 * without loading vitest.
 *
 * Web still speaks the pre-redesign vocabulary — the `--desk/--bath/…` bridge
 * in ui/tokens.css, literal radii and font sizes, a few hex colours — and the
 * plan is to migrate page by page. This is a RATCHET, not an allowlist: every
 * file keeps at most the count it has today, a file that reaches zero must
 * leave its table, and a NEW file starts at zero. The patterns are desktop's
 * (desktop/src/styles/tokens.legacy.test.ts) verbatim, so the two guards
 * cannot disagree about what a legacy token is.
 *
 * Regenerate a table with `bun run ratchet:web` (scripts/web-token-ratchet.ts)
 * after a migration pass, then paste — never raise a number by hand.
 */

const ROOT = "src";
const EXTENSIONS = new Set([".vue", ".ts", ".css"]);

const BOUND = "(?<![\\w-])";
const END = "(?![\\w-])";
export const LEGACY_PATTERNS = [
  new RegExp(
    `${BOUND}--(desk|bath|bench|rebate|halide|safelight|stop|ink|ink-2|ink-3|danger|edge|ce|sel-[a-z]+|card-hi|grad|print|on-media|on-status|f-(display|body|mono)|radius-(control|control-sm|control-lg|card|card-lg|pill)|control-edge|empty-surface|dur-(quick|base|slow)|ease)${END}`,
    "g",
  ),
  new RegExp(
    `${BOUND}(bg|text|border|fill|stroke|ring|from|to|via|shadow|accent|caret|divide|placeholder)-(desk|bath|bench|rebate|halide|safelight|stop|ink|ink-2|ink-3|edge|ce|control-edge|print-surface|empty-surface|card-hi)${END}`,
    "g",
  ),
  new RegExp(
    `${BOUND}(font-(display|body|utility)|text-(display|display-sm|display-lg|body|body-lg|caption|data|edge-code)|rounded-(chrome|media|pill|card|card-lg)|shadow-raised|edge-code|data-mono|kbd-hint|grain-shimmer)${END}`,
    "g",
  ),
];

export const LITERAL_STYLE = /\b(border-radius|font-size)\s*:\s*[^;{}]*\b\d+(?:\.\d+)?px/;

/** A hex colour in a style declaration. Issue refs in comments (`#1224`) are
 *  four decimal digits and never a colour; `#fff`/`#0B0B12` are. */
export const HEX_COLOUR = /(?<![\w&])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})(?![\w-])/;

export function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (entry === "node_modules" || entry.startsWith("dist")) continue;
    if (statSync(path).isDirectory()) yield* walk(path);
    else if (EXTENSIONS.has(path.slice(path.lastIndexOf(".")))) yield path;
  }
}

export function legacyUses(text: string): number {
  let count = 0;
  for (const line of text.split("\n")) {
    const probe = line.replace(/font-display:/g, "");
    for (const pattern of LEGACY_PATTERNS) {
      count += [...probe.matchAll(pattern)].length;
    }
  }
  return count;
}

export function literalStyles(text: string): number {
  let count = 0;
  for (const line of text.split("\n")) {
    if (line.includes("/* literal:")) continue;
    if (LITERAL_STYLE.test(line)) count += 1;
  }
  return count;
}

export function hexColours(text: string): number {
  let count = 0;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    // Only a declaration can paint: `prop: #hex`. A comment or a data URI is
    // not a colour anyone sees.
    if (!/^[a-z-]+\s*:/.test(trimmed) || trimmed.startsWith("//")) continue;
    if (/\bhex\b|url\(/.test(trimmed)) continue;
    if (HEX_COLOUR.test(trimmed)) count += 1;
  }
  return count;
}

export function measure(): {
  legacy: Record<string, number>;
  literal: Record<string, number>;
  hex: Record<string, number>;
} {
  const legacy: Record<string, number> = {};
  const literal: Record<string, number> = {};
  const hex: Record<string, number> = {};
  for (const file of walk(ROOT)) {
    if (file.endsWith(".test.ts")) continue;
    const text = readFileSync(file, "utf8");
    const a = legacyUses(text);
    const b = literalStyles(text);
    const c = hexColours(text);
    if (a) legacy[file] = a;
    if (b) literal[file] = b;
    if (c) hex[file] = c;
  }
  return { legacy, literal, hex };
}

