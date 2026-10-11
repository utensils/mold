/**
 * Model names have ONE authority: `mold_core::model_presentation`, reached by
 * the TypeScript surfaces through `/api/models` and the generated
 * `generated/modelPresentationV1.ts`, read only by `modelDisplay.ts` and
 * `modelFamily.ts`. Every hand-written table this replaced (a MiniMax H3 name
 * map, a family-label map in studio and a second copy in a web card) had
 * already drifted from the server. This scan fails on a new one.
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { FAMILY_LABELS } from "./generated/modelPresentationV1";

function repoRoot(): string {
  let directory = process.cwd();
  for (;;) {
    if (existsSync(join(directory, "studio", "lib", "modelDisplay.ts"))) {
      return directory;
    }
    const parent = dirname(directory);
    if (parent === directory) throw new Error("could not find the repo root");
    directory = parent;
  }
}

const ROOT = repoRoot();
const SCANNED = ["studio", "web/src", "desktop/src", "ui"];

function sources(directory: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(directory)) {
    if (name === "node_modules" || name === "generated" || name === "dist") {
      continue;
    }
    const path = join(directory, name);
    if (statSync(path).isDirectory()) out.push(...sources(path));
    else if (
      /\.(ts|vue)$/.test(name) &&
      !/\.test\.ts$|\.testFixtures\.ts$|\.d\.ts$/.test(name)
    ) {
      out.push(path);
    }
  }
  return out;
}

const escape = (value: string) => value.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&");

/* The utility families (`upscaler`, `controlnet`, `qwen3-expand`) double as
 * model KIND slugs, whose labels (`modelMetadata.ts`) are a different fact. */
const UTILITY_FAMILIES = new Set(["upscaler", "controlnet", "qwen3-expand"]);
const GENERATION_FAMILIES = Object.keys(FAMILY_LABELS).filter(
  (slug) => !UTILITY_FAMILIES.has(slug),
);

/** `flux2: "FLUX.2"` / `"qwen-image": "Qwen Image"` — a family slug keyed
 *  to a string literal is a family-label table. */
const FAMILY_ENTRY = new RegExp(
  String.raw`^\s*["']?(?:${GENERATION_FAMILIES.map(escape).join("|")})["']?\s*:\s*["'\x60]`,
);
const RETIRED_TABLES =
  /\b(MINIMAX_H3_DISPLAY_NAMES|FAMILY_LABELS\s*(?::[^=]*)?=|label:\s*"(?:FLUX|SDXL|SD 1\.5|QWEN|Z-IMAGE|LTX VIDEO|WUERSTCHEN)")/;

describe("model presentation authority", () => {
  const files = SCANNED.flatMap((dir) => sources(resolve(ROOT, dir)));

  it("scans the client sources", () => {
    expect(files.length).toBeGreaterThan(100);
  });

  it("no client source declares its own model-name or family-label table", () => {
    const offenders: string[] = [];
    for (const file of files) {
      readFileSync(file, "utf8")
        .split("\n")
        .forEach((line, index) => {
          if (FAMILY_ENTRY.test(line) || RETIRED_TABLES.test(line)) {
            offenders.push(
              `${relative(ROOT, file)}:${index + 1}: ${line.trim()}`,
            );
          }
        });
    }
    expect(offenders).toEqual([]);
  });
});
