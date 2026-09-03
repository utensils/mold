/**
 * The reference-image contract, read from the SHARED fixture the Rust side
 * reads (`tests/fixtures/flux2/reference-parity-v1.json`) — the wan
 * `surface-parity-v1.json` precedent.
 *
 * What is pinned here is that the ADVERTISED block decides, on every surface:
 * the recipe beats any name sniff, a `hidden` block means no reference UI at
 * all, and `source_relation` alone separates Qwen's target-first strip, Dev's
 * references-replace-source strip, and Klein's two mutually exclusive wells.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { baseGenerationCapabilities } from "./generationCapabilities";
import type { ReferenceImagesProfile } from "./referenceImagesProfile";
import { sourceMediaPlan } from "./sourceMediaPlan";
import { sdxlRecipe } from "./generationProfile.testFixtures";
import type { GenerationRecipeProfile } from "./generationProfile";

const FIXTURE_RELATIVE = "tests/fixtures/flux2/reference-parity-v1.json";

interface ParityRow {
  model: string;
  family: string;
  mode: ReferenceImagesProfile["mode"];
  required: boolean;
  max_count: number | null;
  primary_is_target: boolean;
  source_relation: ReferenceImagesProfile["source_relation"];
}

function fixturePath(): string {
  let directory = process.cwd();
  for (;;) {
    const candidate = resolve(directory, FIXTURE_RELATIVE);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory) {
      throw new Error(
        `could not find ${FIXTURE_RELATIVE} above ${process.cwd()}`,
      );
    }
    directory = parent;
  }
}

const fixture = JSON.parse(readFileSync(fixturePath(), "utf8")) as {
  schema: string;
  models: ParityRow[];
};

/** A recipe carrying nothing but the row's advertised reference block. */
function recipeFor(row: ParityRow): GenerationRecipeProfile {
  const recipe = sdxlRecipe();
  const block: ReferenceImagesProfile = {
    mode: row.mode,
    required: row.required,
    max_count: row.max_count,
    primary_is_target: row.primary_is_target,
    source_relation: row.source_relation,
  };
  return {
    ...recipe,
    capabilities: {
      ...recipe.capabilities,
      reference_images: block,
    },
  } as GenerationRecipeProfile;
}

function capsFor(row: ParityRow) {
  return baseGenerationCapabilities(
    row.family,
    row.model,
    null,
    null,
    null,
    recipeFor(row),
  );
}

describe("flux2 reference parity fixture", () => {
  it("is the v1 schema", () => {
    expect(fixture.schema).toBe("mold.flux2.reference-parity.v1");
    expect(fixture.models.length).toBeGreaterThan(0);
  });

  it("projects every advertised block onto the client capability", () => {
    for (const row of fixture.models) {
      const caps = capsFor(row);
      if (row.mode === "hidden") {
        expect(caps.referenceImages, row.model).toBeNull();
        continue;
      }
      expect(caps.referenceImages, row.model).toEqual({
        required: row.required,
        max: row.max_count,
        primaryIsTarget: row.primary_is_target,
        sourceRelation: row.source_relation,
        maxPixelsSingle: null,
        maxPixelsMulti: null,
        reason: null,
      });
    }
  });

  it("derives one source-image mode per relation", () => {
    const modes = Object.fromEntries(
      fixture.models.map((row) => [row.model, capsFor(row).sourceImageMode]),
    );
    expect(modes).toEqual({
      "flux2-dev:bf16": "references",
      "flux2-klein:bf16": "single-or-references",
      "flux2-klein-base-9b:q8": "single-or-references",
      "qwen-image-edit-2511:q4": "qwen-edit",
      "flux-dev:q4": "single",
    });
  });

  it("renders Klein's two wells and Dev's strip from the same block", () => {
    const klein = fixture.models.find((r) => r.model === "flux2-klein:bf16")!;
    const plan = sourceMediaPlan(capsFor(klein));
    expect(plan).toEqual({
      kind: "single-or-references",
      single: { required: false, endFrame: false, video: false },
      references: { max: 4, maxPixelsSingle: null, maxPixelsMulti: null },
    });

    const dev = fixture.models.find((r) => r.model === "flux2-dev:bf16")!;
    expect(sourceMediaPlan(capsFor(dev))).toEqual({
      kind: "attachments",
      max: 4,
      required: false,
      primary: null,
    });
  });

  it("falls back to the legacy sniff ONLY when the block is absent", () => {
    // An older host advertises no recipe at all: Dev and Qwen keep the
    // behaviour every client shipped before the contract existed…
    expect(
      baseGenerationCapabilities("flux2", "flux2-dev:bf16").sourceImageMode,
    ).toBe("references");
    expect(
      baseGenerationCapabilities("qwen-image-edit", "qwen-image-edit-2511:q4")
        .sourceImageMode,
    ).toBe("qwen-edit");
    // …and Klein deliberately does not: an older host has no Klein reference
    // engine, so offering the wells would promise a render it would refuse.
    expect(
      baseGenerationCapabilities("flux2", "flux2-klein:bf16").sourceImageMode,
    ).toBe("single");
    expect(
      baseGenerationCapabilities("flux2", "flux2-klein:bf16").referenceImages,
    ).toBeNull();
  });

  it("lets a recipe overrule the name sniff in both directions", () => {
    // A klein recipe that advertises references wins over "klein has none"…
    const klein = fixture.models.find((r) => r.model === "flux2-klein:bf16")!;
    expect(capsFor(klein).referenceImages?.sourceRelation).toBe("exclusive");
    // …and a dev recipe whose block is hidden wins over the dev name sniff.
    const hiddenDev: ParityRow = {
      ...fixture.models.find((r) => r.model === "flux2-dev:bf16")!,
      mode: "hidden",
      max_count: 0,
    };
    expect(capsFor(hiddenDev).referenceImages).toBeNull();
    expect(capsFor(hiddenDev).sourceImageMode).toBe("single");
  });
});
