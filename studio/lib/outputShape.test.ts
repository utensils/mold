import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  canonicalFamilyFor,
  outputFamilyLabel,
  resolveOutputShape,
  sizeForFamily,
  snapOutputSize,
  SOURCE_FAMILY_ID,
  type OutputShapeInput,
  type OutputShapeModel,
} from "./outputShape";
import { resolutionProfileError } from "./generationProfile";
import {
  resolveSourceCanvasTransition,
  resolveSourceResolution,
} from "./sourceResolution";
import type { GenerationProfileSet } from "./generated/generationProfileV1";

interface ProfileFixture {
  profiles: {
    models: { model: string; family: string }[];
    profile: GenerationProfileSet;
  }[];
}

/**
 * The generated Rust profile registry — the same file
 * `generate_generation_profiles --check` keeps in sync with `mold-core`. The
 * browser resolver is contract-tested against the server's own tables so the
 * two cannot drift.
 */
function fixturePath(): string {
  const relative = "docs/generated/generation-profiles-v1.json";
  let directory = process.cwd();
  for (let depth = 0; depth < 8; depth += 1) {
    const candidate = resolve(directory, relative);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  throw new Error(`${relative} not found above ${process.cwd()}`);
}

const fixture = JSON.parse(
  readFileSync(fixturePath(), "utf8"),
) as ProfileFixture;

function modelFor(name: string): OutputShapeModel {
  for (const entry of fixture.profiles) {
    const model = entry.models.find((candidate) => candidate.model === name);
    if (!model) continue;
    const recipe =
      entry.profile.recipes.find(
        (candidate) => candidate.id === entry.profile.default_recipe_id,
      ) ?? entry.profile.recipes[0]!;
    return {
      name: model.model,
      family: model.family,
      default_width: recipe.defaults.width,
      default_height: recipe.defaults.height,
      generation_profile: entry.profile,
    };
  }
  throw new Error(`unknown fixture model: ${name}`);
}

/** The composed LTX-2 tier from the #1166 report: alignment 64, 4096 px span. */
const composedLtx2 = modelFor("ltx-2-19b-dev:fp8");

function shape(overrides: Partial<OutputShapeInput> = {}) {
  const input: OutputShapeInput = {
    model: composedLtx2,
    pipeline: null,
    width: 1216,
    height: 704,
    intent: "model-default",
    ...overrides,
  };
  return { input, result: resolveOutputShape(input) };
}

describe("canonical shape families", () => {
  it("collapses model-specific near-misses onto one widescreen family", () => {
    expect(canonicalFamilyFor(1216, 704).id).toBe("16:9"); // 19:11
    expect(canonicalFamilyFor(1920, 1088).id).toBe("16:9"); // 30:17
    expect(canonicalFamilyFor(3840, 2112).id).toBe("16:9"); // 20:11
    expect(canonicalFamilyFor(832, 480).id).toBe("16:9"); // wan 26:15
    expect(canonicalFamilyFor(1344, 768).id).toBe("16:9"); // minimax 7:4
    expect(canonicalFamilyFor(704, 1216).id).toBe("9:16");
  });

  it("keeps 3:2 and 4:3 distinct shapes", () => {
    expect(canonicalFamilyFor(704, 480).id).toBe("3:2"); // 22:15
    expect(canonicalFamilyFor(1216, 832).id).toBe("3:2"); // 19:13
    expect(canonicalFamilyFor(1024, 768).id).toBe("4:3");
    expect(canonicalFamilyFor(1152, 896).id).toBe("5:4"); // 9:7
  });

  it("gives an off-tolerance size its own reduced label rather than hiding it", () => {
    expect(canonicalFamilyFor(1000, 400).id).toBe("5:2");
    expect(outputFamilyLabel(1000, 400)).toBe("5:2");
  });

  it("names the header chip from the canonical family", () => {
    expect(outputFamilyLabel(1216, 704)).toBe("16:9");
    expect(outputFamilyLabel(1024, 1024)).toBe("1:1");
  });
});

describe("resolveOutputShape", () => {
  it("collapses LTX-2's eleven gcd chips to five canonical families", () => {
    const { result } = shape();
    expect(result.families.map((family) => family.id)).toEqual([
      "1:1",
      "3:2",
      "2:3",
      "16:9",
      "9:16",
    ]);
  });

  it("labels sizes by pixels with data-derived marks, never list position", () => {
    const { result } = shape();
    expect(result.selectedFamilyId).toBe("16:9");
    expect(result.sizes.map((size) => size.label)).toEqual([
      "1024×576",
      "1216×704",
      "1920×1088",
      "2560×1408",
      "3840×2112",
    ]);
    expect(result.sizes.map((size) => size.mark)).toEqual([
      null,
      "Default",
      null,
      null,
      null,
    ]);
    expect(result.sizes[1]!.megapixels).toBe("0.9 MP");
    expect(result.state).toBe("model-default");
    expect(result.status).toBe("1216×704 · Model default");
    expect(result.badge).toBe("Default");
  });

  it("follows a 1:1 source onto the closest model-valid size", () => {
    const source = { width: 1024, height: 1024 };
    const automatic = sizeForFamily(SOURCE_FAMILY_ID, {
      model: composedLtx2,
      width: 1216,
      height: 704,
      source,
      intent: "source",
    });
    expect(automatic).toEqual({ width: 768, height: 768 });

    const { result } = shape({
      source,
      intent: "source",
      width: 768,
      height: 768,
    });
    expect(result.state).toBe("follows-source");
    expect(result.badge).toBe("Source");
    expect(result.selectedFamilyId).toBe(SOURCE_FAMILY_ID);
    expect(result.family.label).toBe("Source");
    expect(result.status).toBe(
      "768×768 · Follows source (1024×1024, 64 px grid)",
    );
    // The 1:1 ladder plus the exact source size, so Match source is a pill.
    expect(result.sizes.map((size) => size.label)).toEqual([
      "512×512",
      "768×768",
      "1024×1024",
    ]);
    expect(result.sizes.at(-1)!.mark).toBe("Source");
  });

  it("keeps Source square for a Wan model whose authored presets are not square", () => {
    const wan = {
      family: "wan",
      default_width: 832,
      default_height: 480,
      recommended_dimensions: [
        { width: 832, height: 480 },
        { width: 480, height: 832 },
        { width: 1280, height: 720 },
        { width: 720, height: 1280 },
      ],
      max_pixels: 1_800_000,
      dimension_alignment: 16,
    };
    const source = { width: 1024, height: 1024 };

    expect(
      sizeForFamily(SOURCE_FAMILY_ID, {
        model: wan,
        width: 832,
        height: 480,
        source,
        intent: "source",
      }),
    ).toEqual(source);

    const result = resolveOutputShape({
      model: wan,
      width: 1024,
      height: 1024,
      source,
      intent: "source",
    });
    expect(result.selectedFamilyId).toBe(SOURCE_FAMILY_ID);
    expect(result.status).toBe("1024×1024 · Matches source");
    expect(result.sizes.map((size) => size.label)).toEqual(["1024×1024"]);
  });

  it("reports an exact source canvas as Matches source", () => {
    const { result } = shape({
      source: { width: 1024, height: 1024 },
      intent: "source-exact",
      width: 1024,
      height: 1024,
    });
    expect(result.state).toBe("matches-source");
    expect(result.status).toBe("1024×1024 · Matches source");
  });

  it("never claims a source badge for a canvas the source did not produce", () => {
    const { result } = shape({
      source: { width: 1024, height: 1024 },
      intent: "manual",
      width: 1216,
      height: 704,
    });
    expect(result.state).toBe("model-default");
    expect(result.badge).not.toBe("Source");
    expect(result.selectedFamilyId).toBe("16:9");
    expect(result.status).toBe("1216×704 · Model default · source cropped 42%");
  });

  it("describes a manual canvas as manual", () => {
    const { result } = shape({ intent: "manual", width: 1920, height: 1088 });
    expect(result.state).toBe("manual");
    expect(result.badge).toBe("Manual");
    expect(result.status).toBe("1920×1088 · Manual");
  });

  it("marks an off-ladder custom size and still lights its family chip", () => {
    const { result } = shape({ intent: "manual", width: 1280, height: 704 });
    expect(result.family.id).toBe("16:9");
    expect(result.selectedFamilyId).toBe("16:9");
    expect(result.approximate).toBe(true);
    expect(result.sizes.find((size) => size.label === "1280×704")?.mark).toBe(
      "Custom",
    );
  });

  it("calls a canvas outside every rendered family custom", () => {
    const { result } = shape({ intent: "manual", width: 1280, height: 512 });
    expect(result.family.id).toBe("custom");
    expect(result.family.label).toBe("5:2");
    expect(result.selectedFamilyId).toBe("16:9");
    expect(result.approximate).toBe(true);
  });

  it("surfaces the profile's advisory finding as a warning", () => {
    const { result } = shape({ intent: "manual", width: 1210, height: 704 });
    expect(result.warnings).toHaveLength(1);
    expect(result.warnings[0]!.level).toBe("warn");
  });

  it("keeps a family chip on the authored ladder, never on invented pixels", () => {
    const { input } = shape();
    expect(sizeForFamily("1:1", { ...input, intent: "manual" })).toEqual({
      width: 768,
      height: 768,
    });
    expect(sizeForFamily("9:16", { ...input, intent: "manual" })).toEqual({
      width: 704,
      height: 1216,
    });
  });

  it("keeps MiniMax H3's authored sizes exact", () => {
    const h3 = modelFor("minimax-h3-fl2va:official-bf16");
    const result = resolveOutputShape({
      model: h3,
      width: 1344,
      height: 768,
      intent: "model-default",
    });
    expect(result.selectedFamilyId).toBe("16:9");
    expect(result.approximate).toBe(false);
    expect(
      sizeForFamily("16:9", {
        model: h3,
        width: 1344,
        height: 768,
        intent: "manual",
      }),
    ).toEqual({ width: 1344, height: 768 });
  });

  it("offers the compact H3 recommendations and admits the whole rule", () => {
    const compact = modelFor("minimax-h3-fl2va:comfy-pruned-int8");
    const input: OutputShapeInput = {
      model: compact,
      width: 1344,
      height: 768,
      intent: "model-default",
    };
    const result = resolveOutputShape(input);

    // The compact stack is a RANGE with a recommended ladder now, so the
    // chips span landscape, square, and portrait rather than the two
    // canvases a campaign happened to run.
    expect(result.selectedFamilyId).toBe("16:9");
    expect(result.families.map((family) => family.id)).toEqual([
      "1:1",
      "16:9",
      "9:16",
    ]);
    expect(sizeForFamily("16:9", { ...input, intent: "manual" })).toEqual({
      width: 1344,
      height: 768,
    });
    expect(sizeForFamily("1:1", { ...input, intent: "manual" })).toEqual({
      width: 960,
      height: 960,
    });
    expect(sizeForFamily("9:16", { ...input, intent: "manual" })).toEqual({
      width: 768,
      height: 1344,
    });

    // The 16:9 ladder carries every recommended landscape canvas, and only
    // the model default is marked.
    expect(result.sizes.map((size) => [size.width, size.height])).toEqual([
      [1024, 576],
      [1280, 704],
      [1344, 768],
    ]);
    expect(result.sizes.map((size) => size.mark)).toEqual([
      null,
      null,
      "Default",
    ]);
    const square = resolveOutputShape({
      ...input,
      width: 768,
      height: 768,
      intent: "manual",
    });
    expect(square.selectedFamilyId).toBe("1:1");
    expect(square.sizes.map((size) => [size.width, size.height])).toEqual([
      [768, 768],
      [960, 960],
    ]);

    // A canonical upstream resolver output nobody ran a campaign on is now
    // admitted by the rule, and so is an ordinary aligned canvas; only a
    // shape outside the rule is refused.
    const recipe = compact.generation_profile!.recipes[0]!;
    for (const [width, height] of [
      [1344, 768],
      [768, 768],
      [1024, 768],
      [1024, 576],
      [512, 1984],
    ]) {
      expect(
        resolutionProfileError(width!, height!, recipe.resolution),
      ).toBeNull();
    }
    // Off the 32 stride, over the area ceiling, and under the axis floor.
    for (const [width, height] of [
      [1000, 600],
      [1056, 992],
      [224, 896],
    ]) {
      expect(
        resolutionProfileError(width!, height!, recipe.resolution),
      ).toBeTruthy();
    }
  });

  it("keeps following the source across a model switch, and manual across it too", () => {
    const source = { width: 1024, height: 1024 };
    const sdxl = modelFor("sdxl-base:fp16");
    const following = resolveSourceCanvasTransition({
      source: resolveSourceResolution(source, sdxl),
      automatic: sizeForFamily(SOURCE_FAMILY_ID, {
        model: sdxl,
        width: 1216,
        height: 704,
        source,
        intent: "source",
      })!,
      replaced: false,
      intent: "source",
    });
    expect(following).toEqual({ width: 1024, height: 1024 });
    expect(
      resolveOutputShape({
        model: sdxl,
        width: following!.width,
        height: following!.height,
        source,
        intent: "source",
      }).state,
    ).toBe("matches-source");

    // A manual pick is never re-resolved by the same switch.
    expect(
      resolveSourceCanvasTransition({
        source: resolveSourceResolution(source, sdxl),
        automatic: { width: 1024, height: 1024 },
        replaced: false,
        intent: "manual",
      }),
    ).toBeNull();
    const manual = resolveOutputShape({
      model: sdxl,
      width: 1344,
      height: 768,
      source,
      intent: "manual",
    });
    expect(manual.state).toBe("manual");
    expect(manual.badge).toBe("Manual");
    expect(manual.selectedFamilyId).toBe("16:9");
  });

  it("snaps a typed size onto the recipe grid", () => {
    expect(snapOutputSize({ width: 1201, height: 705 }, composedLtx2)).toEqual({
      width: 1152,
      height: 704,
    });
  });
});

describe("profile contract", () => {
  const recipes = fixture.profiles.flatMap((entry) =>
    entry.profile.recipes.map((recipe) => ({
      name: `${entry.models[0]!.model} · ${recipe.id}`,
      family: entry.models[0]!.family,
      pipeline: recipe.request_selector.pipeline ?? null,
      profile: entry.profile,
      recipe,
    })),
  );

  it("covers every shipped profile", () => {
    expect(recipes.length).toBeGreaterThan(50);
  });

  it("assigns every authored preset to exactly one rendered family", () => {
    for (const entry of recipes) {
      const model: OutputShapeModel = {
        name: entry.name,
        family: entry.family,
        default_width: entry.recipe.defaults.width,
        default_height: entry.recipe.defaults.height,
        generation_profile: entry.profile,
      };
      const presets = entry.recipe.resolution.aspect_groups.flatMap(
        (group) => group.presets,
      );
      if (presets.length === 0) continue;
      const result = resolveOutputShape({
        model,
        pipeline: entry.pipeline,
        width: entry.recipe.defaults.width,
        height: entry.recipe.defaults.height,
        intent: "model-default",
      });
      const familyIds = new Set(result.families.map((family) => family.id));
      for (const preset of presets) {
        const family = canonicalFamilyFor(preset.width, preset.height);
        expect(
          familyIds.has(family.id),
          `${entry.name}: ${preset.width}x${preset.height} dropped from the shape row`,
        ).toBe(true);
        const ladder = resolveOutputShape({
          model,
          pipeline: entry.pipeline,
          width: preset.width,
          height: preset.height,
          intent: "manual",
        });
        expect(
          ladder.sizes.some(
            (size) =>
              size.width === preset.width && size.height === preset.height,
          ),
          `${entry.name}: ${preset.width}x${preset.height} missing from its own ladder`,
        ).toBe(true);
      }
    }
  });

  it("only ever resolves a size the recipe admits", () => {
    for (const entry of recipes) {
      const model: OutputShapeModel = {
        name: entry.name,
        family: entry.family,
        default_width: entry.recipe.defaults.width,
        default_height: entry.recipe.defaults.height,
        generation_profile: entry.profile,
      };
      const input: OutputShapeInput = {
        model,
        pipeline: entry.pipeline,
        width: entry.recipe.defaults.width,
        height: entry.recipe.defaults.height,
        intent: "model-default",
      };
      const result = resolveOutputShape(input);
      for (const family of result.families) {
        const size = sizeForFamily(family.id, { ...input, intent: "manual" });
        if (!size) continue;
        expect(
          resolutionProfileError(
            size.width,
            size.height,
            entry.recipe.resolution,
          ),
          `${entry.name}: ${family.id} resolved an inadmissible ${size.width}x${size.height}`,
        ).toBeNull();
      }
    }
  });
});

describe("a canvasless mesh recipe", () => {
  it("offers no shapes or sizes and says so", () => {
    const model = modelFor("hunyuan3d-mini-turbo:fp16");
    const result = resolveOutputShape({
      model,
      pipeline: null,
      width: 0,
      height: 0,
      intent: "model-default",
    });
    expect(result.canvasless).toBe(true);
    expect(result.families).toEqual([]);
    expect(result.sizes).toEqual([]);
    expect(result.selectedFamilyId).toBe("");
    expect(result.warnings).toEqual([]);
    expect(result.status).toContain("3-D");
  });

  it("stays canvasless even when a source is attached", () => {
    const model = modelFor("hunyuan3d-mini-turbo:fp16");
    const result = resolveOutputShape({
      model,
      pipeline: null,
      width: 0,
      height: 0,
      intent: "source",
      source: { width: 1024, height: 768 },
    });
    expect(result.canvasless).toBe(true);
    expect(result.families).toEqual([]);
    expect(result.sizes).toEqual([]);
  });

  it("is false for every raster recipe", () => {
    expect(shape().result.canvasless).toBe(false);
  });

  /**
   * A client that cannot resolve the model row — Create aimed at a machine
   * that must download the checkpoint first, or a host advertising no profile
   * at all — still knows the family. Reading the missing recipe as "has a
   * canvas" is what bound a Shape and Resolution pair to a 3-D print's own
   * 0 × 0 and rendered it as `NaN×NaN px`.
   */
  it("stays canvasless on the family rule when no recipe is in hand", () => {
    const result = resolveOutputShape({
      model: null,
      family: "hunyuan3d",
      pipeline: null,
      width: 0,
      height: 0,
      intent: "model-default",
    });
    expect(result.canvasless).toBe(true);
    expect(result.families).toEqual([]);
    expect(result.sizes).toEqual([]);
    expect(result.status).toContain("3-D");
  });

  it("keeps a raster family's canvas when no recipe is in hand", () => {
    const result = resolveOutputShape({
      model: null,
      family: "sdxl",
      pipeline: null,
      width: 1024,
      height: 1024,
      intent: "model-default",
    });
    expect(result.canvasless).toBe(false);
  });
});
