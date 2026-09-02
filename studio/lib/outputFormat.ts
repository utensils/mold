/**
 * The one output-format coercion every build/restore site applies.
 *
 * A mesh recipe stores binary glTF and nothing else, so a raster format left
 * over from the previous model — or sent by an older client that always
 * says `png` — is PINNED to `glb` rather than refused; this is the same rule
 * `GenerateRequest::pin_output_format_for_family` applies at admission and
 * `mold run`'s `default_output_format` applies on the CLI. Everywhere else an
 * unadvertised format is a real mistake the recipe's own default corrects
 * (a `glb` lingering after switching back to SDXL would be a 422 at the door).
 */
import { isMeshFamily } from "./legacyRecipeRules";

const MESH_FORMATS: ReadonlySet<string> = new Set(["glb", "obj"]);

/** Just enough of a resolved recipe to answer the format question. */
export type OutputFormatRecipe = {
  capabilities: {
    mesh?: unknown;
    output: { formats: readonly string[]; default_format: string };
  };
};

export function isMeshOutputFormat(format: string | null | undefined): boolean {
  return format != null && MESH_FORMATS.has(format);
}

/**
 * Resolve the format a request for `recipe` may carry.
 *
 * - A mesh recipe (advertised `mesh` block, or a legacy host's mesh family)
 *   answers its own default — `glb` — for anything but an advertised mesh
 *   format.
 * - A recipe that advertises a format list answers `format` when listed and
 *   its default otherwise.
 * - Without a recipe, `legacyFormats` (the surface's own family rule) plays
 *   the same role; with neither, the format passes through unchanged.
 */
export function coerceOutputFormatForRecipe<F extends string>(
  recipe: OutputFormatRecipe | null | undefined,
  family: string | null | undefined,
  format: F | null | undefined,
  legacyFormats?: readonly string[],
): F | undefined {
  const current = format ?? undefined;
  const meshRecipe = recipe
    ? recipe.capabilities.mesh != null ||
      (recipe.capabilities.output.formats.length > 0 &&
        recipe.capabilities.output.formats.every(isMeshOutputFormat))
    : isMeshFamily(family);
  if (meshRecipe) {
    const advertised = recipe?.capabilities.output.formats ?? ["glb"];
    if (
      current &&
      isMeshOutputFormat(current) &&
      advertised.includes(current)
    ) {
      return current;
    }
    return (recipe?.capabilities.output.default_format ?? "glb") as F;
  }
  if (recipe && recipe.capabilities.output.formats.length > 0) {
    if (current && recipe.capabilities.output.formats.includes(current)) {
      return current;
    }
    return current === undefined
      ? undefined
      : (recipe.capabilities.output.default_format as F);
  }
  if (legacyFormats && legacyFormats.length > 0) {
    if (current && legacyFormats.includes(current)) return current;
    return current === undefined ? undefined : (legacyFormats[0] as F);
  }
  return current;
}
