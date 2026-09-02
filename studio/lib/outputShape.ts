/**
 * The one output-canvas resolver every Create surface renders from.
 *
 * The server's `generation_profile` stays the only data authority: alignment,
 * bounds, ceilings and the authored preset ladder all come from it. What this
 * module owns is the *presentation contract* — which canonical shape family a
 * size belongs to, which sizes that family offers, what state the canvas is
 * in, and the single sentence describing it. Web, desktop and iPhone read
 * exactly this object, so the header chip, the shape chips, the size pills and
 * the status line cannot disagree with each other (#1166).
 *
 * Shape families are canonical, not gcd-reduced preset groups: LTX-2's
 * 1216x704 (19:11), 1920x1088 (30:17) and 3840x2112 (20:11) are all
 * *widescreen*, and rendering them as three separate chips gave every model a
 * different, model-specific shape vocabulary. A family chip never picks a size
 * by ratio math — it selects from the authored ladder, whose pills show exact
 * pixels — so #998's "a 16:9 click must not silently land on 19:11" still
 * holds: the pixels are on screen before the click.
 */
import {
  effectiveGenerationRecipe,
  recipeIsCanvasless,
  resolutionProfileFinding,
  type GenerationProfileModel,
  type GenerationRecipeProfile,
  type ResolutionFinding,
} from "./generationProfile";
import {
  megapixelLabel,
  presetsForModel,
  rawAspectRatioLabel,
  type ModelResolutionContract,
} from "./resolutions";
import {
  resolveDefaultSourceResolution,
  resolveSourceResolution,
  type SourceDimensions,
  type SourceResolutionResult,
} from "./sourceResolution";
import { isMeshFamily } from "./legacyRecipeRules";

/** A model contract as both the profile and the resolution helpers read it. */
export type OutputShapeModel = GenerationProfileModel & ModelResolutionContract;

/**
 * Why the canvas currently holds the size it holds.
 *
 * This replaces the `previousAutomatic` heuristic the surfaces used to reason
 * with. Picking a model writes the model's defaults into the form *before* any
 * source watcher runs, so "is the canvas still the value I last computed"
 * cannot survive a model switch — the intent has to be recorded when the user
 * acts, not inferred afterwards (#1166 bug 1).
 *
 * - `source` — follow the attached source, on the model's own preset ladder.
 * - `source-exact` — follow the source at its own aligned, capped size.
 * - `model-default` — no source authority; the model's default canvas.
 * - `manual` — the user chose this canvas; nothing may move it.
 */
export type CanvasIntent =
  "source" | "source-exact" | "model-default" | "manual";

/** True while the canvas is still following an attached source. */
export function intentFollowsSource(intent: CanvasIntent): boolean {
  return intent === "source" || intent === "source-exact";
}

export interface OutputShapeFamily {
  /** `1:1`, `16:9`, `source`, `custom`, or an off-tolerance reduced ratio. */
  id: string;
  label: string;
  ratio: number;
}

export interface OutputShapeSize {
  width: number;
  height: number;
  /** Stable selection key for a segmented control. */
  id: string;
  /** `1216×704` — the primary label. Pixels, never a position in a list. */
  label: string;
  /** `0.9 MP` — secondary. */
  megapixels: string;
  /** The profile's own tier string, passed through untouched. */
  tier: string;
  /** Data-derived mark (`Default`, `Source`, `Custom`) or null. */
  mark: string | null;
}

export type OutputShapeState =
  "matches-source" | "follows-source" | "model-default" | "manual";

export interface SourceTreatment {
  kind: "exact" | "scaled" | "cropped" | "letterboxed";
  /** Percentage of the source frame lost (cropped) or padded (letterboxed). */
  percent?: number;
}

export interface OutputShape {
  width: number;
  height: number;
  /** The canonical family of the CURRENT canvas. */
  family: OutputShapeFamily;
  /** The chips to render, in canonical order, plus Source when attached. */
  families: OutputShapeFamily[];
  /** The chip that should light up; "" when nothing matches. */
  selectedFamilyId: string;
  /** The lit chip is the nearest match to an off-ladder size. */
  approximate: boolean;
  /** The active family's size ladder, smallest first. */
  sizes: OutputShapeSize[];
  /** The selected size's id, for a segmented control. */
  selectedSizeId: string;
  state: OutputShapeState;
  /** Short state badge — `Source`, `Manual`, … — never disagrees with `state`. */
  badge: string;
  /** The ONE sentence every surface renders under the size row. */
  status: string;
  sourceTreatment: SourceTreatment | null;
  warnings: ResolutionFinding[];
  /**
   * The recipe renders with no pixel canvas at all (a 3-D mesh): there are
   * no families, no sizes, and every surface hides the Shape and Resolution
   * controls instead of rendering a canvas the request ignores.
   */
  canvasless: boolean;
}

export interface OutputShapeInput {
  model?: OutputShapeModel | null;
  /** Family fallback for a host that advertises no model contract. */
  family?: string | null;
  pipeline?: string | null;
  width: number;
  height: number;
  source?: SourceDimensions | null;
  intent: CanvasIntent;
  /**
   * How a source that does not match the canvas maps onto it. `cover` (the
   * default) crops; `contain` letterboxes.
   */
  sourceFit?: "cover" | "contain";
}

/**
 * The canonical shape vocabulary, in render order. Portrait twins are their
 * own entries so assignment stays orientation-symmetric.
 */
const CANONICAL_FAMILIES: readonly OutputShapeFamily[] = [
  { id: "1:1", label: "1:1", ratio: 1 },
  { id: "5:4", label: "5:4", ratio: 5 / 4 },
  { id: "4:5", label: "4:5", ratio: 4 / 5 },
  { id: "4:3", label: "4:3", ratio: 4 / 3 },
  { id: "3:4", label: "3:4", ratio: 3 / 4 },
  { id: "3:2", label: "3:2", ratio: 3 / 2 },
  { id: "2:3", label: "2:3", ratio: 2 / 3 },
  { id: "16:9", label: "16:9", ratio: 16 / 9 },
  { id: "9:16", label: "9:16", ratio: 9 / 16 },
  { id: "21:9", label: "21:9", ratio: 21 / 9 },
  { id: "9:21", label: "9:21", ratio: 9 / 21 },
];

/**
 * Log-ratio radius around a canonical family. 0.06 is wide enough to collect
 * every authored preset the shipped profiles express as a near miss (19:11,
 * 30:17, 20:11, 26:15 and 7:4 are all 16:9; 22:15 and 19:13 are 3:2; 9:7 is
 * 5:4) and narrow enough that 3:2 and 4:3 stay distinct shapes.
 */
export const FAMILY_LOG_TOLERANCE = 0.06;

export const SOURCE_FAMILY_ID = "source";
export const CUSTOM_FAMILY_ID = "custom";

function ratioOf(width: number, height: number): number {
  return width > 0 && height > 0 ? width / height : 1;
}

function reducedLabel(width: number, height: number): string {
  const label = rawAspectRatioLabel(width, height);
  const [left = "1", right = "1"] = label.split(":");
  // A reduced pair such as 1234:567 is noise, not a shape name.
  return Number(left) > 99 || Number(right) > 99
    ? `${ratioOf(width, height).toFixed(2)}:1`
    : label;
}

/**
 * The canonical family a size belongs to. A size outside every canonical
 * tolerance keeps its own reduced ratio as a family of its own, so no
 * authored preset is ever hidden from the shape row.
 */
export function canonicalFamilyFor(
  width: number,
  height: number,
): OutputShapeFamily {
  const ratio = ratioOf(width, height);
  let best: OutputShapeFamily | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const family of CANONICAL_FAMILIES) {
    const distance = Math.abs(Math.log(ratio / family.ratio));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = family;
    }
  }
  if (best && bestDistance <= FAMILY_LOG_TOLERANCE) return best;
  const label = reducedLabel(width, height);
  return { id: label, label, ratio };
}

/** `16:9` for any canvas — the header chip's shape name. */
export function outputFamilyLabel(width: number, height: number): string {
  return canonicalFamilyFor(width, height).label;
}

interface AssignedPreset {
  width: number;
  height: number;
  familyId: string;
  tier: string;
}

function assignPresets(
  contract: OutputShapeModel | string,
  pipeline: string | null | undefined,
  recipe: GenerationRecipeProfile | null,
): AssignedPreset[] {
  const tiers = new Map<string, string>();
  for (const group of recipe?.resolution.aspect_groups ?? []) {
    for (const preset of group.presets) {
      tiers.set(`${preset.width}x${preset.height}`, preset.tier);
    }
  }
  return presetsForModel(contract, pipeline).map(({ width, height }) => ({
    width,
    height,
    familyId: canonicalFamilyFor(width, height).id,
    tier: tiers.get(`${width}x${height}`) ?? "recommended",
  }));
}

function markFor(
  width: number,
  height: number,
  tier: string,
  recipe: GenerationRecipeProfile | null,
): string | null {
  if (
    recipe &&
    recipe.defaults.width === width &&
    recipe.defaults.height === height
  ) {
    return "Default";
  }
  // Every shipped profile authors a single "recommended" tier today, so
  // marking all of them says nothing. Anything else is real data, and shows.
  if (!tier || tier === "recommended") return null;
  return tier.charAt(0).toUpperCase() + tier.slice(1);
}

function toSize(
  preset: AssignedPreset,
  recipe: GenerationRecipeProfile | null,
): OutputShapeSize {
  return {
    width: preset.width,
    height: preset.height,
    id: `${preset.width}x${preset.height}`,
    label: `${preset.width}×${preset.height}`,
    megapixels: megapixelLabel(preset.width, preset.height),
    tier: preset.tier,
    mark: markFor(preset.width, preset.height, preset.tier, recipe),
  };
}

function ladderFor(
  presets: readonly AssignedPreset[],
  familyId: string,
  recipe: GenerationRecipeProfile | null,
): OutputShapeSize[] {
  return presets
    .filter((preset) => preset.familyId === familyId)
    .sort(
      (left, right) => left.width * left.height - right.width * right.height,
    )
    .map((preset) => toSize(preset, recipe));
}

function insertSize(
  sizes: OutputShapeSize[],
  dimensions: SourceDimensions,
  mark: string,
  recipe: GenerationRecipeProfile | null,
): void {
  if (
    sizes.some(
      (size) =>
        size.width === dimensions.width && size.height === dimensions.height,
    )
  ) {
    return;
  }
  const entry: OutputShapeSize = {
    ...toSize(
      {
        width: dimensions.width,
        height: dimensions.height,
        familyId: "",
        tier: "custom",
      },
      recipe,
    ),
    mark,
  };
  const area = entry.width * entry.height;
  const index = sizes.findIndex((size) => size.width * size.height > area);
  if (index < 0) sizes.push(entry);
  else sizes.splice(index, 0, entry);
}

function treatmentFor(
  source: SourceDimensions | null,
  width: number,
  height: number,
  fit: "cover" | "contain",
): SourceTreatment | null {
  if (!source) return null;
  if (source.width === width && source.height === height) {
    return { kind: "exact" };
  }
  const sourceRatio = ratioOf(source.width, source.height);
  const canvasRatio = ratioOf(width, height);
  const overlap = Math.min(
    sourceRatio / canvasRatio,
    canvasRatio / sourceRatio,
  );
  if (overlap > 0.995) return { kind: "scaled" };
  const percent = Math.round((1 - overlap) * 100);
  return fit === "contain"
    ? { kind: "letterboxed", percent }
    : { kind: "cropped", percent };
}

function treatmentClause(treatment: SourceTreatment | null): string {
  if (treatment?.kind === "cropped") {
    return ` · source cropped ${treatment.percent}%`;
  }
  if (treatment?.kind === "letterboxed") {
    return ` · source letterboxed ${treatment.percent}%`;
  }
  return "";
}

function badgeFor(state: OutputShapeState): string {
  if (state === "matches-source" || state === "follows-source") return "Source";
  if (state === "model-default") return "Default";
  return "Manual";
}

function statusFor(
  state: OutputShapeState,
  width: number,
  height: number,
  source: SourceDimensions | null,
  sourceExact: SourceResolutionResult | null,
  treatment: SourceTreatment | null,
): string {
  const canvas = `${width}×${height}`;
  if (state === "matches-source") return `${canvas} · Matches source`;
  if (state === "follows-source" && source) {
    const grid = sourceExact ? `, ${sourceExact.alignment} px grid` : "";
    return `${canvas} · Follows source (${source.width}×${source.height}${grid})`;
  }
  const base =
    state === "model-default"
      ? `${canvas} · Model default`
      : `${canvas} · Manual`;
  return `${base}${treatmentClause(treatment)}`;
}

function nearestFamilyId(
  families: readonly OutputShapeFamily[],
  ratio: number,
): string {
  const candidates = families.filter(
    (family) => family.id !== SOURCE_FAMILY_ID,
  );
  const nearest = [...candidates].sort(
    (left, right) =>
      Math.abs(Math.log(ratio / left.ratio)) -
      Math.abs(Math.log(ratio / right.ratio)),
  )[0];
  return nearest?.id ?? "";
}

/**
 * Resolve the complete output-canvas presentation for one recipe, source and
 * intent. Pure: it reads the form, it never writes it.
 */
export function resolveOutputShape(input: OutputShapeInput): OutputShape {
  const { model, pipeline, width, height, intent } = input;
  const contract: OutputShapeModel | string = model ?? input.family ?? "";
  const recipe = model ? effectiveGenerationRecipe(model, pipeline) : null;
  // The recipe answers when there is one; the pre-profile family rule answers
  // for a model row this client could not resolve (aimed at a machine that
  // must download the checkpoint first) and for a host that advertises no
  // profile at all. Either way a 3-D print has no canvas to pick.
  if (recipe ? recipeIsCanvasless(recipe) : isMeshFamily(input.family))
    return canvaslessShape(width, height);
  const source = input.source ?? null;
  const sourceExact = source
    ? resolveSourceResolution(source, contract, pipeline)
    : null;
  const sourceLadderPick = source
    ? resolveDefaultSourceResolution(source, contract, pipeline)
    : null;
  const presets = contract ? assignPresets(contract, pipeline, recipe) : [];

  const families: OutputShapeFamily[] = CANONICAL_FAMILIES.filter((family) =>
    presets.some((preset) => preset.familyId === family.id),
  ).slice();
  for (const preset of presets) {
    if (!families.some((family) => family.id === preset.familyId)) {
      families.push(canonicalFamilyFor(preset.width, preset.height));
    }
  }
  if (source) {
    families.push({
      id: SOURCE_FAMILY_ID,
      label: "Source",
      ratio: ratioOf(source.width, source.height),
    });
  }

  const matchesSource =
    source !== null && source.width === width && source.height === height;
  const followsSource =
    source !== null &&
    ((sourceExact !== null &&
      sourceExact.output.width === width &&
      sourceExact.output.height === height) ||
      (sourceLadderPick !== null &&
        sourceLadderPick.width === width &&
        sourceLadderPick.height === height));
  const sourceActive =
    source !== null && followsSource && intentFollowsSource(intent);

  const canvasFamily = canonicalFamilyFor(width, height);
  const listed = families.some((family) => family.id === canvasFamily.id);
  const onLadder = presets.some(
    (preset) => preset.width === width && preset.height === height,
  );
  const family: OutputShapeFamily = sourceActive
    ? { id: SOURCE_FAMILY_ID, label: "Source", ratio: ratioOf(width, height) }
    : listed || onLadder
      ? canvasFamily
      : {
          id: CUSTOM_FAMILY_ID,
          label: canvasFamily.label,
          ratio: canvasFamily.ratio,
        };
  const selectedFamilyId = sourceActive
    ? SOURCE_FAMILY_ID
    : listed
      ? canvasFamily.id
      : nearestFamilyId(families, ratioOf(width, height));
  const approximate =
    selectedFamilyId !== SOURCE_FAMILY_ID &&
    selectedFamilyId !== "" &&
    !onLadder;

  const ladderFamilyId = sourceActive
    ? canvasFamily.id
    : family.id === CUSTOM_FAMILY_ID
      ? canvasFamily.id
      : family.id;
  const sizes = ladderFor(presets, ladderFamilyId, recipe);
  if (sourceExact && sourceActive) {
    insertSize(sizes, sourceExact.output, "Source", recipe);
  }
  if (!sizes.some((size) => size.width === width && size.height === height)) {
    insertSize(
      sizes,
      { width, height },
      followsSource ? "Source" : "Custom",
      recipe,
    );
  }

  const state: OutputShapeState = matchesSource
    ? "matches-source"
    : followsSource
      ? "follows-source"
      : recipe &&
          recipe.defaults.width === width &&
          recipe.defaults.height === height
        ? "model-default"
        : "manual";
  const treatment = treatmentFor(
    source,
    width,
    height,
    input.sourceFit ?? "cover",
  );
  const finding = resolutionProfileFinding(
    width,
    height,
    recipe?.resolution ?? null,
  );

  return {
    width,
    height,
    family,
    families,
    selectedFamilyId,
    approximate,
    sizes,
    selectedSizeId: `${width}x${height}`,
    state,
    badge: badgeFor(state),
    status: statusFor(state, width, height, source, sourceExact, treatment),
    sourceTreatment: treatment,
    warnings: finding ? [finding] : [],
    canvasless: false,
  };
}

const CANVASLESS_FAMILY: OutputShapeFamily = { id: "", label: "3-D", ratio: 1 };

/** The shape of a recipe that has no canvas: nothing to pick, one sentence. */
function canvaslessShape(width: number, height: number): OutputShape {
  return {
    width,
    height,
    family: CANVASLESS_FAMILY,
    families: [],
    selectedFamilyId: "",
    approximate: false,
    sizes: [],
    selectedSizeId: "",
    state: "model-default",
    badge: "",
    status: "3-D mesh · no canvas",
    sourceTreatment: null,
    warnings: [],
    canvasless: true,
  };
}

/**
 * The size a family chip should land on: that family's authored ladder entry
 * closest in area to the current canvas. The chip never invents a ratio —
 * every candidate is a preset the pills already show.
 */
export function sizeForFamily(
  familyId: string,
  input: OutputShapeInput,
): SourceDimensions | null {
  const contract: OutputShapeModel | string = input.model ?? input.family ?? "";
  if (familyId === SOURCE_FAMILY_ID) {
    return input.source
      ? resolveDefaultSourceResolution(input.source, contract, input.pipeline)
      : null;
  }
  if (!contract) return null;
  const recipe = input.model
    ? effectiveGenerationRecipe(input.model, input.pipeline)
    : null;
  const ladder = ladderFor(
    assignPresets(contract, input.pipeline, recipe),
    familyId,
    recipe,
  );
  const area = Math.max(1, input.width * input.height);
  const best = [...ladder].sort(
    (left, right) =>
      Math.abs(left.width * left.height - area) -
      Math.abs(right.width * right.height - area),
  )[0];
  return best ? { width: best.width, height: best.height } : null;
}

/**
 * The intent a canvas the user just picked should record. Choosing the pill
 * that IS the source-derived size keeps the canvas following the source; any
 * other pick is the user taking the canvas over.
 */
export function intentForCanvas(
  input: OutputShapeInput,
  canvas: SourceDimensions,
): CanvasIntent {
  const source = input.source ?? null;
  if (!source) return "manual";
  const contract: OutputShapeModel | string = input.model ?? input.family ?? "";
  const exact = resolveSourceResolution(
    source,
    contract,
    input.pipeline,
  ).output;
  if (exact.width === canvas.width && exact.height === canvas.height) {
    return "source-exact";
  }
  const ladder = resolveDefaultSourceResolution(
    source,
    contract,
    input.pipeline,
  );
  return ladder.width === canvas.width && ladder.height === canvas.height
    ? "source"
    : "manual";
}

/**
 * Snap an exact width/height entry onto the recipe's grid and inside its
 * ceilings. The Advanced size field and mobile's proportional inputs route
 * through this, so a typed size is admissible by the same authority that
 * draws the pills.
 */
export function snapOutputSize(
  dimensions: SourceDimensions,
  model: OutputShapeModel | string | null | undefined,
  pipeline?: string | null,
): SourceDimensions {
  const contract = model ?? "";
  if (!contract) return dimensions;
  return resolveSourceResolution(dimensions, contract, pipeline).output;
}
