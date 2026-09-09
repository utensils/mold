/**
 * Which installed styles can make something from a prompt.
 *
 * A machine's `/api/models` lists everything it holds, including things that
 * are not styles at all: the prompt-expansion LLM, upscalers, ControlNets,
 * companion encoders. Every picker that offers "pick a style" has to exclude
 * them, and this is the ONE place that decides — the desktop store and the web
 * filter each had their own list, and the two had already drifted apart
 * (desktop knew `real-esrgan`, web knew `companion` and `control-net`).
 *
 * The 3-D Studio's Picture style picker is what surfaced it: it partitioned by
 * output kind, correctly excluding clip and mesh styles, and then offered
 * "Qwen3-1.7B — prompt expansion LLM" and "Real-ESRGAN x4+ — upscaler" as
 * things to draw a picture with.
 */

/** Families that are never a style. The union of what each surface knew. */
export const NON_GENERATION_FAMILIES: ReadonlySet<string> = new Set([
  "real-esrgan",
  "upscaler",
  "qwen3-expand",
  "companion",
  "controlnet",
  "control-net",
]);

/**
 * Names that give a support artifact away when its family does not.
 *
 * A catalog install can land an encoder or an upscaler under a family this
 * build has never heard of, so the family test alone is not enough.
 */
export const SUPPORT_NAME_PATTERNS: readonly RegExp[] = [
  /\bupscal(e|er|ing)\b/i,
  /\breal[-_ ]?esrgan\b/i,
  /\bcontrol[-_ ]?net\b/i,
  /\b(qwen3|prompt)[-_ ]?expand/i,
  /\bclip[-_ ]?[lg]?\b/i,
  /\btext[-_ ]?encoder\b/i,
  /\btokenizer\b/i,
  /\bvae\b/i,
];

/** Whether a family can make something from a prompt. */
export function isGenerationFamily(family: string | null | undefined): boolean {
  return !NON_GENERATION_FAMILIES.has((family ?? "").trim().toLowerCase());
}

/** The least a row needs for this module to answer. */
export interface GenerationModelLike {
  family: string;
  name?: string | null;
  display_name?: string | null;
  description?: string | null;
  hf_repo?: string | null;
}

/**
 * Whether a row is a style a person can pick, rather than a support artifact.
 *
 * The family answers first; the name is the fallback for a catalog install
 * whose family this build does not know.
 */
export function isGenerationModel(model: GenerationModelLike): boolean {
  if (!isGenerationFamily(model.family)) return false;
  const searchable = [
    model.name,
    model.display_name,
    model.description,
    model.hf_repo,
  ]
    .filter((value): value is string => typeof value === "string")
    .join(" ")
    .toLowerCase();
  return !SUPPORT_NAME_PATTERNS.some((pattern) => pattern.test(searchable));
}
