/**
 * The user-facing meaning of the wire field `GenerateRequest.strength`,
 * which carries two OPPOSITE conventions depending on the model family
 * (#1055):
 *
 * - SD-lineage img2img (SD1.5/SDXL/SD3/FLUX/Z-Image, …) treats it as
 *   DENOISE strength: higher = more change, 1.0 = ignore the source.
 * - LTX-2 treats it as SOURCE/conditioning strength (upstream
 *   `latent_cond.py:41`: `denoise_mask = 1 - strength`): higher = more
 *   source preservation, 1.0 = pin the opening frame.
 *
 * The wire value is never inverted anywhere — only the help text and the
 * direction flag change. The label itself is the binding lexicon's one
 * phrase (docs/design/README.md §2), never "img2img" or "denoise
 * strength", on every family; `higherMeansSource` is what tells a surface
 * which end of the track keeps the photo. This helper is the single label
 * policy for web, desktop, and iPhone; components must not restate it.
 */
export interface StrengthSemantics {
  label: string;
  hint: string;
  /** True when a larger value preserves MORE of the source. */
  higherMeansSource: boolean;
}

const LTX2_FAMILIES = new Set(["ltx2", "ltx-2"]);

const LABEL = "How much to change it";

export function strengthSemantics(family: string): StrengthSemantics {
  // `ltx-video` (the 0.9.x family) is deliberately NOT relabelled here
  // until its engine semantics are audited; it keeps the SD direction.
  if (LTX2_FAMILIES.has(family.trim().toLowerCase())) {
    return {
      label: LABEL,
      hint: "Higher keeps more of the source; 1.0 pins the opening frame.",
      higherMeansSource: true,
    };
  }
  return {
    label: LABEL,
    hint: "Higher allows more change from the source.",
    higherMeansSource: false,
  };
}

/**
 * Model-aware variant for saved prints, where only the raw model id (and
 * possibly an inventory-resolved family) is available. The family wins
 * when known; otherwise the model id itself is sniffed for the LTX-2
 * name markers (`ltx-2*`, `ltx2.*` — deliberately NOT `ltx-video`).
 * Catalog `cv:`/`hf:` ids without an inventory hit keep the SD direction —
 * a wrong hint understates, never inverts, an unknown model.
 */
export function strengthSemanticsForModel(
  model: string | null | undefined,
  family?: string | null,
): StrengthSemantics {
  if (family) return strengthSemantics(family);
  const id = (model ?? "").trim().toLowerCase();
  if (id.includes("ltx-2") || id.includes("ltx2.")) {
    return strengthSemantics("ltx2");
  }
  return strengthSemantics(id);
}
