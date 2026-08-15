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
 * The wire value is never inverted anywhere — only the label and help
 * text change. This helper is the single label policy for web, desktop,
 * and iPhone; components must not restate it.
 */
export interface StrengthSemantics {
  label: string;
  hint: string;
  /** True when a larger value preserves MORE of the source. */
  higherMeansSource: boolean;
}

const LTX2_FAMILIES = new Set(["ltx2", "ltx-2"]);

export function strengthSemantics(family: string): StrengthSemantics {
  // `ltx-video` (the 0.9.x family) is deliberately NOT relabelled here
  // until its engine semantics are audited; it keeps the SD wording.
  if (LTX2_FAMILIES.has(family.trim().toLowerCase())) {
    return {
      label: "Source strength",
      hint: "Higher keeps more of the source; 1.0 pins the opening frame.",
      higherMeansSource: true,
    };
  }
  return {
    label: "Denoise strength",
    hint: "Higher allows more change from the source.",
    higherMeansSource: false,
  };
}

/**
 * Model-aware variant for saved prints, where only the raw model id (and
 * possibly an inventory-resolved family) is available. The family wins
 * when known; otherwise the model id itself is sniffed for the LTX-2
 * name markers (`ltx-2*`, `ltx2.*` — deliberately NOT `ltx-video`).
 * Catalog `cv:`/`hf:` ids without an inventory hit keep the SD wording —
 * a wrong "denoise" caption understates, never inverts, an unknown model.
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
