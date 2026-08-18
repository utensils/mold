/**
 * How a model family is named to a person.
 *
 * The wire carries a slug (`wan`, `ltx-2`, `qwen-image-edit`); every surface
 * has to turn that into the name the family is actually known by. This lived
 * only in `web/`, so desktop and iPhone rendered the raw slug — a Wan row read
 * "wan" on desktop and "Wan Video" on web, which is exactly the cross-surface
 * drift #806 exists to close. The CLI and TUI already share their own label
 * function pinned to `tests/fixtures/wan/surface-parity-v1.json`; this is the
 * Studio half of the same contract.
 *
 * Unknown slugs title-case rather than fall back to the raw string, so a family
 * added server-side reads as a name on day one without a client release.
 */

const FAMILY_LABELS: Record<string, string> = {
  flux: "FLUX",
  flux2: "Flux.2",
  "flux-2": "Flux.2",
  sd15: "SD 1.5",
  "sd1.5": "SD 1.5",
  sdxl: "SDXL",
  sd3: "SD3",
  "sd3.5": "SD3.5",
  "z-image": "Z-Image",
  "qwen-image": "Qwen Image",
  "qwen-image-edit": "Qwen Image Edit",
  wuerstchen: "Wuerstchen",
  "ltx-video": "LTX Video",
  ltx2: "LTX-2",
  "ltx-2": "LTX-2",
  wan: "Wan Video",
  "minimax-h3": "MiniMax H3",
  minimax_h3: "MiniMax H3",
  minimaxh3: "MiniMax H3",
};

export function familyLabel(family: string): string {
  return (
    FAMILY_LABELS[family] ??
    family
      .split(/[-_]/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
  );
}

/** Catalog browsing groups edit-specialized Qwen Image variants with the
 * Qwen Image shelf. Runtime forms keep the concrete `qwen-image-edit` family;
 * this only controls Models taxonomy/filtering. */
export function catalogFamily(family: string): string {
  return family === "qwen-image-edit" ? "qwen-image" : family;
}

export function matchesCatalogFamily(
  family: string,
  selected: string,
): boolean {
  return !selected || catalogFamily(family) === catalogFamily(selected);
}
