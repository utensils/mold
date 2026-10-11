/**
 * How a model family is named to a person.
 *
 * The wire carries a slug (`wan`, `ltx2`, `qwen-image-edit`); the name comes
 * from the ONE authority, `mold_core::model_presentation::family_label`, via
 * the generated `FAMILY_LABELS` (#806 closed the cross-surface drift a
 * per-client table caused). Spellings that differ only in separators
 * (`ltx-2`, `minimax_h3`, `zimage`) resolve to the canonical slug's label.
 *
 * Unknown slugs title-case rather than fall back to the raw string, so a family
 * added server-side reads as a name on day one without a client release.
 */
import { FAMILY_LABELS } from "./generated/modelPresentationV1";

const looseKey = (family: string) =>
  family
    .trim()
    .toLowerCase()
    .replace(/[-_.\s]/g, "");

const LABELS_BY_LOOSE_KEY = new Map(
  Object.entries(FAMILY_LABELS).map(([slug, label]) => [looseKey(slug), label]),
);

export function familyLabel(family: string): string {
  return (
    FAMILY_LABELS[family] ??
    LABELS_BY_LOOSE_KEY.get(looseKey(family)) ??
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
