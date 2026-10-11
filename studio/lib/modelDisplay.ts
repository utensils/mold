/**
 * How a model is NAMED to a person — the one TypeScript reader of
 * `mold_core::model_presentation`.
 *
 * A curated manifest id presents as a short `title` (the model without
 * precision), a `variant` (what tells siblings apart), a one-line `summary`
 * of why to pick it, and a `familyLabel`. `/api/models` rows carry those four
 * fields (additive); the generated `MODEL_PRESENTATIONS` table names an id no
 * connected machine reports (queues, history, prints) or a row from an older
 * server. Catalog installs (`cv:` / `hf:`) are never renamed: their provider
 * title is `display_name`, else the catalog description. No surface keeps a
 * name or family table of its own (`modelPresentationAuthority.test.ts`).
 */
import type { ModelPresentationFields } from "./api/serverTypes";
import { MODEL_PRESENTATIONS } from "./generated/modelPresentationV1";
import { familyLabel } from "./modelFamily";

export interface DisplayableModel extends ModelPresentationFields {
  name: string;
  description?: string | null;
  family?: string | null;
}

export interface ModelPresentation {
  /** The plain name, without precision. For an id nothing names, the
   *  family's label stands in when the row carries one — the id already
   *  rides beside it in mono wherever a title is drawn. */
  title: string;
  /** What tells siblings apart (`Q4`, `Turbo 4-step 768p`). */
  variant: string | null;
  /** One line on why you would pick it. */
  summary: string | null;
  familyLabel: string | null;
  /** The standalone name — `title variant` — for any place that names a
   *  model without its id beside it. Falls back to the raw id, never to a
   *  family label, so a queue row stays identifiable. */
  displayName: string;
}

/** True for opaque catalog-install identifiers (`cv:<id>` / `hf:<repo>`). */
export function isCatalogModelId(name: string): boolean {
  return name.startsWith("cv:") || name.startsWith("hf:");
}

/**
 * True when the id says nothing a person could recognise — a Civitai
 * install is a bare number. An `hf:owner/repo` id names the repository and a
 * manifest id names the checkpoint, so both stay beside the title in mono.
 */
export function isOpaqueModelId(name: string): boolean {
  return name.startsWith("cv:");
}

function text(value: string | null | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

function joinName(title: string, variant: string | null): string {
  return variant ? `${title} ${variant}` : title;
}

function catalogFallbackTitle(name: string): string {
  if (name.startsWith("cv:")) return `Civitai model #${name.slice(3)}`;
  const repo = name.slice(3).split("/").pop() ?? name.slice(3);
  return repo.replaceAll("-", " ").replaceAll("_", " ");
}

/**
 * Precedence: the row's own server fields → the generated table → (catalog
 * ids) the provider title, then the catalog description, then a readable
 * form of the id → an older server's `display_name` → the raw id.
 */
export function modelPresentation(
  modelOrId: DisplayableModel | string,
): ModelPresentation {
  const model: DisplayableModel =
    typeof modelOrId === "string" ? { name: modelOrId } : modelOrId;
  const name = model.name;
  const familyName = text(model.family) ? familyLabel(model.family!) : null;

  const serverTitle = text(model.title);
  if (serverTitle) {
    const variant = text(model.variant);
    return {
      title: serverTitle,
      variant,
      summary: text(model.summary),
      familyLabel: text(model.family_label) ?? familyName,
      displayName: joinName(serverTitle, variant),
    };
  }

  const generated =
    MODEL_PRESENTATIONS[name] ?? MODEL_PRESENTATIONS[name.toLowerCase()];
  if (generated) {
    const variant = generated.variant ?? null;
    return {
      title: generated.title,
      variant,
      summary: generated.summary ?? null,
      familyLabel: generated.family_label ?? familyName,
      displayName: joinName(generated.title, variant),
    };
  }

  const displayName = text(model.display_name);
  const description = text(model.description);
  if (isCatalogModelId(name)) {
    const title = displayName ?? description ?? catalogFallbackTitle(name);
    return {
      title,
      variant: null,
      summary: description && description !== title ? description : null,
      familyLabel: familyName,
      displayName: title,
    };
  }

  // Nothing curated names this id: its own description is the best reason.
  const summary = description && description !== name ? description : null;
  if (displayName) {
    return {
      title: displayName,
      variant: null,
      summary: summary !== displayName ? summary : null,
      familyLabel: familyName,
      displayName,
    };
  }

  return {
    title: familyName ?? name,
    variant: null,
    summary,
    familyLabel: familyName,
    displayName: name,
  };
}

/** The standalone name for a model row (title + variant). */
export function modelDisplayName(model: DisplayableModel): string {
  return modelPresentation(model).displayName;
}

/** Resolve a model id carried by queue/status/download/history wire records. */
export function modelDisplayNameForId(
  name: string,
  models: readonly DisplayableModel[],
): string {
  return modelDisplayName(
    models.find((model) => model.name === name) ?? { name },
  );
}

/**
 * The name drawn BESIDE the mono id (chips, menu rows, shelf rows): title and
 * variant, with the family's label standing in for an id nothing names, so
 * the id is never said twice.
 */
export function styleName(model: DisplayableModel): string {
  const presented = modelPresentation(model);
  return joinName(presented.title, presented.variant);
}

/** Native option text: the friendly name with the exact runnable id beside it. */
export function styleLabel(model: DisplayableModel): string {
  const friendly = styleName(model);
  return friendly === model.name ? model.name : `${friendly} · ${model.name}`;
}
