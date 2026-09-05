/**
 * Client-side generation templates for the desktop app. Templates are the
 * Generate form snapshotted to `localStorage` so a set of params can be
 * recalled with one click. Ported from the web SPA's `generationTemplates.ts`,
 * but keyed on a desktop-only storage key so the two never share blobs.
 *
 * Large media bytes never enter localStorage. Source/edit images are stored
 * durably in the shared IndexedDB media store and restored with the template;
 * unsupported auxiliary media retains a human-facing re-selection hint.
 */
import type { GenerateForm } from "./generateForm";
import { createUuid } from "@studio/lib/id";
import { emptyGuidanceOverrides } from "@studio/lib/guidanceOverrides";
import {
  deleteGenerationTemplateMedia,
  hydrateGenerationTemplateMedia,
  persistGenerationTemplateMedia,
  type GenerationTemplateMediaAsset,
  type TemplateMediaPersistence,
} from "@studio/lib/templateMediaStore";
import { redactGenerationReference } from "@studio/lib/generationReferences";
import {
  emptyMinimaxH3AuthoringState,
  stripMinimaxH3AuthoringMedia,
  type MinimaxH3BoundaryImage,
} from "@studio/lib/minimaxH3Authoring";

/** Storage key — MUST differ from the web SPA's `mold.generation.templates.v1`. */
export const GENERATION_TEMPLATES_STORAGE_KEY = "mold.desktop.generation.templates.v1";

export type GenerationTemplateSort = "updated-desc" | "updated-asc" | "name-asc" | "name-desc";

/** Human-facing labels for media present when the template was saved. */
export type GenerationTemplateMediaField =
  | "source"
  | "mask"
  | "control"
  | "sourceVideo"
  | "keyframes"
  | "endFrame"
  | "editImages"
  | "audioFile"
  | "h3FirstFrame"
  | "h3LastFrame"
  | "h3References";

export interface GenerationTemplate {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  /** The saved form with base64 media stripped to null / empty. */
  form: GenerateForm;
  /** Which media slots were populated when saved. */
  mediaReferences: GenerationTemplateMediaField[];
  /** Durable client-local source snapshots. Legacy templates omit this. */
  mediaAssets?: GenerationTemplateMediaAsset[];
  /** Optional host scope for remote-only clients. Legacy/desktop templates omit it. */
  scopeId?: string;
}

const MEDIA_REFERENCE_LABELS: Record<GenerationTemplateMediaField, string> = {
  source: "source photo",
  mask: "mask",
  control: "control photo",
  sourceVideo: "source video",
  keyframes: "keyframes",
  endFrame: "end frame",
  editImages: "edit pictures",
  audioFile: "conditioning audio",
  h3FirstFrame: "H3 first frame",
  h3LastFrame: "H3 last frame",
  h3References: "H3 ordered references",
};

export function formatTemplateMediaReferences(
  references: readonly GenerationTemplateMediaField[],
): string {
  return references.map((reference) => MEDIA_REFERENCE_LABELS[reference]).join(", ");
}

/** Media that definitely has no durable snapshot and must be re-selected. */
export function unsnapshottedTemplateMediaReferences(
  template: GenerationTemplate,
): GenerationTemplateMediaField[] {
  const assets = template.mediaAssets ?? [];
  const hasSource = assets.some((asset) => asset.field === "sourceImage");
  const hasAttachments = assets.some((asset) => asset.field === "imageAttachments");
  const hasH3FirstFrame = assets.some((asset) => asset.field === "h3FirstFrame");
  const hasH3LastFrame = assets.some((asset) => asset.field === "h3LastFrame");
  const hasH3References =
    assets.filter((asset) => asset.field === "h3References").length ===
    (template.form.h3Authoring?.references.length ?? 0);
  return template.mediaReferences.filter(
    (reference) =>
      !(
        (reference === "source" && hasSource) ||
        (reference === "editImages" && hasAttachments) ||
        (reference === "h3FirstFrame" && hasH3FirstFrame) ||
        (reference === "h3LastFrame" && hasH3LastFrame) ||
        (reference === "h3References" && hasH3References)
      ),
  );
}

function now(): number {
  return Date.now();
}

function templateId(): string {
  return createUuid();
}

function normalizeName(name: string): string {
  return name.trim() || "Untitled template";
}

/**
 * The name used when the user saves without typing one: a prompt slice, else
 * the model name, else a generic label. Mirrors the web panel's fallback.
 */
export function fallbackTemplateName(form: GenerateForm): string {
  return form.prompt.trim().slice(0, 48) || form.model || "Untitled template";
}

export function collectTemplateMediaReferences(form: GenerateForm): GenerationTemplateMediaField[] {
  const refs: GenerationTemplateMediaField[] = [];
  if (form.sourceImage) refs.push("source");
  if (form.maskImage) refs.push("mask");
  if (form.controlImage) refs.push("control");
  if (form.imageAttachments.length > 0) refs.push("editImages");
  if (form.sourceVideo) refs.push("sourceVideo");
  if (form.keyframes.length > 0) refs.push("keyframes");
  if (form.endFrame) refs.push("endFrame");
  if (form.audioFile) refs.push("audioFile");
  if (form.h3Authoring?.firstFrame) refs.push("h3FirstFrame");
  if (form.h3Authoring?.lastFrame) refs.push("h3LastFrame");
  if (form.h3Authoring?.references.length) refs.push("h3References");
  return refs;
}

/** Deep-clone the form and null out every base64 media field before persisting. */
function stripTemplateForm(form: GenerateForm): GenerateForm {
  const clone = JSON.parse(JSON.stringify(form)) as GenerateForm;
  clone.sourceImage = null;
  clone.sourceImageName = null;
  clone.maskImage = null;
  clone.controlImage = null;
  clone.imageAttachments = [];
  clone.sourceVideo = null;
  clone.keyframes = [];
  // The wan end frame has no durable asset of its own, so it is recorded as a
  // media reference and reported missing on hydrate rather than persisted.
  clone.endFrame = null;
  clone.audioFile = null;
  clone.h3Authoring = stripMinimaxH3AuthoringMedia(clone.h3Authoring);
  return clone;
}

function isTemplate(value: unknown): value is GenerationTemplate {
  if (!value || typeof value !== "object") return false;
  const c = value as Partial<GenerationTemplate>;
  return (
    typeof c.id === "string" &&
    typeof c.name === "string" &&
    typeof c.createdAt === "number" &&
    typeof c.updatedAt === "number" &&
    !!c.form &&
    Array.isArray(c.mediaReferences)
  );
}

function parseTemplates(raw: string | null): GenerationTemplate[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isTemplate).map((template) => ({
      ...template,
      form: {
        ...template.form,
        guidanceOverrides: template.form.guidanceOverrides ?? emptyGuidanceOverrides(),
        // A template saved before "Save every result" existed carries no
        // preference, and the form's default is to save. Hydrating it
        // `undefined` would leave `buildRequest` reading a missing value.
        saveResult: template.form.saveResult ?? true,
      },
    }));
  } catch {
    return [];
  }
}

function writeTemplates(
  templates: GenerationTemplate[],
  storageKey = GENERATION_TEMPLATES_STORAGE_KEY,
): void {
  localStorage.setItem(storageKey, JSON.stringify(templates));
}

export function sortGenerationTemplates(
  templates: GenerationTemplate[],
  sort: GenerationTemplateSort = "updated-desc",
): GenerationTemplate[] {
  const sorted = templates.slice();
  sorted.sort((a, b) => {
    switch (sort) {
      case "updated-asc":
        return a.updatedAt - b.updatedAt;
      case "name-asc":
        return a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
      case "name-desc":
        return b.name.localeCompare(a.name, undefined, { sensitivity: "base" });
      case "updated-desc":
      default:
        return b.updatedAt - a.updatedAt;
    }
  });
  return sorted;
}

export function loadGenerationTemplates(
  sort: GenerationTemplateSort = "updated-desc",
  storageKey = GENERATION_TEMPLATES_STORAGE_KEY,
): GenerationTemplate[] {
  return sortGenerationTemplates(parseTemplates(localStorage.getItem(storageKey)), sort);
}

export function saveGenerationTemplate(
  name: string,
  form: GenerateForm,
  storageKey = GENERATION_TEMPLATES_STORAGE_KEY,
  scopeId?: string,
): GenerationTemplate {
  const timestamp = now();
  const template: GenerationTemplate = {
    id: templateId(),
    name: normalizeName(name || fallbackTemplateName(form)),
    createdAt: timestamp,
    updatedAt: timestamp,
    form: stripTemplateForm(form),
    mediaReferences: collectTemplateMediaReferences(form),
    ...(scopeId ? { scopeId } : {}),
  };
  writeTemplates([template, ...loadGenerationTemplates("updated-desc", storageKey)], storageKey);
  return template;
}

/** Save the template only after its source bytes are durable in IndexedDB. */
export async function saveGenerationTemplateWithMedia(
  name: string,
  form: GenerateForm,
  storageKey = GENERATION_TEMPLATES_STORAGE_KEY,
  scopeId?: string,
  persistence?: TemplateMediaPersistence,
): Promise<GenerationTemplate> {
  const timestamp = now();
  const id = templateId();
  const h3 = form.h3Authoring ?? emptyMinimaxH3AuthoringState();
  const inputs = [
    ...(form.sourceImage
      ? [
          {
            field: "sourceImage" as const,
            filename: form.sourceImageName || "Source image",
            base64: form.sourceImage,
          },
        ]
      : []),
    ...form.imageAttachments.map((base64, index) => ({
      field: "imageAttachments" as const,
      index,
      filename: `Reference ${index + 1}`,
      base64,
    })),
    ...(h3.firstFrame?.data
      ? [
          {
            field: "h3FirstFrame" as const,
            filename: h3.firstFrame.filename,
            kind: "upload" as const,
            width: h3.firstFrame.width,
            height: h3.firstFrame.height,
            mime: h3.firstFrame.mimeType,
            ...(h3.firstFrame.sha256 !== undefined ? { sha256: h3.firstFrame.sha256 } : {}),
            base64: h3.firstFrame.data,
          },
        ]
      : []),
    ...(h3.lastFrame?.data
      ? [
          {
            field: "h3LastFrame" as const,
            filename: h3.lastFrame.filename,
            kind: "upload" as const,
            width: h3.lastFrame.width,
            height: h3.lastFrame.height,
            mime: h3.lastFrame.mimeType,
            ...(h3.lastFrame.sha256 !== undefined ? { sha256: h3.lastFrame.sha256 } : {}),
            base64: h3.lastFrame.data,
          },
        ]
      : []),
    ...h3.references.flatMap(({ reference }, index) =>
      reference.media.authority === "inline" && reference.media.data
        ? [
            {
              field: "h3References" as const,
              index,
              filename: reference.provenance?.name || `${reference.kind} reference ${index + 1}`,
              kind: "upload" as const,
              mime: reference.mime_type,
              reference: redactGenerationReference(reference),
              base64: reference.media.data,
            },
          ]
        : [],
    ),
  ];
  const mediaAssets = await persistGenerationTemplateMedia(id, inputs, persistence);
  const template: GenerationTemplate = {
    id,
    name: normalizeName(name || fallbackTemplateName(form)),
    createdAt: timestamp,
    updatedAt: timestamp,
    form: stripTemplateForm(form),
    mediaReferences: collectTemplateMediaReferences(form),
    ...(mediaAssets.length ? { mediaAssets } : {}),
    ...(scopeId ? { scopeId } : {}),
  };
  try {
    writeTemplates([template, ...loadGenerationTemplates("updated-desc", storageKey)], storageKey);
  } catch (error) {
    await deleteGenerationTemplateMedia(mediaAssets, persistence);
    throw error;
  }
  return template;
}

export async function hydrateGenerationTemplate(
  template: GenerationTemplate,
  persistence?: TemplateMediaPersistence,
): Promise<{
  form: GenerateForm;
  missingMediaReferences: GenerationTemplateMediaField[];
}> {
  const form = JSON.parse(JSON.stringify(template.form)) as GenerateForm;
  const assets = template.mediaAssets ?? [];
  const { media, missing } = await hydrateGenerationTemplateMedia(assets, persistence);
  const source = media.find((asset) => asset.field === "sourceImage");
  if (source) {
    form.sourceImage = source.base64;
    form.sourceImageName = source.filename;
  }
  const attachments = media
    .filter((asset) => asset.field === "imageAttachments")
    .sort((left, right) => (left.index ?? 0) - (right.index ?? 0));
  if (attachments.length) form.imageAttachments = attachments.map((asset) => asset.base64);
  form.h3Authoring ??= emptyMinimaxH3AuthoringState();
  const hydrateBoundary = (
    field: "h3FirstFrame" | "h3LastFrame",
  ): MinimaxH3BoundaryImage | null => {
    const asset = media.find((candidate) => candidate.field === field);
    if (!asset) return null;
    return {
      filename: asset.filename,
      mimeType: asset.mime || "image/png",
      width: asset.width ?? 0,
      height: asset.height ?? 0,
      data: asset.base64,
      ...(asset.sha256 !== undefined ? { sha256: asset.sha256 } : {}),
      draftId: asset.assetId,
    };
  };
  const first = hydrateBoundary("h3FirstFrame");
  const last = hydrateBoundary("h3LastFrame");
  if (first) form.h3Authoring.firstFrame = first;
  if (last) form.h3Authoring.lastFrame = last;
  const h3References = media
    .filter((asset) => asset.field === "h3References" && asset.reference)
    .sort((left, right) => (left.index ?? 0) - (right.index ?? 0));
  for (const asset of h3References) {
    const index = asset.index ?? 0;
    form.h3Authoring.references[index] = {
      reference: {
        ...asset.reference!,
        media: { authority: "inline", data: asset.base64 },
      },
      draftId: asset.assetId,
    };
  }

  const restoredSource = Boolean(source) && !missing.some((asset) => asset.field === "sourceImage");
  const expectedAttachments = assets.filter((asset) => asset.field === "imageAttachments").length;
  const restoredAttachments =
    expectedAttachments > 0 &&
    attachments.length === expectedAttachments &&
    !missing.some((asset) => asset.field === "imageAttachments");
  const restoredH3First =
    Boolean(first) && !missing.some((asset) => asset.field === "h3FirstFrame");
  const restoredH3Last = Boolean(last) && !missing.some((asset) => asset.field === "h3LastFrame");
  const restoredH3References =
    h3References.length === form.h3Authoring.references.length &&
    !missing.some((asset) => asset.field === "h3References");
  return {
    form,
    missingMediaReferences: template.mediaReferences.filter(
      (reference) =>
        !(
          (reference === "source" && restoredSource) ||
          (reference === "editImages" && restoredAttachments) ||
          (reference === "h3FirstFrame" && restoredH3First) ||
          (reference === "h3LastFrame" && restoredH3Last) ||
          (reference === "h3References" && restoredH3References)
        ),
    ),
  };
}

export function renameGenerationTemplate(
  id: string,
  name: string,
  storageKey = GENERATION_TEMPLATES_STORAGE_KEY,
): GenerationTemplate | null {
  let renamed: GenerationTemplate | null = null;
  const next = loadGenerationTemplates("updated-desc", storageKey).map((template) => {
    if (template.id !== id) return template;
    renamed = { ...template, name: normalizeName(name), updatedAt: now() };
    return renamed;
  });
  writeTemplates(next, storageKey);
  return renamed;
}

export function deleteGenerationTemplate(
  id: string,
  storageKey = GENERATION_TEMPLATES_STORAGE_KEY,
): void {
  writeTemplates(
    loadGenerationTemplates("updated-desc", storageKey).filter((template) => template.id !== id),
    storageKey,
  );
}

export async function deleteGenerationTemplateWithMedia(
  template: GenerationTemplate,
  storageKey = GENERATION_TEMPLATES_STORAGE_KEY,
  persistence?: TemplateMediaPersistence,
): Promise<void> {
  deleteGenerationTemplate(template.id, storageKey);
  await deleteGenerationTemplateMedia(template.mediaAssets ?? [], persistence);
}

export function searchGenerationTemplates(
  query: string,
  sort: GenerationTemplateSort = "updated-desc",
  storageKey = GENERATION_TEMPLATES_STORAGE_KEY,
): GenerationTemplate[] {
  const q = query.trim().toLowerCase();
  const templates = loadGenerationTemplates(sort, storageKey);
  if (!q) return templates;
  return templates.filter((template) =>
    [
      template.name,
      template.form.model,
      template.form.prompt,
      template.form.negativePrompt,
      ...template.form.loras.map((lora) => lora.path),
    ]
      .join(" ")
      .toLowerCase()
      .includes(q),
  );
}
