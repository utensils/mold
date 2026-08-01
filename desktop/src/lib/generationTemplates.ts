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

/** Storage key — MUST differ from the web SPA's `mold.generation.templates.v1`. */
export const GENERATION_TEMPLATES_STORAGE_KEY = "mold.desktop.generation.templates.v1";

export type GenerationTemplateSort = "updated-desc" | "updated-asc" | "name-asc" | "name-desc";

/** Human-facing labels for media present when the template was saved. */
export type GenerationTemplateMediaField =
  "source" | "mask" | "control" | "sourceVideo" | "keyframes" | "editImages" | "audioFile";

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
  editImages: "edit pictures",
  audioFile: "conditioning audio",
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
  return template.mediaReferences.filter(
    (reference) =>
      !((reference === "source" && hasSource) || (reference === "editImages" && hasAttachments)),
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
  if (form.audioFile) refs.push("audioFile");
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
  clone.audioFile = null;
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

  const restoredSource = Boolean(source) && !missing.some((asset) => asset.field === "sourceImage");
  const expectedAttachments = assets.filter((asset) => asset.field === "imageAttachments").length;
  const restoredAttachments =
    expectedAttachments > 0 &&
    attachments.length === expectedAttachments &&
    !missing.some((asset) => asset.field === "imageAttachments");
  return {
    form,
    missingMediaReferences: template.mediaReferences.filter(
      (reference) =>
        !(
          (reference === "source" && restoredSource) ||
          (reference === "editImages" && restoredAttachments)
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
