/**
 * Client-side generation templates for the desktop app. Templates are the
 * Generate form snapshotted to `localStorage` so a set of params can be
 * recalled with one click. Ported from the web SPA's `generationTemplates.ts`,
 * but keyed on a desktop-only storage key so the two never share blobs.
 *
 * Media fields (`sourceImage` / `maskImage` / `controlImage`, plus video,
 * keyframe, and audio conditioning files) hold browser-local base64 that
 * we deliberately do NOT persist — a saved template records only a
 * human-facing `mediaReferences` hint so the user knows to re-select bytes.
 */
import type { GenerateForm } from "./generateForm";
import { createUuid } from "@studio/lib/id";

/** Storage key — MUST differ from the web SPA's `mold.generation.templates.v1`. */
export const GENERATION_TEMPLATES_STORAGE_KEY = "mold.desktop.generation.templates.v1";

export type GenerationTemplateSort = "updated-desc" | "updated-asc" | "name-asc" | "name-desc";

/** Human-facing labels for media that a template could not persist. */
export type GenerationTemplateMediaField =
  "source" | "mask" | "control" | "sourceVideo" | "keyframes" | "editImages" | "audioFile";

export interface GenerationTemplate {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  /** The saved form with base64 media stripped to null / empty. */
  form: GenerateForm;
  /** Which media slots were populated when saved — the user must re-select them. */
  mediaReferences: GenerationTemplateMediaField[];
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
    return parsed.filter(isTemplate);
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
