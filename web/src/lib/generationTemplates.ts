import type { GenerateFormState, SourceImageState } from "../types";
import { cloneTemplateForm } from "../composables/useGenerateForm";

export const GENERATION_TEMPLATES_STORAGE_KEY = "mold.generation.templates.v1";

export type GenerationTemplateSort =
  | "updated-desc"
  | "updated-asc"
  | "name-asc"
  | "name-desc";

export interface GenerationTemplateMediaReference {
  field:
    | "imageAttachments"
    | "maskImage"
    | "controlImage"
    | "audioFile"
    | "sourceVideo"
    | "keyframes";
  kind: SourceImageState["kind"] | "upload";
  filename: string;
  frame?: number;
}

export interface GenerationTemplate {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  form: GenerateFormState;
  /** Human-facing references for media that could not be persisted without
   * storing browser-local base64 blobs. Loading a template restores safe path
   * fields from `form`, but upload/gallery bytes must be re-selected. */
  mediaReferences: GenerationTemplateMediaReference[];
}

function now(): number {
  return Date.now();
}

function templateId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `tpl-${now()}-${Math.random().toString(36).slice(2)}`;
}

function normalizeName(name: string): string {
  return name.trim() || "Untitled template";
}

function mediaRef(
  field: GenerationTemplateMediaReference["field"],
  media: { kind: "upload" | "gallery"; filename: string } | null,
): GenerationTemplateMediaReference | null {
  if (!media) return null;
  return { field, kind: media.kind, filename: media.filename };
}

export function collectTemplateMediaReferences(
  form: GenerateFormState,
): GenerationTemplateMediaReference[] {
  return [
    ...form.imageAttachments.map<GenerationTemplateMediaReference>((image) => ({
      field: "imageAttachments",
      kind: image.kind,
      filename: image.filename,
    })),
    mediaRef("maskImage", form.maskImage),
    mediaRef("controlImage", form.controlImage),
    form.audioFile
      ? {
          field: "audioFile",
          kind: form.audioFile.kind,
          filename: form.audioFile.filename,
        }
      : null,
    form.sourceVideo
      ? {
          field: "sourceVideo",
          kind: form.sourceVideo.kind,
          filename: form.sourceVideo.filename,
        }
      : null,
    ...form.keyframes.map<GenerationTemplateMediaReference>((keyframe) => ({
      field: "keyframes",
      kind: keyframe.image.kind,
      filename: keyframe.image.filename,
      frame: keyframe.frame,
    })),
  ].filter((ref): ref is GenerationTemplateMediaReference => ref !== null);
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

function isTemplate(value: unknown): value is GenerationTemplate {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<GenerationTemplate>;
  return (
    typeof candidate.id === "string" &&
    typeof candidate.name === "string" &&
    typeof candidate.createdAt === "number" &&
    typeof candidate.updatedAt === "number" &&
    !!candidate.form &&
    Array.isArray(candidate.mediaReferences)
  );
}

function writeTemplates(templates: GenerationTemplate[]) {
  localStorage.setItem(
    GENERATION_TEMPLATES_STORAGE_KEY,
    JSON.stringify(templates),
  );
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
        return a.name.localeCompare(b.name);
      case "name-desc":
        return b.name.localeCompare(a.name);
      case "updated-desc":
      default:
        return b.updatedAt - a.updatedAt;
    }
  });
  return sorted;
}

export function loadGenerationTemplates(
  sort: GenerationTemplateSort = "updated-desc",
): GenerationTemplate[] {
  return sortGenerationTemplates(
    parseTemplates(localStorage.getItem(GENERATION_TEMPLATES_STORAGE_KEY)),
    sort,
  );
}

export function saveGenerationTemplate(
  name: string,
  form: GenerateFormState,
): GenerationTemplate {
  const timestamp = now();
  const template: GenerationTemplate = {
    id: templateId(),
    name: normalizeName(name),
    createdAt: timestamp,
    updatedAt: timestamp,
    form: cloneTemplateForm(form),
    mediaReferences: collectTemplateMediaReferences(form),
  };
  const templates = [template, ...loadGenerationTemplates()];
  writeTemplates(templates);
  return template;
}

export function renameGenerationTemplate(
  id: string,
  name: string,
): GenerationTemplate | null {
  const templates = loadGenerationTemplates();
  let renamed: GenerationTemplate | null = null;
  const next = templates.map((template) => {
    if (template.id !== id) return template;
    renamed = {
      ...template,
      name: normalizeName(name),
      updatedAt: now(),
    };
    return renamed;
  });
  writeTemplates(next);
  return renamed;
}

export function deleteGenerationTemplate(id: string): void {
  writeTemplates(
    loadGenerationTemplates().filter((template) => template.id !== id),
  );
}

export function searchGenerationTemplates(
  query: string,
  sort: GenerationTemplateSort = "updated-desc",
): GenerationTemplate[] {
  const q = query.trim().toLowerCase();
  const templates = loadGenerationTemplates(sort);
  if (!q) return templates;
  return templates.filter((template) =>
    [
      template.name,
      template.form.model,
      template.form.prompt,
      template.form.negativePrompt,
      ...template.form.loras.map((lora) => lora.path),
      ...template.mediaReferences.map((ref) => ref.filename),
    ]
      .join(" ")
      .toLowerCase()
      .includes(q),
  );
}
