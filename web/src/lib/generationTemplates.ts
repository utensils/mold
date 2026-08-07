import type { GenerateFormState, SourceImageState } from "../types";
import { cloneTemplateForm } from "../composables/useGenerateForm";
import { createUuid } from "@studio/lib/id";
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
  type MinimaxH3BoundaryImage,
} from "@studio/lib/minimaxH3Authoring";

export const GENERATION_TEMPLATES_STORAGE_KEY = "mold.generation.templates.v1";

export type GenerationTemplateSort =
  "updated-desc" | "updated-asc" | "name-asc" | "name-desc";

export interface GenerationTemplateMediaReference {
  field:
    | "imageAttachments"
    | "maskImage"
    | "controlImage"
    | "audioFile"
    | "sourceVideo"
    | "keyframes"
    | "h3FirstFrame"
    | "h3LastFrame"
    | "h3References";
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
  /** Human-facing references for media present when the template was saved.
   * Source images also have durable `mediaAssets`; unsupported auxiliary
   * media keeps this metadata so the UI can request re-selection. */
  mediaReferences: GenerationTemplateMediaReference[];
  /** Durable client-local source snapshots. Legacy templates omit this. */
  mediaAssets?: GenerationTemplateMediaAsset[];
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
    form.h3Authoring?.firstFrame
      ? {
          field: "h3FirstFrame",
          kind: "upload",
          filename: form.h3Authoring.firstFrame.filename,
        }
      : null,
    form.h3Authoring?.lastFrame
      ? {
          field: "h3LastFrame",
          kind: "upload",
          filename: form.h3Authoring.lastFrame.filename,
        }
      : null,
    ...(
      form.h3Authoring?.references ?? []
    ).map<GenerationTemplateMediaReference>(({ reference }) => ({
      field: "h3References",
      kind: "upload",
      filename: reference.provenance?.name || `${reference.kind} reference`,
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

/** Save the template only after source bytes are durable in IndexedDB. */
export async function saveGenerationTemplateWithMedia(
  name: string,
  form: GenerateFormState,
  persistence?: TemplateMediaPersistence,
): Promise<GenerationTemplate> {
  const timestamp = now();
  const id = templateId();
  const h3 = form.h3Authoring ?? emptyMinimaxH3AuthoringState();
  const mediaAssets = await persistGenerationTemplateMedia(
    id,
    [
      ...form.imageAttachments.map((image, index) => ({
        field: "imageAttachments" as const,
        index,
        filename: image.filename,
        kind: image.kind,
        width: image.width,
        height: image.height,
        mime: image.mime,
        base64: image.base64,
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
              ...(h3.firstFrame.sha256 !== undefined
                ? { sha256: h3.firstFrame.sha256 }
                : {}),
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
              ...(h3.lastFrame.sha256 !== undefined
                ? { sha256: h3.lastFrame.sha256 }
                : {}),
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
                filename:
                  reference.provenance?.name ||
                  `${reference.kind} reference ${index + 1}`,
                kind: "upload" as const,
                mime: reference.mime_type,
                reference: redactGenerationReference(reference),
                base64: reference.media.data,
              },
            ]
          : [],
      ),
    ],
    persistence,
  );
  const template: GenerationTemplate = {
    id,
    name: normalizeName(name),
    createdAt: timestamp,
    updatedAt: timestamp,
    form: cloneTemplateForm(form),
    mediaReferences: collectTemplateMediaReferences(form),
    ...(mediaAssets.length ? { mediaAssets } : {}),
  };
  try {
    writeTemplates([template, ...loadGenerationTemplates()]);
  } catch (error) {
    await deleteGenerationTemplateMedia(mediaAssets, persistence);
    throw error;
  }
  return template;
}

export async function hydrateGenerationTemplate(
  template: GenerationTemplate,
  persistence?: TemplateMediaPersistence,
): Promise<{ form: GenerateFormState; sourceMissing: boolean }> {
  const form = JSON.parse(JSON.stringify(template.form)) as GenerateFormState;
  // Legacy byte-free attachment markers must not masquerade as usable source
  // images. New templates rebuild the ordered list from durable assets below.
  form.imageAttachments = [];
  const assets = template.mediaAssets ?? [];
  const { media, missing } = await hydrateGenerationTemplateMedia(
    assets,
    persistence,
  );
  const attachments = media
    .filter((asset) => asset.field === "imageAttachments")
    .sort((left, right) => (left.index ?? 0) - (right.index ?? 0))
    .map<SourceImageState>((asset) => ({
      kind: asset.kind ?? "upload",
      filename: asset.filename,
      base64: asset.base64,
      ...(asset.width != null ? { width: asset.width } : {}),
      ...(asset.height != null ? { height: asset.height } : {}),
      ...(asset.mime != null ? { mime: asset.mime } : {}),
    }));
  form.imageAttachments = attachments;
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
  const restoredH3References = media
    .filter((asset) => asset.field === "h3References" && asset.reference)
    .sort((left, right) => (left.index ?? 0) - (right.index ?? 0));
  for (const asset of restoredH3References) {
    const index = asset.index ?? 0;
    form.h3Authoring.references[index] = {
      reference: {
        ...asset.reference!,
        media: { authority: "inline", data: asset.base64 },
      },
      draftId: asset.assetId,
    };
  }
  const expected = assets.filter(
    (asset) => asset.field === "imageAttachments",
  ).length;
  const h3ReferenceCount = template.mediaReferences.filter(
    (reference) => reference.field === "h3References",
  ).length;
  const h3MediaMissing =
    (template.mediaReferences.some(
      (reference) => reference.field === "h3FirstFrame",
    ) &&
      !first) ||
    (template.mediaReferences.some(
      (reference) => reference.field === "h3LastFrame",
    ) &&
      !last) ||
    restoredH3References.length !== h3ReferenceCount ||
    missing.some((asset) => asset.field.startsWith("h3"));
  return {
    form,
    sourceMissing:
      (template.mediaReferences.some(
        (reference) => reference.field === "imageAttachments",
      ) &&
        (expected === 0 ||
          attachments.length !== expected ||
          missing.some((asset) => asset.field === "imageAttachments"))) ||
      h3MediaMissing,
  };
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

export async function deleteGenerationTemplateWithMedia(
  template: GenerationTemplate,
  persistence?: TemplateMediaPersistence,
): Promise<void> {
  deleteGenerationTemplate(template.id);
  await deleteGenerationTemplateMedia(template.mediaAssets ?? [], persistence);
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
