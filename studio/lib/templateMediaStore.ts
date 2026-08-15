import {
  deleteDraftMedia,
  getDraftMedia,
  putDurableMedia,
} from "./draftMediaStore";
import type { GenerationReference } from "./generationReferences";

export type GenerationTemplateSourceField =
  | "sourceImage"
  | "imageAttachments"
  | "h3FirstFrame"
  | "h3LastFrame"
  | "h3References";

export interface GenerationTemplateMediaAsset {
  assetId: string;
  field: GenerationTemplateSourceField;
  index?: number;
  filename: string;
  kind?: "upload" | "gallery";
  width?: number | null;
  height?: number | null;
  mime?: string | null;
  sha256?: string | null;
  /** Descriptor-only H3 reference shape. Media bytes remain in IndexedDB. */
  reference?: GenerationReference;
}

export interface GenerationTemplateMediaInput
  extends Omit<GenerationTemplateMediaAsset, "assetId"> {
  base64: string;
}

export interface StoredTemplateMedia extends GenerationTemplateMediaAsset {
  draftId: string;
  base64: string;
}

export interface HydratedGenerationTemplateMedia
  extends GenerationTemplateMediaAsset {
  base64: string;
}

export interface TemplateMediaPersistence {
  put(record: StoredTemplateMedia): Promise<boolean>;
  get(assetId: string): Promise<StoredTemplateMedia | null>;
  delete(assetId: string): Promise<void>;
}

const indexedDbPersistence: TemplateMediaPersistence = {
  put: putDurableMedia,
  get: (assetId) => getDraftMedia<StoredTemplateMedia>(assetId),
  delete: deleteDraftMedia,
};

function assetIdFor(
  templateId: string,
  input: GenerationTemplateMediaInput,
): string {
  const suffix =
    input.field === "sourceImage"
      ? "source"
      : input.field === "imageAttachments"
        ? `attachment-${input.index ?? 0}`
        : input.field === "h3FirstFrame"
          ? "h3-first-frame"
          : input.field === "h3LastFrame"
            ? "h3-last-frame"
            : `h3-reference-${input.index ?? 0}`;
  return `template:${templateId}:${suffix}`;
}

export async function persistGenerationTemplateMedia(
  templateId: string,
  inputs: readonly GenerationTemplateMediaInput[],
  persistence: TemplateMediaPersistence = indexedDbPersistence,
): Promise<GenerationTemplateMediaAsset[]> {
  const stored: StoredTemplateMedia[] = inputs.map((input) => {
    const assetId = assetIdFor(templateId, input);
    return { ...input, assetId, draftId: assetId };
  });
  const written: string[] = [];
  try {
    for (const record of stored) {
      if (!(await persistence.put(record))) {
        throw new Error("Template media storage is unavailable.");
      }
      written.push(record.assetId);
    }
  } catch (error) {
    await Promise.allSettled(
      written.map((assetId) => persistence.delete(assetId)),
    );
    throw error;
  }
  return stored.map(
    ({ base64: _base64, draftId: _draftId, ...asset }) => asset,
  );
}

export async function hydrateGenerationTemplateMedia(
  assets: readonly GenerationTemplateMediaAsset[],
  persistence: TemplateMediaPersistence = indexedDbPersistence,
): Promise<{
  media: HydratedGenerationTemplateMedia[];
  missing: GenerationTemplateMediaAsset[];
}> {
  const resolved = await Promise.all(
    assets.map(async (asset) => {
      const stored = await persistence.get(asset.assetId).catch(() => null);
      return stored?.base64 ? { ...asset, base64: stored.base64 } : null;
    }),
  );
  return {
    media: resolved.filter(
      (entry): entry is HydratedGenerationTemplateMedia => entry !== null,
    ),
    missing: assets.filter((_, index) => resolved[index] === null),
  };
}

export async function deleteGenerationTemplateMedia(
  assets: readonly GenerationTemplateMediaAsset[],
  persistence: TemplateMediaPersistence = indexedDbPersistence,
): Promise<void> {
  await Promise.allSettled(
    assets.map((asset) => persistence.delete(asset.assetId)),
  );
}
