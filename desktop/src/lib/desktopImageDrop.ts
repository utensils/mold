import { generationCapabilitiesForFamily } from "./capabilities";
import type { ModelEntry, OutputMetadata } from "./api/types";
import { applyMetadataToForm, type GenerateForm } from "./generateForm";

/** A local still read by the native desktop backend after an OS file drop. */
export interface DesktopImageImport {
  filename: string;
  base64: string;
  width: number;
  height: number;
  /** Content key used by the native app to reopen the original local path. */
  sha256?: string;
  metadata: OutputMetadata | null;
}

export interface DesktopImageDropResult {
  attached: boolean;
  metadataApplied: boolean;
}

/**
 * Apply one native file drop to Generate. Embedded Mold metadata restores the
 * complete serialized generation first; the dropped still then becomes the
 * source for single-image families or the Target for Qwen edit families.
 */
export function applyDesktopImageDrop(
  form: GenerateForm,
  image: DesktopImageImport,
  models: ModelEntry[] = [],
): DesktopImageDropResult {
  const metadataApplied = image.metadata != null;
  if (image.metadata) applyMetadataToForm(form, image.metadata, models);

  const caps = generationCapabilitiesForFamily(form.family, form.model);
  if (!caps.supportsImg2img) return { attached: false, metadataApplied };

  if (caps.sourceImageMode !== "single") {
    form.sourceImage = null;
    form.sourceImageName = null;
    form.imageAttachments = [image.base64];
  } else if (caps.sourceImageMode === "single") {
    form.imageAttachments = [];
    form.sourceImage = image.base64;
    form.sourceImageName = image.filename;
    form.sourceFit = { mode: "lanczos-resize" };
  } else {
    return { attached: false, metadataApplied };
  }
  form.sourceImageWidth = image.width;
  form.sourceImageHeight = image.height;

  return { attached: true, metadataApplied };
}
