import type { GenerateForm, PickedImage } from "./generateForm";
import { defaultSourceFitPolicy } from "@studio/lib/sourceFit";

/** One source-attachment policy shared by uploads, gallery picks, and Library. */
export function attachPickedImage(form: GenerateForm, picked: PickedImage): void {
  form.sourceImage = picked.base64;
  form.sourceImageName = picked.filename || null;
  form.sourceFit = defaultSourceFitPolicy();
}

/** Video outputs become LTX source-video bytes, including cross-host reuse. */
export function attachPickedVideo(form: GenerateForm, picked: PickedImage): void {
  form.sourceVideo = picked;
}
