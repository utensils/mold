import {
  imageAttachmentChoices,
  replaceEndpointKeyframe,
  type SourceAttachmentIntent,
} from "@studio/lib/sourceAttachmentChoices";
import type { GenerationCapabilities } from "./capabilities";
import { appendMinimaxH3PickedMediaReference } from "@studio/lib/minimaxH3PickedMedia";
import { emptyMinimaxH3AuthoringState } from "@studio/lib/minimaxH3Authoring";
import { watch, isReactive } from "vue";
import { normalizeInputImage } from "@studio/lib/inputImage";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { resolveDropTarget } from "@studio/lib/imageDropRouting";
import { MINIMAX_H3_MAX_REFERENCES } from "@studio/lib/minimaxH3Authoring";
import { applyDropToForm } from "./desktopImageDrop";
import { blobToBase64 } from "@studio/lib/base64";
import { sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import { generationCapabilitiesForForm } from "./capabilities";
import { cloneGenerateForm, type GenerateForm } from "./generateForm";
import { type UseAsSourceOutcome } from "./gallery/useAsSource";

/** Stage on a private draft; commit only while both origin and destination remain current. */
export async function attachRetainedSource(
  form: GenerateForm,
  filename: string,
  kind: string,
  read: () => Promise<Blob>,
  stillCurrent: () => boolean,
  attachment?: SourceAttachmentIntent,
  resolvedCapabilities?: GenerationCapabilities,
): Promise<UseAsSourceOutcome> {
  const before = JSON.stringify(form);
  const staged = cloneGenerateForm(form);
  const caps =
    resolvedCapabilities ??
    generationCapabilitiesForForm(
      form.family,
      form.model,
      form.pipeline,
      form.guidanceCapabilities,
      form.sourceImageCapability,
      form.recipeCapabilities,
    );
  const plan = sourceMediaPlan(caps);
  if (attachment && (kind !== "image" || !imageAttachmentChoices(caps).includes(attachment)))
    return { ok: false, error: "The selected style does not accept that video endpoint." };
  if (
    (kind === "image" && plan.kind === "none") ||
    (kind === "video" && plan.kind !== "h3-references" && !caps.supportsSourceVideo) ||
    (kind === "audio" && plan.kind !== "h3-references" && !caps.supportsAudioInput) ||
    !["image", "video", "audio"].includes(kind)
  ) {
    return {
      ok: false,
      error: "Choose a style that supports this source media before attaching it.",
    };
  }
  let changed = false;
  const stop = isReactive(form)
    ? watch(
        form,
        () => {
          changed = true;
        },
        { deep: true, flush: "sync" },
      )
    : () => {};
  const current = () => !changed && stillCurrent() && JSON.stringify(form) === before;
  try {
    if (!current())
      return {
        ok: false,
        error: "The draft or source machine changed. Try attaching the source again.",
      };
    const blob = await read();
    if (!current())
      return {
        ok: false,
        error: "The draft or source machine changed. Try attaching the source again.",
      };
    if (plan.kind === "h3-references" && kind !== "image")
      staged.h3Authoring = await appendMinimaxH3PickedMediaReference(
        staged.h3Authoring ?? emptyMinimaxH3AuthoringState(),
        new File([blob], filename, {
          // Retained content is deliberately served as octet-stream. The typed
          // importer probes the actual media before admitting the reference.
          type: kind === "audio" ? "audio/wav" : "video/mp4",
        }),
      );
    else if (kind === "audio") staged.audioFile = { filename, base64: await blobToBase64(blob) };
    else if (kind === "video") staged.sourceVideo = { filename, base64: await blobToBase64(blob) };
    else {
      const prepared = await normalizeInputImage(await blobToBase64(blob), filename);
      const dimensions = imageDimensionsFromBase64(prepared.base64) ?? {
        width: form.width,
        height: form.height,
      };
      const keyframe =
        attachment &&
        caps.supportsKeyframes &&
        !caps.supportsEndFrame &&
        plan.kind !== "h3-boundaries";
      const endpointTarget = attachment
        ? plan.kind === "h3-boundaries"
          ? attachment === "first"
            ? "h3-first"
            : "h3-last"
          : attachment === "first"
            ? "source"
            : "end"
        : null;
      const routed = resolveDropTarget(plan, endpointTarget, {
        hasSource: Boolean(staged.sourceImage),
        referenceCount: staged.imageAttachments.length,
        identityVisible: false,
        openingVisible: false,
        h3FirstPresent: Boolean(staged.h3Authoring?.firstFrame),
        h3ReferenceCount: staged.h3Authoring?.references.length ?? 0,
        h3ReferenceMax: MINIMAX_H3_MAX_REFERENCES,
        lastWrite: staged.exclusiveWell ?? null,
        refusalReason: caps.referenceImagesReason,
      });
      if (keyframe) {
        if (
          attachment === "last" &&
          staged.sourceImage &&
          !staged.keyframes.some((frame) => frame.frame === 0)
        ) {
          staged.keyframes.unshift({
            frame: 0,
            image: {
              filename: staged.sourceImageName ?? "first-frame.png",
              base64: staged.sourceImage,
            },
          });
        }
        staged.keyframes = replaceEndpointKeyframe(staged.keyframes, staged.frames, attachment!, {
          filename: prepared.filename,
          base64: prepared.base64,
        });
        staged.sourceImage = null;
        staged.sourceImageName = null;
        staged.maskImage = null;
      } else {
        if (typeof routed !== "string") return { ok: false, error: routed.refused };
        const outcome = await applyDropToForm(staged, routed, {
          filename: prepared.filename,
          base64: prepared.base64,
          ...dimensions,
          metadata: null,
        });
        if (!outcome.ok) return outcome;
        if (routed === "source") staged.maskImage = null;
      }
    }
    if (!current())
      return {
        ok: false,
        error: "The draft or source machine changed. Try attaching the source again.",
      };
    stop();
    // Only attachment fields cross back; all unrelated authoring state stays intact.
    form.sourceImage = staged.sourceImage;
    form.sourceImageName = staged.sourceImageName;
    form.sourceImageWidth = staged.sourceImageWidth;
    form.sourceImageHeight = staged.sourceImageHeight;
    form.maskImage = staged.maskImage;
    form.exclusiveWell = staged.exclusiveWell;
    form.endFrame = staged.endFrame;
    form.keyframes = staged.keyframes;
    form.sourceFit = staged.sourceFit;
    form.sourceVideo = staged.sourceVideo;
    form.audioFile = staged.audioFile;
    form.imageAttachments = staged.imageAttachments;
    if (staged.h3Authoring) form.h3Authoring = staged.h3Authoring;
    return { ok: true, message: "Source attached to your draft" };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  } finally {
    stop();
  }
}
